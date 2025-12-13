# Deteccao de Queda para o projeto Lucas
# Copyright (C) <year>  <name of author>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Heloísa Dias Viotto
#
# =============================================================================
#  Header
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier

PATH = os.path.expanduser("~/fall_detection_video_dataset")

KYP_NAMES = [ "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle" ]

# =============================================================================
#  Funções
# =============================================================================

# Classe MLP
class MLP(nn.Module):
    def __init__(self, enters, outs):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(enters, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, outs)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)


    def forward(self, x):
        return self.mlp(x)
    
    # Loop de treino
    def train_loop(self, train_loader, epochs, device):
        self.train()
        self.to(device)

        for epoch in range(epochs):
            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)
    
                self.optimizer.zero_grad()
                outputs = self(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()


    # Loop de Avaliação
    def evaluate(self, data_loader, device):
        self.eval()
        self.to(device)

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = self(features)

                proba = torch.softmax(outputs, dim=1)

                _, predicted = torch.max(outputs, 1)
    
                all_labels.append(labels.cpu())
                all_preds.append(predicted.cpu())
                all_probs.append(proba.cpu())

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)

        return all_labels, all_preds


#Estrutura de dados utilizado com o MLP
class VideoDataset (Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)
        
    def __getitem__ (self, idx):
        return self.data[idx], self.labels[idx]


# Filtra os atributos do csv  
def filter (list_filter, csv_all):

    df1 = pd.read_csv(list_filter)
    df2 = pd.read_csv(csv_all)

    df = df1.merge(df2, on=["video", "frame"], how="left")

    # Transforma a coluna video em número
    df['video_int'] = df['video'].astype('category').cat.codes
    df = df.drop(columns=["full", "video", "type", "bbox_x1", "bbox_x2", "bbox_y1", "bbox_y2", "len_factor", "left_body_x", "left_body_y", "right_body_x", "right_body_y"])

    df["group"] = df["video_int"].astype(str) + "_" + df["fall"].astype(str)
    df = df.drop(columns=["video_int"])

    # Retira as colunas de confiabilidade
    for col in df.columns:
        if 'conf' in col:
            df = df.drop(columns=col)

    return df.dropna()

# Realiza a cross_validation do mlp
def mlp (feat, alvo, group, gkf, scaler):

    # Define a entrada
    input = feat.shape[1]

    # Define a estrutura de dados de métricas
    metrics = {"acc": [], "ses": [], "esp": [], "f1": []}
        
    for fold, (train, val) in enumerate(gkf.split(feat, alvo, group)):
        
        # Seleciona as splits
        X_train, X_val = feat.iloc[train], feat.iloc[val]
        y_train, y_val = alvo.iloc[train], alvo.iloc[val]

        # Normaliza os dados
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Seleciona o dataset e dataloader de treino
        train_dataset = VideoDataset(X_train, y_train)
        train_dataloader = DataLoader (train_dataset, batch_size=32, shuffle=True)

        # Seleciona o dataset e dataloader de test
        test_dataset = VideoDataset (X_val, y_val)
        test_dataloader = DataLoader (test_dataset, batch_size = 32, shuffle=True)

        # Instancia o modelo
        model = MLP(input, 2)

        # Treina o modelo por 10 épcocas
        model.train_loop (train_dataloader, 10, 'cpu')

        # Avalia o modelo
        labels, preds = model.evaluate (test_dataloader, 'cpu')

        # Gera:
        #   tn: True negative (verdadeiro negativo)
        #   fp: false positive (falso positivo)
        #   fn: false negative (falso negativo)
        #   tp: true positive (verdadeiro positivo)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()           

        # Calcula as métricas e adiciona na respectiva lista
        metrics["acc"].append((tp + tn)/(tp+tn+fp+fn))
        metrics["esp"].append(tn/(tn+fp))
        metrics["ses"].append(tp/(tp+fn))
        metrics["f1"].append((2*tp)/(2*tp + fp + fn))

    # Retorna as métricas
    return metrics

def xgboost (feat, alvo, group, gkf, scaler):

    # Define a estrutura de dados de métricas
    metrics = {"acc": [], "ses": [], "esp": [], "f1": []}

    # Loop
    for fold, (train, val) in enumerate(gkf.split(feat, alvo, group)):
        
        # Seleciona as splits
        X_train, X_val = feat.iloc[train], feat.iloc[val]
        y_train, y_val = alvo.iloc[train], alvo.iloc[val]

        # Normalização
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = XGBClassifier()

        # Treina o modelo
        model.fit(X_train, y_train)

        # Realiza a predição/avaliação
        proba = model.predict_proba(X_val)[:,1]

        thr = 0.35
        y_pred = (proba >= thr).astype(int)

        # Gera:
        #   tn: True negative (verdadeiro negativo)
        #   fp: false positive (falso positivo)
        #   fn: false negative (falso negativo)
        #   tp: true positive (verdadeiro positivo)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()           

        # Calcula e armazena as métricas
        metrics["acc"].append((tp + tn)/(tp+tn+fp+fn))
        metrics["esp"].append(tn/(tn+fp))
        metrics["ses"].append(tp/(tp+fn))
        metrics["f1"].append((2*tp)/(2*tp + fp + fn))

    return metrics



# Realiza o cross_validation para outros modelos (KNN, Random Forest e SVM)
def models (feat, alvo, group, gkf, scaler, model_name):

    metrics = {"acc": [], "ses": [], "esp": [], "f1": []}
    
    for fold, (train, val) in enumerate(gkf.split(feat, alvo, group)):

        # Instancia o modelo de acordo com o nome
        if model_name.lower() == 'svm':
            model = SVC()

        elif model_name.lower() == 'rf':
            model = RandomForestClassifier()

        else:
            model = KNeighborsClassifier()

        # Seleciona as splits
        X_train, X_val = feat.iloc[train], feat.iloc[val]
        y_train, y_val = alvo.iloc[train], alvo.iloc[val]

        # Normalização
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Treina o modelo
        model.fit(X_train, y_train)

        # Realiza a predição/avaliação
        y_pred = model.predict(X_val)
            
        # Gera:
        #   tn: True negative (verdadeiro negativo)
        #   fp: false positive (falso positivo)
        #   fn: false negative (falso negativo)
        #   tp: true positive (verdadeiro positivo)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()           

        # Calcula e armazena as métricas
        metrics["acc"].append((tp + tn)/(tp+tn+fp+fn))
        metrics["esp"].append(tn/(tn+fp))
        metrics["ses"].append(tp/(tp+fn))
        metrics["f1"].append((2*tp)/(2*tp + fp + fn))

    return metrics


# Realiza a validação cruzada 
def cross_validation (df, model_name):

    feat = df.drop(columns=['fall'], axis=1)
    alvo = df['fall']
    group = df['group']

    gkf = GroupKFold(n_splits=5)
    scaler = StandardScaler()

    if model_name.lower() == 'mlp':
        return mlp(feat, alvo, group, gkf, scaler)

    if model_name.lower() == 'xgb':
        return xgboost(feat, alvo, group, gkf, scaler)

    return models (feat, alvo, group, gkf, scaler, model_name)


# =============================================================================
#  Main
# =============================================================================


"""
    SVM                   - python3 models.py svm
    Random Forest         - python3 models.py rf
    Multilayer Perceptron - python3 models.py mlp
    5-NN                  - python3 models.py knn

    Gerar CSV de pontos   - python3 models.py MODEL gerar 
"""

if __name__ == "__main__":

    model_name = sys.argv[1]
    list_filter = sys.argv[2]
    csv_all = sys.argv[3]

    df = filter(list_filter, csv_all)
    
    metrics = cross_validation(df, model_name)

    print(f"========== {model_name.upper()} {list_filter} ==========")
    print(f"Acurácia: {round(stat.mean(metrics['acc']), 4)} ({round(stat.stdev(metrics['acc']), 4)})")
    print(f"Sensibilidade: {round(stat.mean(metrics['ses']), 4)} ({round(stat.stdev(metrics['ses']), 4)})")
    print(f"Especificidade: {round(stat.mean(metrics['esp']), 4)} ({round(stat.stdev(metrics['esp']), 4)})")
    print(f"F1-score: {round(stat.mean(metrics['f1']), 4)} ({round(stat.stdev(metrics['f1']), 4)})")
    

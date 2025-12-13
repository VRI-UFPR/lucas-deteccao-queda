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
# Código baseado no artigo Fall detection system for monitoring elderly people using YOLOv7-pose detection model

# =============================================================================
#  Header
# =============================================================================

import pandas as pd
import sys
import statistics as stat

# =============================================================================
#  Funções
# =============================================================================

def filter (list_filter, csv_all):
    """
        Filtra os campos de um arquivo CSV

        Parameters:
            csv_name (path): endereço do arquivo CSV com as posições

        Returns:
            (vetor de dicionário): dados filtrados do CSV
    """

    df1 = pd.read_csv(list_filter)
    df2 = pd.read_csv(csv_all)

    df = df1.merge(df2, on=["video", "frame"], how="left")
   
    keep = ["video", "frame", "person_id", "len_factor",
            "bbox_x1", "bbox_x2", "bbox_y1", "bbox_y2",
            "left_body_y", "right_body_y", "left_shoulder_y",
            "right_shoulder_y", "left_ankle_y", "right_ankle_y", "fall"]

    # Seleciona os pontos para manter
    df_filter = df[keep]
    df_filter = df_filter.dropna()
    
    return df_filter


def detect_fall (df_person):
    """
        Detecta uma queda de acordo com o artigo

        Parameters:
            df: numero de positivos

        Returns:
            caiu (boolean): true para caiu e false para não caiu
    """

    for _, row in df_person.iterrows():

        if row["left_shoulder_y"] > row["left_ankle_y"] - row["len_factor"]:
            a = True
        else:
            a = False

        if row["left_body_y"] > row["left_ankle_y"] - (row["len_factor"]/2):
            b = True
        else:
            b = False

        if row["left_shoulder_y"] > row["left_body_y"] - (row["len_factor"]/2):
            c = True
        else:
            c = False

        
        if row["right_shoulder_y"] > row["right_ankle_y"] - row["len_factor"]:
            d = True
        else:
            d = False

        if row["right_body_y"] > row["right_ankle_y"] - (row["len_factor"]/2):
            e = True
        else:
            e = False

        if row["right_shoulder_y"] > row["right_body_y"] - (row["len_factor"]/2):
            f = True
        else:
            f = False


        dx = int(row["bbox_x2"]) - int(row["bbox_x1"])
        dy = int(row["bbox_y2"]) - int(row["bbox_y1"])
        diff = dy - dx

        if (diff < 0):
            g = True
        else:
            g = False


        if (a and b and c) or (d and e and f) or g:
            return True
        
    return False


def classifier (df):
    """
        Classifica um conjunto de casos vindo de um arquivo CSV

        Parameters:
            df: numero de positivos

        Returns:
            total (int): numero total de casos
            tp (int): numero de positivos
            fp (int): numero de falso positivos
            tn (int): numero de negativos
            fn (int): numero de falsos negativos
    """

    total = 0
    tp = fp = fn = tn = 0
    for (video, fall_label), df_video in df.groupby(['video', 'fall']):
        for person_id, df_person in df_video.groupby('person_id'):
            x = detect_fall(df_person)

            total += 1

            if x and fall_label == 1:
                tp += 1
            elif x and fall_label == 0:
                fp += 1
            elif not x and fall_label == 1:
                fn += 1
            else:
                tn += 1

    return total, tp, fp, tn, fn


# Calcula as étricas (Acurácia, Sensibilidade, Especificidade e F1-Score)
def metrics (tp, fp, tn, fn):
    """
        Calcula a acuracia dado o numero de positivos, falsos positivo,
        negativos e o falsos negativos

        Parameters:
            tp(int): numero de positivos
            fp(int): numero de falsos positivos
            tn(int): numero de negativos
            fn(int): numero de falsos negativos

        Returns:
            float: acurácia
            float: sensibilidade
            float: especificidade
            float: f1-score
    """
    acc = (tp + tn) / (tp + tn + fp + fn)
    esp = tn/(tn+fp)
    ses = tp/(tp+fn)
    f1 = (2*tp)/(2*tp + fp + fn)

    return acc, ses, esp, f1

# =============================================================================
#  Main
# =============================================================================

"""
1) calcula as métricas usando o frames.csv já existente
python3 geom.py

2) calcula as métricas gerando o arquivo frames.csv a partir das imagens
python3 geom.py gerar
"""

if __name__ == "__main__":

    list_filter = sys.argv[1]
    csv_all = sys.argv[2]

    df = filter(list_filter, csv_all)
    
    total, tp, fp, tn, fn = classifier(df)
    acc, ses, esp, f1 = metrics (tp, fp, tn, fn)

    print(f"========== GEOMÉTRICO {list_filter} ==========")
    print("Acurácia: ", round(acc, 4))
    print("Sensibilidade: ", round(ses, 4))
    print("Especificidade: ", round(esp, 4))
    print("F1-score: ", round(f1, 4))

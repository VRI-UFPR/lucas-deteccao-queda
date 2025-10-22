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
# =============================================================================
#  Header
# =============================================================================

from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import math

PATH = "../database/"

KYP_NAMES = [ "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle" ]

# =============================================================================
#  Funções
# =============================================================================

def get_pose_bbox(keypoints):

    keypoints = np.array(keypoints, dtype=float)
    keypoints = keypoints[~np.isnan(keypoints).any(axis=1)]

    if len(keypoints) == 0:
        return None
    
    cx = np.mean(keypoints[:, 0])
    cy = np.mean(keypoints[:, 1])

    min_x, min_y = np.min(keypoints, axis=0)
    max_x, max_y = np.max(keypoints, axis=0)

    w = max_x - min_x
    h = max_y - min_y

    x_min = cx - w/2
    y_min = cy - h/2
    x_max = cx + w/2
    y_max = cy + h/2

    return x_min, y_min, x_max, y_max

def body (shoulder, hip):

    x = (shoulder[0] + hip[0]) / 2
    y = (shoulder[1] + hip[1]) / 2

    return x, y

def len_factor_from_pose(keypoints):

    try:
        left_shoulder = keypoints[5]
        left_hip = keypoints[11]
        right_shoulder = keypoints[6]
        right_hip = keypoints[12]
    except IndexError:
        return np.nan
    
    left_body_x, left_body_y = body(left_shoulder, left_hip)

    len_factor = np.sqrt((left_shoulder[1] - left_body_y)**2 + (left_shoulder[0] - left_body_x)**2)
    
    return len_factor



def generate_frames_csv (name):

    model = YOLO('yolo11n-pose.pt')

    rows = []
    for d0 in os.listdir(PATH):

        path = os.path.join(PATH, d0)

        for d1 in os.listdir(path):

            path0 = os.path.join(path, d1)

            for d2 in os.listdir(path0):
                
                if d2 != 'frames':
                    continue

                path1 = os.path.join(path0, d2)

                for d3 in os.listdir(path1):

                    path2 = os.path.join(path1, d3)
                    print(d3)

                    for img in os.listdir(path2):
                        idx_frame = int(img.split('.')[0].split('_')[1])
                        path3 = os.path.join(path2, img)

                        results = model(path3, verbose=False)
                    
                        kyp_xy = results[0].keypoints.xy.cpu().numpy().astype(float)
                        kyp_conf = results[0].keypoints.conf.cpu().numpy().astype(float)

                        for person_id, (xy, confs) in enumerate(zip(kyp_xy, kyp_conf)):
                            
                            bbox = get_pose_bbox(xy)
                            len_factor = len_factor_from_pose(xy)
                            left_body_x, left_body_y = body(xy[5], xy[11])
                            right_body_x, right_body_y = body(xy[6], xy[12])

                            if bbox is not None:
                                x1, y1, x2, y2 = bbox
                            else:
                                x1, y1, x2, y2 = np.nan

                            if 'no' in d0.lower():
                                fall = 0
                            else:
                                fall = 1
                            
                            row = {
                                "video": d3, 
                                "frame": idx_frame, 
                                "person_id": person_id,
                                "fall": fall,
                                "bbox_x1": x1,
                                "bbox_y1": y1,
                                "bbox_x2": x2,
                                "bbox_y2": y2,
                                "len_factor": len_factor,
                                "left_body_x": left_body_x,
                                "left_body_y": left_body_y,
                                "right_body_x": right_body_x,
                                "right_body_y": right_body_y,
                                }
                            
                            for i, name in enumerate(KYP_NAMES):
                                row[f"{name}_x"] = xy[i, 0]
                                row[f"{name}_y"] = xy[i, 1]
                                row[f"{name}_conf"] = confs[i]
                        
                            rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv(name, index=False)
    print(df)
    
    
        
def filter (csv_name):
    """
        Filtra os campos de um arquivo CSV

        Parameters:
            csv_name (path): endereço do arquivo CSV com as posições

        Returns:
            (vetor de dicionario): dados filtrados do CSV
    """

    df = pd.read_csv(csv_name)

    keep = ["video", "frame", "person_id", "len_factor",
            "bbox_x1", "bbox_x2", "bbox_y1", "bbox_y2",
            "left_body_y", "right_body_y", "left_shoulder_y",
            "right_shoulder_y", "left_ankle_y", "right_ankle_y", "fall"]

    df_filter = df[keep]
    
    return df_filter


def detect_fall (df_person):
    """
        Detecta uma queda

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
    for video, df_video in df.groupby('video'):
        for person_id, df_person in df_video.groupby('person_id'):
            x = detect_fall(df_person)

            total += 1

            if x and df_person["fall"].iloc[0] == 1:
                tp += 1
            elif x and df_person["fall"].iloc[0] == 0:
                fp += 1
            elif not x and df_person['fall'].iloc[0] == 1:
                fn += 1
            else:
                tn += 1

    return total, tp, fp, tn, fn
            

def accuracy (tp, fp, tn, fn):
    """
        Calcula a acuracia dado o numero de positivos, falsos positivo,
        negativos e o falsos negativos

        Parameters:
            tp(int): numero de positivos
            fp(int): numero de falsos positivos
            tn(int): numero de negativos
            fn(int): numero de falsos negativos

        Returns:
            float: porcentagem da acuracia de 0% a 100%
    """
    return ((tp + tn) / (tp + tn + fp + fn)) * 100

# =============================================================================
#  Main
# =============================================================================

"""
1) calcula a acuracia usando o frames.csv já existente
python3 geom.py

2) calcula a acuracia gerando o arquivo frames.csv a partir das imagens
python3 geom.py gerar
"""

if __name__ == "__main__":
    csv_name = 'frames.csv'

    if len(sys.argv) > 1:
        generate_frames_csv(csv_name)

    df = filter(csv_name)

    total, tp, fp, tn, fn = classifier(df)

    acc = accuracy(tp, fp, tn, fn)

    print(acc)
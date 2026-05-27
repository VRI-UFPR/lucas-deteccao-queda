# Deteccao de Queda para o projeto Lucas
# Copyright (C) 2026  Heloísa Dias Viotto
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
# Código baseado no artigo Fall detection system for monitoring elderly people using YOLOv7-pose detection model

# =============================================================================
#  Header
# =============================================================================

import numpy as np

KYP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# =============================================================================
#  Funções
# =============================================================================

# Seleciona a caixa da pessoa
def get_pose_bbox(keypoints):

    # Arrumando estrutura de dados
    keypoints = np.array(keypoints, dtype=float)
    keypoints = keypoints[~np.isnan(keypoints).any(axis=1)]

    # Caso não haja keypoints
    if len(keypoints) == 0:
        return None
    
    # Selecionando o centro dos eixos X e Y
    cx = np.mean(keypoints[:, 0])
    cy = np.mean(keypoints[:, 1])

    # Selecionando os pontos mínimos e máximos
    min_x, min_y = np.min(keypoints, axis=0)
    max_x, max_y = np.max(keypoints, axis=0)

    # Definindo altura e largura
    w = max_x - min_x
    h = max_y - min_y

    # Selecionando as bordas a partir do centro
    x_min = cx - w/2
    y_min = cy - h/2
    x_max = cx + w/2
    y_max = cy + h/2

    return x_min, y_min, x_max, y_max


# Selecionando o ponto BODY
def body (shoulder, hip):
    # Selecionando o meio do corpo a partir do meio do ombro e do quadril
    x = (shoulder[0] + hip[0]) / 2
    y = (shoulder[1] + hip[1]) / 2
    return x, y

# Aplicando a fórmula do len_factor do artigo
def len_factor_from_pose(keypoints):

    try:
        left_shoulder = keypoints[5]
        left_hip = keypoints[11]
        right_shoulder = keypoints[6]
        right_hip = keypoints[12]
    except IndexError:
        return np.nan
    
    left_body_x, left_body_y = body(left_shoulder, left_hip)

    len_factor = np.sqrt(
        (left_shoulder[1] - left_body_y)**2 
        + (left_shoulder[0] - left_body_x)**2
    )
    
    return len_factor

def detect_fall (row):
    """
        Detecta uma queda de acordo com o artigo

        Parameters:
            df: numero de positivos

        Returns:
            caiu (boolean): true para caiu e false para não caiu
    """

    

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
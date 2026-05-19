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

"""

Dado um nome de vídeo (argumento 1) e um caminho para salvar os frames, o vídeo é dividido 
em frames de 1 segundos e salavdos na pasta criado em PATH_TO_SAVE/VIDEO_NAME

"""

# Importações
import cv2
from ultralytics import YOLO
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


def mostra_pose(frame, keypoints, id):

    cores = [ (255,0,0), (0,255,0), (0,255,255), (0,0,255) ]
    cor = cores[id]

    # Iterar sobre cada ponto (nariz, olhos, ombros, etc.)
    for i, keypoint in enumerate(keypoints):
        x, y = int(keypoint[0]), int(keypoint[1])
        
        # Desenhar apenas se o ponto for válido (> 0)
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 5, cor, -1) # Círculo verde

    cv2.imshow("aaa", frame)

# =============================================================================
#  Main
# =============================================================================

video = cv2.VideoCapture('S_N_174.mp4')
# video = cv2.VideoCapture('S_N_550_resized.mp4')


model = YOLO('yolo11n-pose.pt')

frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Diminui o FPS para 2 a 4 FPS
    if frame_count < 6:
        frame_count += 1
        continue
    frame_count = 0

    # Executa a Yolo
    # results = model(frame, verbose=False)
    results = model.track(frame, show=True, tracker="bytetrack.yaml", device='cpu')
                    
    # Seleciona os pontos e confianças
    kyp_xy = results[0].keypoints.xy.cpu().numpy().astype(float)
    kyp_conf = results[0].keypoints.conf.cpu().numpy().astype(float)

    # Para cada pessoa no vídeo
    for person_id, (xy, confs) in enumerate(zip(kyp_xy, kyp_conf)):
        # mostra_pose(frame, xy, person_id)

        bbox = get_pose_bbox(xy)

        # Calcula o len_factor
        len_factor = len_factor_from_pose(xy)

        # Calcula os pontos do corpo
        left_body_x, left_body_y = body(xy[5], xy[11])
        right_body_x, right_body_y = body(xy[6], xy[12])

        # Abre o bbox (caixa)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = np.nan

        # Define dados básicos
        row = {
            "frame": 0, 
            "person_id": person_id,
            "fall": 0,
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
        
        # Relaciona nome com pontos
        for i, name in enumerate(KYP_NAMES):
            row[f"{name}_x"] = xy[i, 0]
            row[f"{name}_y"] = xy[i, 1]
            row[f"{name}_conf"] = confs[i]

        print(xy)
        # print(row)

    cv2.waitKey(0)





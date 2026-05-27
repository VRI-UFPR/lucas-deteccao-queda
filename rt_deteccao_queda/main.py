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
# =============================================================================
#  Header
# =============================================================================

import ufr
import cv2
from ultralytics import YOLO
from calc import get_pose_bbox, len_factor_from_pose, body, KYP_NAMES, detect_fall

# =============================================================================
#  Main
# =============================================================================

# 1. Abre o link da camera
video = ufr.Subscriber("@new video @@new mqtt @@coder msgpack @@topic camera @@host 177.153.62.174")

# 2. Abre o modelo Yolo
model = YOLO('yolo11n-pose.pt')

# 3. Loop principal
while True:
    # 3.1. Lê uma nova imagem e aplica a Yolo com rastreamento
    imagem = video.recv_cv_image()
    results = model.track(imagem, tracker="bytetrack.yaml", device='cpu')

    # 3.2. Plota o resultado e mostra imagem para depuracao
    imagem_com_resultado = results[0].plot()
    cv2.imshow("camera", imagem_com_resultado)
    cv2.waitKey(1)

    # 3.3. Verifica se existe alguma pessoa
    if results[0].boxes.id is None:
        continue

    # 3.4. Copia o resultado da Yolo para variaveis locais
    kyp_xy = results[0].keypoints.xy.cpu().numpy().astype(float)
    kyp_conf = results[0].keypoints.conf.cpu().numpy().astype(float)
    ids = results[0].boxes.id.cpu().numpy()

    # 3.5. Para cada pessoa, confere se houve queda
    for (person_id, xy, confs) in zip(ids, kyp_xy, kyp_conf):

        if person_id != 1:
            continue

        # Prepara a variavel Row para detectar a queda
        bbox = get_pose_bbox(xy)
        len_factor = len_factor_from_pose(xy)

        left_body_x, left_body_y = body(xy[5], xy[11])
        right_body_x, right_body_y = body(xy[6], xy[12])

        if bbox is not None:
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = np.nan

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
        for i, name in enumerate(KYP_NAMES):
            row[f"{name}_x"] = xy[i, 0]
            row[f"{name}_y"] = xy[i, 1]
            row[f"{name}_conf"] = confs[i]

        # Verifica se caiu
        if(detect_fall(row)):
            print(f"===== {row['person_id']} - CAIU")
        else:
            print(f"===== {row['person_id']} - PÉ")

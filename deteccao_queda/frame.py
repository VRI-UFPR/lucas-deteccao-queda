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

"""

Dado um nome de vídeo (argumento 1) e um caminho para salvar os frames, o vídeo é dividido 
em frames de 0.1 segundos e salavdos na pasta criado em PATH_TO_SAVE/VIDEO_NAME

"""

# Importações
import cv2
import sys
import os

# =============================================================================
#  Main
# =============================================================================

# Adquirindo argumentos
video_name = sys.argv[1]
path_to_save = sys.argv[2]

# Definições
video = cv2.VideoCapture(video_name)
fps = int(video.get(cv2.CAP_PROP_FPS))  # frames por segundo
intervalo = 1  # segundos
frame_intervalo = fps * intervalo

video_base = os.path.splitext(os.path.basename(video_name))[0]
save_dir = os.path.join(path_to_save, video_base)

os.makedirs(save_dir, exist_ok=True)

# Divisão do vídeo em frame
frame_id = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    if frame_id % frame_intervalo == 0:
        cv2.imwrite(f"{save_dir}/frame_{frame_id}.jpg", frame)
    
    frame_id += 1

# Fechamento do vídeo
video.release()

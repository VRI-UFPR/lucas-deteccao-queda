"""

Dado um nome de vídeo (argumento 1) e um caminho para salvar os frames, o vídeo é dividido 
em frames de 0.1 segundos e salavdos na pasta criado em PATH_TO_SAVE/VIDEO_NAME

"""

# Importações
import cv2
import sys
import os

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

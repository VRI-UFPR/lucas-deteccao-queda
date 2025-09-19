# This file is part of the Intercampi (https://github.com/VRI-UFPR/intercampi)
# Copyright (c) 2025 VRI
#  - Maite
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# =======================================================================================
#  Header
# =======================================================================================

import cv2
import csv
from ultralytics import YOLO

yolo_modelo_arquivo = "yolo11n-pose.pt"

# =======================================================================================
#  Funcoes
# =======================================================================================




# =======================================================================================
#  Main
# =======================================================================================

# 1. Carrega o modelo de pose
model = YOLO("yolo11n-pose.pt") 
video_path = "/home/mask/Documentos/escolhidos/fall chair/C_D_0096_resized.mp4"

# 2. Configura a escrita do vídeo de saída
cap = cv2.VideoCapture(video_path) #objeto para ler o video de entrada(permite ao programa acessar o video quadro por quadro)
if not cap.isOpened(): #verifica se o video foir aberto corretamente
    print("Erro: Não foi possível abrir o vídeo de entrada.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #obtem a largura do video original
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #obtem a altura do video original
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_video_path = "video_com_pontos.mp4" #define o nome do arquivo mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v') #define o codec de video para criar arquivos mp4
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height)) #cria um objeto para escrever o novo video

# 3. Configura a escrita do arquivo CSV de saída
output_csv_file = "keypoints_do_video.csv" #define o nome do arquivo csv
file = open(output_csv_file, 'w', newline='') #abre o arquivo csv no modo de escrita, newline é para evitar espaços em branco
writer = csv.writer(file) #cria o objeto que o programa usará para formatar os dados csv
writer.writerow(["frame_number", "person_id", "point_id", "x", "y"]) #escreve o cabeçalho do arquivo csv

# 4. Processa o vídeo frame por frame e salva os dados e o vídeo ******mudar para pegar a cada frame e nao o video inteiro
frame_number = 0 #inicializa uma variável que irá contar quantos arquivos foram processados
results = model(video_path, stream=True) #inicia o processo da YOLO no vídeo(o argumento 'stream=True' garante que o modelo processe os quadros um por um, ao invés de pegar o video inteiro)

for result in results: #loop para iterar cada quadro do video 
    frame_number += 1
    
    # Extrai os pontos e escreve no CSV
    if result.keypoints.xy is not None and len(result.keypoints.xy) > 0: #garante que os dados só vão ser processados se o modelo captar as keypoints do quadro
        for person_id, person_keypoints in enumerate(result.keypoints.xy): #itera sobre cada pessoa no quadro(o 'enumerate' fornece uma ID para cada pessoa do quadro)
            for point_id, point_coords in enumerate(person_keypoints.cpu().numpy()): #itera sobre cada um dos 17 pontos da pessoa, pegando o ID do ponto e suas coordenadas
                x, y = point_coords #desempacota a coordenada do ponto em duas variáveis x e y
                writer.writerow([frame_number, person_id + 1, point_id, x, y]) #escreve uma nova linha no csv com o numero do quadro, id da pessoa e as coordenadas
    
    # Desenha os pontos no frame e salva no novo vídeo
    annotated_frame = result.plot() #usa a função plot da Ultralytics para desenhar as keypoints e as caixas
    out_video.write(annotated_frame) #salva o quadro com as anotações no arquivo de saída 

    # Exibe o frame (opcional)
    cv2.imshow("Detecção de Keypoints", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #exibe o video em tempo real e permite que feche a janela usando 'q'
        break

#Montar código de classificação de uma queda


# 5. Finaliza o processo, fechando todos os arquivos
cap.release() #libera o obejeto que estava lendo o vídeo de entrada
out_video.release() #libera o objeto que estava escrevendo o video de saída 
file.close() # Fecha o arquivo CSV
cv2.destroyAllWindows() #fecha todas as janelas do OpenCV

print(f"Vídeo com os pontos salvo em '{output_video_path}'")
print(f"Dados dos keypoints salvos em '{output_csv_file}'")
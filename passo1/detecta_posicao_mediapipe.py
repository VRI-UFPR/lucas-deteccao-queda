#!/bin/python3

# =============================================================================
#  Header
# =============================================================================

# pip3 install opencv-python, mediapipe, numpy
#                                        2.2.6

import cv2
import mediapipe as mp
import numpy as np

# Inicializa o MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  # 0=leve, 1=padrão, 2=pesado (mais preciso)
)

# =============================================================================
#  Functions
# =============================================================================

# Função para calcular ângulos entre joints
def calculate_angle(a, b, c):
    a = np.array(a)  # Primeiro ponto
    b = np.array(b)  # Ponto central
    c = np.array(c)  # Último ponto
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# Função para determinar a posição da pessoa
def determine_position(landmarks):
    # Coordenadas dos landmarks relevantes
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    
    # Cálculo de ângulos
    torso_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    
    # Determinação da posição
    if torso_angle < 120:
        return "Em pé"
    elif 120 <= torso_angle < 160:
        return "Inclinado"
    else:
        return "Sentado/Agachado"

# =============================================================================
#  Main
# =============================================================================

video = cv2.VideoCapture(0)
# video = cv2.VideoCapture("video.mp4")

# Loop principal
while True:
    status, image = video.read()
    
    # Converte a imagem para RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Processa a imagem com MediaPipe
    results = pose.process(image)
    
    # Desenha os landmarks na imagem
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        # Desenha os landmarks e conexões
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        # Determina a posição
        position = determine_position(results.pose_landmarks.landmark)
        
        # Exibe a posição na tela
        cv2.putText(image, f"Posicao: {position}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Exibe coordenadas do quadril (centro do corpo)
        hip_x = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x + 
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2 * image.shape[1])
        hip_y = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y + 
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2 * image.shape[0])
        
        cv2.putText(image, f"Centro: ({hip_x}, {hip_y})", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Mostra a imagem
    cv2.imshow('MediaPipe Pose Detection', image)
    
    # Pressione 'q' para sair
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Libera os recursos
cv2.destroyAllWindows()

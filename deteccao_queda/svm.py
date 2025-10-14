from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

PATH = "../database/"

KYP_NAMES = [ "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle" ]

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
                            
                            if 'no' in d0.lower():
                                fall = 0
                            else:
                                fall = 1
                            
                            row = {
                                "video": d3, 
                                "frame": idx_frame, 
                                "person_id": person_id,
                                "fall": fall,
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

    df = pd.read_csv(csv_name)
    df['video_int'] = df['video'].astype('category').cat.codes
    df = df.drop(columns="video")

    return df

def cross_validation (df):

    feat = df.drop(columns=['fall'], axis=1)
    alvo = df['fall']
    group = df['video_int']

    model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))

    gkf = GroupKFold(n_splits=5)

    acc = []
    for fold, (train, val) in enumerate(gkf.split(feat, alvo, group)):
        X_train, X_val = feat.iloc[train], feat.iloc[val]
        y_train, y_val = alvo.iloc[train], alvo.iloc[val]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc.append(accuracy_score(y_val, y_pred))

    return acc


if __name__ == "__main__":

    csv_name = "frames-svm.csv"

    frames = bool(sys.argv[1])

    #if (frames):
    #    generate_frames_csv(csv_name)

    df = filter(csv_name)
    print(df)

    acc = cross_validation(df)
    print(acc)
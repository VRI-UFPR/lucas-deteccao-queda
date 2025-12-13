# Gera uma lista com osarquivos de devem ser acessados quando executar o experimetnos de X frames.

import os
import sys
import pandas as pd

PATH = os.path.expanduser("~/fall_detection_video_dataset")

def list_frames (frame, typefall):

    row = []

    # Pega a queda ou não queda
    for fall in os.listdir(PATH):
    
        path0 = os.path.join(PATH, fall)

        # PEga o tipo de queda (bed, chair, stand)
        for type_fall in os.listdir(path0):

            if typefall != "all" and type_fall != typefall:
                continue

            path1 = os.path.join(path0, type_fall)

            # Pega frame, mask_video ou video
            for resource in os.listdir(path1):

                if resource != "frames":
                    continue

                path2 = os.path.join(path1, resource)

                # Pega os vídeos
                for video in os.listdir(path2):

                    path3 = os.path.join(path2, video)
                    path4 = os.path.join(path3, "Frame_20")
    
                    # Frame
                    for img in os.listdir(path4):
                        
                        idx = int(img.split("_")[1].split(".")[0])

                        num = int(20 / frame)

                        if fall == "fall":
                            fall_idx = 1
                        else:
                            fall_idx = 0

                        if idx % num == 0:
                            row.append({"full": os.path.join(path4, img),
                                        "video": video,
                                        "frame": idx})


    df = pd.DataFrame(row)
    name = f"lists/{typefall}_{frame}.csv"
    df.to_csv(name, index = False)
    print(f"Salvo dataframe como {name}")
    print(df)

if __name__ == "__main__":

    list_frames(int(sys.argv[1]), sys.argv[2])



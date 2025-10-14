#!/bin/bash

falls=("fall" "no_fall")

for dir in "${falls[@]}"; do 
    
    path="../database/$dir"
    
    for fall in "$path"/*; do

        frame_path="$fall/frames"
        video_path="$fall/video"

        for video in "$video_path"/*; do
        
            video_name="${video##*/}"
            destine="$frame_path/$video_name/"
        
            echo "VIDEO_NAME=$video_name" 

            python3 frame.py $video $frame_path        

        done
    done

done
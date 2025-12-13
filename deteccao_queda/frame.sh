#!/bin/bash

falls=("fall" "no_fall")
#frames=(1 5 10 15 20)
frames=(20)

for frame in "${frames[@]}"; do

	echo "====== FRAME $frame"

	for dir in "${falls[@]}"; do 
    
    		path="$HOME/fall_detection_video_dataset/$dir"
    
    	for fall in "$path"/*; do

        	frame_path="$fall/frames"
        	video_path="$fall/video"

        	for video in "$video_path"/*; do
        
            		video_name="${video##*/}"
            		destine="$frame_path/$video_name/"
        
            		echo "VIDEO_NAME=$video_name" 

            		python3 frame.py "$video" "$frame_path" "$frame"       
		done

        done
    done

done

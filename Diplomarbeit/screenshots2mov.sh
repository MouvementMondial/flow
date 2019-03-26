#!/bin/bash 

ffmpeg -r 20 -i /home/thorsten/flow_2/flow/Diplomarbeit/screenshots/%d.png -c:v qtrle -pix_fmt rgb24 flow_replay.mov

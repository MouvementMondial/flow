#!/bin/bash 

ffmpeg -r 50 -i ./screenshots/%d.png -c:v qtrle -pix_fmt rgb24 flow_replay.mov

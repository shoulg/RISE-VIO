#!/bin/bash
echo "please make sure running this script in root dir of slam_in_autonomous_driving. current working dir: $PWD"
docker run -it \
-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
-v "$PWD":/sad/catkin_ws/src/rb_vins \
-v "$(dirname $(dirname "$PWD"))/dataset":/sad/dataset \
--gpus all \
--name rb_vins-container-v1.0 \
sad:v1.0 \
/bin/bash
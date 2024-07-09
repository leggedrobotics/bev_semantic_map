#!/bin/bash
# --delete
rsync -a -z -h -r -v --exclude '*.neptune/*' --exclude '*build*' --exclude '*results/*'  --exclude '*.git/*' --exclude '__pycache__' --exclude '*.pyc' --exclude '*.ipynb'  --out-format="[%t]:%o:%f:Last Modified %M" $HOME/workspaces/bev_ws/src/perception_bev_learning/ mhpatel@ls6.tacc.utexas.edu:/work/09654/mhpatel/ls6/workspaces/perception_bev_learning
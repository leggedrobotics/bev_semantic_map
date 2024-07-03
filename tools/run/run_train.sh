#!/bin/bash

# Define color codes
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print "Starting" at the beginning in red
echo -e "${RED}Starting${NC}"

# Activate conda environment and print "Running prediction" in red
echo -e "${RED}Training 1${NC}"
/home/rschmid/anaconda3/envs/bev/bin/python /home/rschmid/git/bevnet/run.py -t --img

echo -e "${RED}Training 2${NC}"
/home/rschmid/anaconda3/envs/bev/bin/python /home/rschmid/git/bevnet/run.py -t --img --pcd

# echo -e "${RED}Training 3${NC}"
# /home/rschmid/anaconda3/envs/bev/bin/python /home/rschmid/git/bevnet/run.py -t --pcd

# Print "Done" at the end in red
echo -e "${RED}Done${NC}"
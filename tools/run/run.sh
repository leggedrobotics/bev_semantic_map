#!/bin/bash

# Define color codes
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print "starting" at the beginning in red
echo -e "${RED}starting${NC}"

# Activate conda environment and print "Running prediction" in red
echo -e "${RED}Running prediction${NC}"
/home/rschmid/anaconda3/envs/bev/bin/python /home/rschmid/git/bevnet/run.py -p -d b

# Print "Running extraction" in red
echo -e "${RED}Running extraction${NC}"
/home/rschmid/anaconda3/envs/pt/bin/python /home/rschmid/git/perception_tools/scripts/dataset_generation/traversability_estimation/save_torch_trav_and_labels.py

# Print "Running stacking" in red
echo -e "${RED}Running stacking${NC}"
/home/rschmid/anaconda3/envs/pt/bin/python /home/rschmid/git/perception_tools/scripts/dataset_generation/traversability_estimation/save_and_stack_torch_pred.py

# Print "done" at the end in red
echo -e "${RED}done${NC}"
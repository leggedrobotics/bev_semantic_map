

# Road Runner

## Dataset Generation

### Convert MLS to Rosbags:
The rosbags are already available on nextcloud, therefore these steps can now be skipped!

Raw data living on racer-base, racer-base3, ahoy ** (MLS logs, rosbags, processed rosbags)
Raw MLS format: (on racer-base)
```
/mnt/logs/mls_logs/day/run.log
```
MLS check script (this is doing something unkown and checks if the bag is good):
```
core_agv_logging/logging_mls/scripts/recorded_mls_logging_check.py
```
MLS to rosbag:
```/home/racer/Admin/data_tools/scripts/mls2bag.sh``` to convert a folder of many *.log MLS logs
it runs `mlsplay` which can do conversion to rosbags or play ros topics directly (see data_tools README)

```./mls2bag.sh /mnt/logs/mls_logs/jpl6_camp_roberts_y1_d3/```
all in one bag, convert to happy format (`logging_convert/scripts/reorganize_rosbag.py`)


### Create Odometry, Semantics, Traversability
In the default setup we should have the following bags:
- `color_X.bag`, `gps_X.bag`, `imu_X.bag`, `lidar_X.bag`, `radar_X.bag`, `tf_X.bag`, `thermal_X.bag`, `vehicle_X.bag`


We aim now to create the `crl_rzr_bev_color_X.bag` `crl_rzr_bev_odometry_X.bag` `crl_rzr_bev_velodyne_X.bag`.

We have to run the stack twice.

1. **Setup Stack:**  We used the stack version `release/1.6-post-race2` and the docker image.

    Start the docker container with cuda support:
    ```shell
    ~/Admin/core_environment/docker_cuda/run.sh
    ```
    
    Change the branch of the following repositories:
    
    ```shell
    core_agv_logging                  feature/bev_learning
    core_bringup                      feature/bev_learning
    perception_geometric_mapping      feature/bev_learning
    perception_geometric_mapping      feature/bev_learning
    perception_semantic_inference     feature/bev_learning
    ```
    TODO:
    - Create a racer workspace for this configuration using and the correct instructions:
        ```shell
        racerenv export-workspace -d `pwd`
        racerenv update-workspace -h
        vcs status
        vcs pull, push,
        vcs custom --args <fetch, blahjlfksdjlaksdf>
        ```
    - Additional commands for current docker files
    - Building GTSAM: If cmake fails due to boost add the following: `sudo ln -s /usr/include /include`
    - Hacked the semantic configuration from e3 to e2
    - 



2. **Run Stack:**
    The following script will run the stack multiple times on all trajectories in the folder:
    ```shell
    python scripts/preprocessing/generate_rosbags.py -f /folder/to/trajectories
    ```

    It basically calls the stack with the following two tmux settings on each folder:
    ```shell
    racerenv run -c racer_replay_bev_semantics.yaml -d `pwd`
    racerenv run -c racer_replay_bev_trav.yaml -d `pwd` 
    ```

### Convert to HDF5 files 
Verify that this is the latest version given that we changed how datasets are created!
Latest Version: In the first steps only all header information is extracted from the rosbags
```shell
python scripts/preprocessing/convert_rosbags_h5py.py
```
Ensure that all topics are within the h5py file by running:
```shell
python scripts/preprocessing/summary_h5py.py
```


### Create Header-Dataset Configuration
```shell
scripts/preprocessing/generate_dataset_cfg_h5py.py
```
- Here you need to make sure to configure correctly which part should be used for training, test and validation.

### Convert to HDF5 files 2.0
No we can extract all the data from the rosbag given that we know which data is required to be stored to h5df given that we have the dataset_config.pkl
```
shell
python scripts/preprocessing/create_h5py_from_rosbag_and_cfg.py
```

### Create Supervision in Hindsight
Given that here we are modifying the hdf5 files which broke several times my h5df files I would recommend to at first copy and paste the h5df files that should be modified to a new folder. If something goes wrong you can always restore the backup files.
```shell
scripts/preprocessing/generate_supervision_signal_h5py.py
```

### Create Real-Dataset Configuration
```shell
scripts/preprocessing/generate_dataset_cfg_h5py.py
```
Now run the same script again on the hdf5 files containing only the correct data. 
This will generate now the correct mapping corresponding to this hdf5 files while the previous file was with respect to the hdf5 files that containes all the messages (header only)


### Create Supervision in Hindsight
```shell
scripts/preprocessing/generate_supervision_signal_h5py.py
```
This should be it. 
I would recommend you to run the visualization further explained down below. 
I also added the `scripts/preprocessing/downsample_config.py` which allows you to generate downsampled version of the dataset, which is usefull for debugging and trying to overfit to a small dataset



### Setting up the correct training prameters
Modify in `perception_bev_learning/cfg/params.py` the `dataset_train.cfg_file`, `dataset_test.cfg_file` and `dataset_val.cfg_file` point to the correct file.

### Store the statistics of the training dataset
Run the script to analyze the class imbalance of the `wheel_risk_cvar`. 
This is currently not with a version and the results will be stored in the `assets` folder. 
```shell
scripts/preprocessing/data_statistics.py
```
You are now good to start training a network!


## Installation
If you are using conda I would recommend to create an isolated your conda enviornments.
```shell
echo 'export PYTHONNOUSERSITE=1' >> ~/.bashrc 
```

I would recommend installing things using mamba (for this install mamba in the base conda environment)
```shell
conda create --name bev python=3.8
conda activate bev


mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install -c fvcore -c iopath -c conda-forge fvcore iopath
mamba install pytorch3d=0.7.5=py38_cu121_pyt210 -c pytorch3d
mamba install pip

```
Lesson learned to install now all packages into your conda environment make sure to use the correct pip version.
```
which -a pip
```
will print you all available pip versions. Select the one in your conda environment. 
and run the other commands - just because your conda environment is sourced does not imply that `which pip` points to the correct pip installation. 
 

Install perception_bev_learning:
```shell
cd ~/git
git clone git@gitlab.robotics.caltech.edu:racer/perception_bev_learning.git
cd perception_bev_learning && pip install -e ./ 
```

Pointpillars dependencies:
```shell
conda activate bev
cd perception_bev_learning/utils/ops
python3 setup.py develop
```


Install tf_bag follow the readme here:
[tf_bag](https://github.com/IFL-CAMP/tf_bag)


Make sure to have your neptuneai token setup correctly.

Generate a configuration file for your workstation.
```shell
echo 'export ENV_WORKSTATION_NAME=your_name' >> ~/.bashrc 
```
Then create a yaml file in `cfg/env/your_name.yaml` that contains the `dataset_root_dir` and `result_dir` folder.



## Contributing
### Formatting
We use the black formatter. Simply run:
```
black --line-length 120 .
```

### Transformation Notation
Transform from `map` to `sensor`: Notation in code is H_sensor__map (easy to multiply, H_from_frame__to_frame)
```python
p_sensor = H_sensor__map @ p_map
```

## Training
### Default training:
```shell
python scripts/network/train.py
```

### Multiple ablation runs single machine:
```shell
python scripts/network/sweep.py --file sweep.yaml
```
Where the `sweep.yaml` has to be in `cfg/exp` folder.

### Visualize full dataset
```shell
python scripts/network/train.py --general.name visu/full_dataset --dummy True --plot_sequence True --max_epochs 1 --max_steps -1 --check_val_every_n_epoch 1 --plot_dashboard True --project_gt_BEV_on_image True --project_pred_BEV_on_image False --plot_raw_images True --plot_pcd_bev False --plot_all_maps True --logger.name skip --plot_elevation_BEV False --project_pcd_on_image False
```

### Visualize model
```shell
# Always make sure to skip train otherwise the model paramters are updated

# Test 
python scripts/network/train.py --general.name visu/model_test_data --plot_sequence True --max_epochs 1 --max_steps -1 --check_val_every_n_epoch 1 --logger.name skip --test -1 --resume_from_checkpoint /Data/Results/bev_learning/ws/best_param_run/2023-10-17T17-01-15_best_params_regress_elevation/epoch=16-step=15827.ckpt --skip_training True --fatal_risk 0.5 \
--dataset_test.cfg_file "dataset_config_clean_seperation_subsample_100.pkl" --dataset_test.mode test

# Train
python scripts/network/train.py --general.name visu/model_train_data --plot_sequence True --max_epochs 1 --max_steps -1 --check_val_every_n_epoch 1 --logger.name skip --test -1 --resume_from_checkpoint /Data/Results/bev_learning/ws/best_param_run/2023-10-17T17-01-15_best_params_regress_elevation/epoch=16-step=15827.ckpt --skip_training True --fatal_risk 0.5 \
--dataset_test.cfg_file "dataset_config_clean_seperation_subsample_100.pkl" --dataset_test.mode train

# Val
python scripts/network/train.py --general.name visu/model_val_data --plot_sequence True --max_epochs 1 --max_steps -1 --check_val_every_n_epoch 1 --logger.name skip --test -1 --resume_from_checkpoint /Data/Results/bev_learning/ws/best_param_run/2023-10-17T17-01-15_best_params_regress_elevation/epoch=16-step=15827.ckpt --skip_training True --fatal_risk 0.5 \
--dataset_test.cfg_file "dataset_config_clean_seperation_subsample_100.pkl" --dataset_test.mode val


# Fatal Risk is used to visualize
```


### Evaluate trained checkpoint

```shell
python scripts/network/train.py some_args
```

## Online Deployment in replay mode
Start the docker container running the stack
```shell
/home/jonfrey/Admin/core_environment/docker_cuda/run.sh
## Run stack
racerenv run -c racer_bev_model.yaml 
```
Within the `racer_bev_model` adjust the replayed rosbag.
Run the python rosnode:
```shell
conda activate bev
python perception_bev_learning_ros/scripts/bev_node.py --path_checkpoint_folder /home/data/dir --path_tf_bag tf_merged.tf 
```

This also will do the trick, recheck which nodes are run. 
```shell
# Training data  -r 0.2 -s 20
racerenv run -c racer_replay_demo.yaml -d /data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y1_d2_t1_C2_to_C1_jose_driving_Tue_Jun__7_17-41-33_2022_utc


# Test data -r 0.2 -s 220
racerenv run -c racer_replay_demo.yaml -d /data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y1_d2_t1_A4_to_A5_Tue_Jun__7_19-30-49_2022_utc


racerenv run -c racer_replay_demo.yaml -d  /data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y1_d2_t1_B2_to_A4_Tue_Jun__7_19-10-17_2022_utc


# For testing on WS run:
racerenv run -c racer_replay_bev_both.yaml -d /data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/magenta_test/
```

Currently to avoid TF lookup issues we simply read in the TF of the full bag using the BagTfTransformer. Also the configuration is restored from the path_checkpoint_folder. 

Using Dynamic Reconfigure
```
rosrun rqt_gui rqt_gui -s reconfigure
```


## Acknowledgements
Copied code from SimpleBEV licenced under Apache License 2.0
Check also BEVFusion code before publishing. 






--- TODO 
Either make nuscenes work or remove dependencies
Rewrite the elevation_map estimation using the ADABINS classification following the dinov2 paper
Working on concutinating current elevation - Is this working corectly ? 
Borken meter with number of target layers
---

---

# Deprecated

## Manage Racer stack
gita super checkout release/1.6-post-race2
gita super pull
gita super submodule update --init --recursive
catkin build

## Start Container
/home/jonfrey/Admin/core_environment/docker_cuda/run.sh

## Run stack
racerenv run -c racer_bev_model.yaml 




## Cluster tacc

```shell
squeue -u jonfrey

# Estimated start time
squeue --start -j 167635


scancel 170361
# Detailed info 
scontrol show job 750398


/work/09241/jonfrey/ls6/miniconda3/envs/bev/bin/python3 /home1/09241/jonfrey/perception_bev_learning/scripts/network/train.py \

/work/09241/jonfrey/ls6/miniconda3/envs/bev/bin/python3 /home1/09241/jonfrey/perception_bev_learning/scripts/
network/train.py \
sbatch -J learning_rate/01 /home1/09241/jonfrey/perception_bev_learning/scripts/cluster/schedule_30min.sh \
--general.name learning_rate/01 \
--lr 0.01 \
--dataset_train.percentage_normalization 1.0 \
--logger.name neptune \
--pointcloud_backbone skip \
--comment 'Run_Default_Training_Pointcloud_Only'
```

### Current Develop 

```
python scripts/network/train.py --lr 0.01 --dataset_train.percentage_normalization 1.0 --general.name visu/debug --max_steps -1 --train -1 --val -1 --test -1 --use_single_gpu False --plot_dashboard True --project_gt_BEV_on_image True  --project_pred_BEV_on_image False --plot_raw_images False --plot_all_maps True --dataloader_train.shuffle True --tag_by_dataloader True --logger.name neptune --dummy False --max_epochs 5
```

DONES - Visu fast projection also draw lines (done)
DONES - Check if anything is missing on the cluster for  
DONES - Fix the thing for simple parsing also on the cluster (done)
DONES - Recreate the h5py subsampled datasets on the cluster. (in progress)
DONES - Start initial training with default parameters. 
DONES - PointPillar fix it with high priority after rerunning the KITTI code 
DONES - ANYLZE WHAT IS WRONG IN THE plotting. (We found the issue betten cupy and torch did not work)

LowPriority - PointFormer most likely won`t work given the limited number of points - but still you can fix it maybe for future works
LowPriority - MotionNet (make it to the normal motion net and allow for multiple scans over time)
LowPriority - Handling of multiple scans in MotionNet

TODOs - Create eval for just trained network 
TODOs - PointPillar add cuda code instructions - are their cuda operations necessary ? 
DONES - Check simple way to speed up lift splash shoot using lidar

TODOs - Cluster accumulated pointcloud is visualized wrongly 
TODOs - start intitial training after debugging run
      - find learning rate; verify batch size
      - run lss only; runs pointcloud only; run modified lss; run different pointcloud backbone; make sure to get proper video and numbers in neptune
TODOs - write ROS-Node will training on cluster
TODOs - add elevation_map_prediction support

TODOs - 1. Just focus on training the prediction of the risk map. 
            TODO: Select 20 interresting validation keyframes to be visualized ?
            TODO: Are we overfitting a lot ? 
            Ablate: Learning Rate
            Ablate: Network Type (LSS, LSS-LiDAR, PointPillar, "MotionNet")
            Ablate: Number of Pointclouds (Can we increase the input number easily)
            Ablate: Loss Formulation - balanced/percentage_normalization value (potentially recompute the values)
            Feature: Compute the error map for the whole sequence in BEV for MSE and WMSE (weighted MSE)
        2. Move to only regression on the elevation map (This may be less noisy)

TODOs - Work on project points onto image - Check what happens if we turn data-augmentation on
        What I last saw was that there was heavy dataaugmentation in terms or rotation and translationb being applied
        As a next steps turn the learning of the second LiDAR projection on
TODOs - Only look at LLS and modify the lifiting strategy for a few minutes 
TODOs - Create an ablation for Multi-Head vs Single-Head elevation_map_prediction
TODOs - Evaluate the larger batch size training
TODOs - Evaluate the different lifting strategies
TODOs - Evaluate the different point shooting methods

-----------------------------------------------------1

TODOs - Run big evaluation with the current code setup - Check what runs
TODOs - BEVNode check if storing the sample between input and output works
TODOs - Rerun the stack on the training/test and validation sequence
TODOS - BEV Node learning visualizer - plot the elevation and the risk maps via ROS and store them as PNGs for the paper

---------------------------------------------------------
TODOs - Record the rosbag

```
./scripts/ws/job_tmux_remote.sh --image_backbone=skip --pointcloud_backbone=point_pillars --general.name="ablation/network_type" --comment="only_point_pillar"
```


### Running the replay stack
```
/home/jonfrey/Admin/core_environment/docker_cuda/run.sh
racerenv run -c racer_bev_model.yaml -d 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jonfrey/miniconda3/envs/bev/lib && /home/jonfrey/miniconda3/envs/bev/bin/python /home/jonfrey/workspaces/racer_ws/src/perception_bev_learning/perception_bev_learning_ros/scripts/bev_node.py
```
Default is: 
/data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y6_d2_t6_Tue_Jun__7_23-29-08_2022_utc



#### Test with train bag
racerenv run -c racer_replay_bev_model.yaml -d /data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y1_d2_t1_A4_to_A5_Tue_Jun__7_19-30-49_2022_utc_subsample_5

Opening in mode train the following h5py files:
['jpl6_camp_roberts_shakeout_y1_d2_t1_A4_to_A5_Tue_Jun__7_19-30-49_2022_utc_subsample_5', 
'jpl6_camp_roberts_shakeout_y1_d2_t1_A5_to_A4_Tue_Jun__7_19-21-15_2022_utc_subsample_5', 
'jpl6_camp_roberts_shakeout_y1_d2_t1_B2_to_A4_Tue_Jun__7_19-10-17_2022_utc_subsample_5']

Opening in mode val the following h5py files:
['jpl6_camp_roberts_shakeout_y6_d2_t5_open_Tue_Jun__7_23-17-53_2022_utc_subsample_20', 
'jpl6_camp_roberts_shakeout_y6_d2_t6_Tue_Jun__7_23-29-08_2022_utc_subsample_20', 
'jpl6_camp_roberts_shakeout_y6_d2_t7_delta1_Tue_Jun__7_23-35-19_2022_utc_subsample_20', 
'jpl6_camp_roberts_shakeout_y6_d2_t8_delta2_Tue_Jun__7_23-45-36_2022_utc_subsample_20', 
'jpl6_camp_roberts_shakeout_y6_d2_t9_top_of_hill_Wed_Jun__8_00-13-40_2022_utc_subsample_20']


### Evaluate a model
```shell
python scripts/network/train.py \
--general.name debug/eval \
--test -1 \
--plot_dashboard True \
--plot_all_maps True \
--project_gt_BEV_on_image True \
--project_pred_BEV_on_image True \
--tag_by_dataloader True \
--logger.name skip \
--skip_training True \
--ckpt_path /home/jonfrey/workspaces/racer_ws/src/perception_bev_learning/results/debug/2023-04-12T11-17-50_initial_training/epoch=15-step=5312.ckpt
```


### Training cluster

Train command fast no visu:
```

/work/09241/jonfrey/ls6/miniconda3/envs/bev/bin/python3 /home1/09241/jonfrey/perception_bev_learning/scripts/network/train.py \
sbatch /home1/09241/jonfrey/perception_bev_learning/scripts/cluster/schedule_30min.sh \
--lr 0.01 \
--dataset_train.percentage_normalization 1.0 \
--general.name train/full_training_lr_01 \
--max_steps 10000 \
--train 0 \
--val 0 \
--test -1 \
--use_single_gpu False \
--plot_dashboard False \
--plot_all_maps False \
--project_gt_BEV_on_image False \
--project_pred_BEV_on_image False \
--plot_raw_images False \
--dataloader_train.shuffle True \
--tag_by_dataloader True \
--logger.name neptune
```

Train command visu:
```
/work/09241/jonfrey/ls6/miniconda3/envs/bev/bin/python3 /home1/09241/jonfrey/perception_bev_learning/scripts/network/train.py 
sbatch /home1/09241/jonfrey/perception_bev_learning/scripts/cluster/schedule_30min.sh \
--general.name train_lr_run/0001 --lr 0.0001 --dataset_train.percentage_normalization 1.0 --max_steps 1000 --train 4 --val 4 --test -1 --use_single_gpu False --plot_dashboard True --project_gt_BEV_on_image False  --project_pred_BEV_on_image True --plot_raw_images False --plot_all_maps True --dataloader_train.shuffle True --tag_by_dataloader True --logger.name neptune
```

#### Visu full dataset
```
/work/09241/jonfrey/ls6/miniconda3/envs/bev/bin/python3 /home1/09241/jonfrey/perception_bev_learning/scripts/network/train.py --lr 0.01 --dataset_train.percentage_normalization 1.0 --general.name train/full_training_lr_01 --max_steps 10000 --train -1 --val -1 --test -1 --use_single_gpu False --plot_dashboard True --project_gt_BEV_on_image True  --project_pred_BEV_on_image False --plot_raw_images False --plot_all_maps True --dataloader_train.shuffle False --tag_by_dataloader True --logger.name neptune --dummy True
```

## Job
```
sbatch /home1/09241/jonfrey/perception_bev_learning/scripts/cluster/schedule.sh \
```

## Interactive
```
/work/09241/jonfrey/ls6/miniconda3/envs/bev/bin/python3 /home1/09241/jonfrey/perception_bev_learning/scripts/network/train.py \
--lr 0.0001 \
--dataset_train.percentage_normalization 1.0 \
--general.name visu/full_dataset \
--max_steps 1002 \
--train -1 \
--val -1 \
--test -1 \
--use_single_gpu True \
--plot_dashboard True \
--project_gt_BEV_on_image False \
--plot_raw_images True \
--plot_all_maps True \
--dataloader_train.shuffle False \
--dummy True \
--tag_by_dataloader True \
--logger.name skip \
--tag_with_front_camera True
```

## Dataset overview

Multiple HDF5 files which basicially corrospond to topics:
```
sequence_key.h5py
    sequence_key
        topic_color_camera_info
        topic_color_compressed
        ..
        ..
        ..
        map_micro
        map_micro_gt
        velodyne_merged_points
```
Configuration file:
```
dataset_config.pkl
    -   'image_front':0
        'gridmap_micro': 0
        'gridmap_short': 2
        'image_left': 0
        'image_right': 0
        'image_back': 0
        'pointclouds': [0]
        'sequence_key': 'jpl6_camp_roberts_shakeout_y6_d2_t6_Tue_Jun__7_23-29-08_2022_utc'
        'mode': 'train'
    ..
    ..
```
Sidenodes: Here for map_micro_gt no entry exists given that the two maps are synchronicized.

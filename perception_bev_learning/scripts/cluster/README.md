# Usage of TAAC Lonestar6
Read this at first: https://docs.tacc.utexas.edu/hpc/lonestar6/  
TACC portal: https://tacc.utexas.edu/portal/dashboard

Two-Factor-Authentification: do not use the TAAC smartphone app but instead the Google Chrom - Authenticator Extension which allows you to move the two-factor-authentification between devices and to your computer.   

## Login
```shell
ssh username@ls6.tacc.utexas.edu
```

## Storage overview 
Read Table 3. File Systems: https://docs.tacc.utexas.edu/hpc/lonestar6/  
### Suggested usage:  
- All your code and potential miniconda environment can be stored in $HOME - Make sure your conda environment is below 10GB.  
- Large datasets of 1TB can be stored in $WORK.  
- Tar the dataset if many individual small files as .png into a .tar.  
- Alternativelly, you can store all data in a large h5py file.  
- When starting a job on a node copy all of the data onto the SSD or load it into the RAM.  

## Compute Nodes
Optimize your code to take full advantage of 3 x A100 GPU  
```yaml
- 128 cores 
- 3 NVIDIA A100 GPUs
- 256GB Random Access Memory (RAM)
- 288GB SSD /tmp parition 
```

- I would recommend either you always run 3 times your training parallel and to hyperparameter search this way.  
- Or optimize the code to take full advantage of 3 A100.   
- If you train a image segmentation or classification model that is not very large the bottleneck will most likely be the dataloading.   
- Also the small SSD of 288GB forces you when dealing with large datasets to load the data via the network.   

## Example to run a debug job in an interactive seesion:
```shell
idev -p gpu-a100 -N 1 -n 1 -t 02:00:00 -A JPL-PUB
idev -p vm-small -N 1 -n 1 -t 02:00:00 -A JPL-PUB
```
- I would recommend start out with an interactive session and opening tmux on the compute node.  
- You can run nvidia-smi and htop to monitor the GPU and CPU usage while running your code.    
- If the GPU is not fully utilized you can debug what takes a lot of time in terms of dataloading.  
- When using torch I would e.g. time the get_item function of the dataset and then multiple this with the number of workes. 
- E.g getting a single image takes 1000ms. You have 50 workers this means to loading on average for an image takes 20ms.  
- If your GPU is capable to throughput 10 batches with a batchsize of 16 you should be able to load 160 images per second while you are only able to load 50 images per/s in this case therefore the bottleneck is the dataloading not the GPU.  
- For this is is very imporant to understand from where the data has to be loaded.  
/tmp should be fast but $WORK is slow.  

## Example to run code
Start a job on a compute node using the SLURM scheduler:
```shell
sbatch $HOME/slurm_script.sh --magic_parameter 1 --magic_parameter2 2
```

The slurm_script.sh starts the node and passes the provided command line arguments to the $HOME/job.sh which is actually the code you want to run.

Example your_script.sh script:
```shell

#!/bin/bash
echo "job.sh called with the following command line arguments:"
echo $@
source ~/.bashrc

# Typicially you would like to copy the data needed to /tmp from $WORK.
tar -xvf  $WOKR/dataset.tar -C /tmp/dataset.tar
cp /tmp/dataset.h5py /tmp/dataset.h5py

# Run python code with the provided command line arguments and specified python environment.
/work/09241/jonfrey/ls6/miniconda3/envs/bev/bin/python3 $HOME/perception_bev_learning/scripts/network/train.py $@

# Make sure to store all results in $WORK, $HOME, $SCRATCH not on /tmp given that this will be delted when the job ends.
```


Example slurm_script.sh script:
```shell
#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonestar6 AMD Milan nodes
#
#   *** Serial Job in Normal Queue***
# 
# Last revised: October 22, 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch milan.serial.slurm" on a Lonestar6 login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J %x-%j                                           # Job name
#SBATCH -o /work/09241/jonfrey/ls6/results/%x-%j.out       # Name of stdout output file
#SBATCH -e /work/09241/jonfrey/ls6/results/%x-%j.err       # Name of stderr error file
#SBATCH -p gpu-a100        # Queue (partition) name - queue name deterimens which resource you get
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A JPL-PUB       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=jonfrey@ethz.ch

echo "job.sh called with the following command line arguments:"
echo $@


$HOME/job.sh $@
```
#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted. Check updates
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --mem=180G
#SBATCH --job-name=inference_caption
#SBATCH --ntasks=1
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=0-48:00:00
#SBATCH -o full_haptic_6_9_inference.out

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user
#Skipping many options! see man sbatch
# From here on, we can start our program
. /etc/profile.d/modules.sh
module load anaconda3/5.3.1
module load cuda/10.1
eval "$(conda shell.bash hook)"
conda activate Llama
#your script, in this case: write the hostname and the ids of the chosen gpus and the status of the GPU.
hostname
echo $CUDA_VISIBLE_DEVICES
echo $nvidia-smi
python inference.py

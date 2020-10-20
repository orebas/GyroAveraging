#!/bin/bash
# This line tells the shell how to execute this script, and is unrelated
# to SLURM.
   
# at the beginning of the script, lines beginning with "#SBATCH" are read by
# SLURM and used to set queueing options. You can comment out a SBATCH
# directive with a second leading #, eg:
##SBATCH --nodes=1
   
# we need 1 node, will launch a maximum of one task and use one cpu for the task: 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --array=0-6
   
# we expect the job to finish within 5 hours. If it takes longer than 5
# hours, SLURM can kill it:
#SBATCH --time=10:00:00
   
# we expect the job to use no more than 2GB of memory:
#SBATCH --mem=120GB
   
# we want the job to be named "myTest" rather than something generated
# from the script name. This will affect the name of the job as reported
# by squeue:
#SBATCH --job-name=GACPU
 
# when the job ends, send me an email at this email address.
#SBATCH --mail-type=END
#SBATCH --mail-user=ob749@nyu.edu
   
# both standard output and standard error are directed to the same file.
# It will be placed in the directory I submitted the job from and will
# have a name like slurm_12345.out
#SBATCH --output=hardfunc_%A_%a.out
#SBATCH --error=hardfunc_%a_%a.err
 
# once the first non-comment, non-SBATCH-directive line is encountered, SLURM
# stops looking for SBATCH directives. The remainder of the script is  executed
# as a normal Unix shell script
  
# first we ensure a clean running environment:
module purge
# and load the module for the software we are using:
module load boost/gnu/1.66.0
module load fftw/intel/3.3.5
module load cuda/10.1.105
module load gcc/6.3.0
  

ulimit -c 0


# next we create a unique directory to run this job in. We will record its
# name in the shell variable "RUNDIR", for better readability.
# SLURM sets SLURM_JOB_ID to the job id, ${SLURM_JOB_ID/.*} expands to the job
# id up to the first '.' We make the run directory in our area under $SCRATCH, because at NYU HPC
# $SCRATCH is configured for the disk space and speed required by HPC jobs.
#RUNDIR=$SCRATCH/GA/run-${SLURM_JOB_ID/.*}
#mkdir $RUNDIR
  
OMP_NUM_THREADS=20
SRCDIR=$HOME/GyroAveraging
# we will be reading data in from somewhere, so define that too:
#DATADIR=$SCRATCH/my_project/data
  
# the script will have started running in $HOME, so we need to move into the
# unique directory we just created
cd $RUNDIR
$SRCDIR/GyroAverage-CUDA --calc=$SLURM_ARRAY_TASK_ID --func=2 --cache=/scratch/ob749/GA/cache/


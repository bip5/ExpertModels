#SBATCH --account=xxxx
#SBATCH --partition=xxxx
#SBATCH --job-name=Eval
#SBATCH --output=/Your_dir/BraTS/eval_logs/Eval.%j
#SBATCH --error=/Your_dir/BraTS/eval_logs/Eval.err.%j
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
###SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --export=ALL

module load anaconda
source activate pix2pix

conda list >> result_$SLURM_JOB_ID.txt
########0 means not loading barlow in load barlow, barlow flag and bar_model_name not relevant when load_barlow 0 
python -u '/scratch/a.bip5/BraTS/scripts/Training/training.py'
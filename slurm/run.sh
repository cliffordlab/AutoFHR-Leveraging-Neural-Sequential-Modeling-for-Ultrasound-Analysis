#!/bin/bash
#SBATCH -J AutoFHR Training       # Job name
#SBATCH -p computational-cluster  # Computational Cluster
#SBATCH -G 1                      # Number of GPUs
#SBATCH -t 3-00:00:00             # Time limit (days-hh:mm:ss)
#SBATCH --mem=64G                 # Memory per node
#SBATCH -n 16                     # Number of CPU cores
#SBATCH -N 1                      # Number of nodes
#SBATCH -o logs/AutoFHR_job-%A.out        # Standard output log file
#SBATCH -e logs/AutoFHR_job-%A.err        # Standard error log file


# -------------------------------
# 1️⃣ Start Job & System Logging
# -------------------------------
echo "==========================================="
echo "🚀 Job Started on $(hostname) at $(date)"
echo "==========================================="

# Record job start time
start_time=$(date +%s)

# Unload conflicting modules
module purge

# Explicitly set PATH and PYTHONPATH to ignore Miniconda
export PATH=/opt/modules/Python/3.11.5/bin:$PATH
export PYTHONPATH=/opt/modules/Python/3.x/lib/python3.11.5/site-packages  # Adjust `python3.x` to match the actual version

# Verify Python Environment
echo "🔹 Using Python from: $(which python)"
python --version

# Log system resources
echo "🔹 CPU & Memory Info:"
lscpu | grep "Model name"
free -h
echo "🔹 GPU Info (if available):"
nvidia-smi || echo "No GPU detected"


# -------------------------------
# 2️⃣ Navigate to Working Directory
# -------------------------------
cd /home/AutoFHR/

# -------------------------------
# 3️⃣ Run Training Script with Monitoring
# -------------------------------
echo "🚀 Starting Model Training at $(date)"
python -m 

# Capture exit status
exit_status=$?

# -------------------------------
# 4️⃣ Post-Job Resource Usage Logging
# -------------------------------
echo "🔹 Memory Usage After Job:"
free -h

# -------------------------------
# 5️⃣ Handle Job Completion & Errors
# -------------------------------
end_time=$(date +%s)
runtime=$((end_time - start_time))

if [ $exit_status -eq 0 ]; then
    echo "✅ Training Completed Successfully!"
else
    echo "❌ Training Failed with Exit Code: $exit_status"
    echo "Check logs/job-${SLURM_JOB_ID}-training.log for details."
fi

echo "==========================================="
echo "🏁 Job Finished at $(date)"
echo "⏳ Total Runtime: $((runtime / 3600))h $(((runtime % 3600) / 60))m $((runtime % 60))s"
echo "==========================================="
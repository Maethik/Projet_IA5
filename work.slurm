#!/bin/bash
#SBATCH --job-name=test-pytorch-gpu-uv
#SBATCH --output=test-pytorch-gpu-uv-%j.out
#SBATCH --error=test-pytorch-gpu-uv-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --partition=gpu-l40s

# --- Config ---
NOTEBOOK_PATH="$HOME/test/Projet_IA5/work.ipynb" # A MODIFIER
OUTPUT_NOTEBOOK="$HOME/test/Projet_IA5/work_out_${SLURM_JOB_ID}.ipynb"
export UV_PROJECT=$HOME/test/Projet_IA5 # Répertoire de travail pour ce job (dans votre home)
# Si uv est installé localement
# export UV_HOME=$HOME/.cache/uv
# export PATH="$UV_HOME/bin:$PATH" 

echo "=== Informations SLURM ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Noeud : $SLURM_NODELIST"
echo "GPU(s) alloué(s) : $CUDA_VISIBLE_DEVICES"
echo "--------------------------"

# --- Initialisation uv ---
echo "=== Préparation de l'environnement uv ==="

# Crée un répertoire de travail isolé pour ce job
mkdir -p "$UV_PROJECT"
cd "$UV_PROJECT"

# Initialise le projet s'il n'existe pas encore
if [ ! -f "pyproject.toml" ]; then
  uv init --python 3.12 --quiet
fi

# Installe PyTorch 
uv add torch numpy papermill jupyter datasets matplotlib numpy scikit-learn sentence_transformers wordcloud
uv run papermill "$NOTEBOOK_PATH" "$OUTPUT_NOTEBOOK"

echo "=== Notebook exécuté ==="
echo "Résultat enregistré dans : $OUTPUT_NOTEBOOK"
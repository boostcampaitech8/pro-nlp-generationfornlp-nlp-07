#!/bin/bash

# ==============================================================================
# Batch Runner for Auto Experiment Script
# ==============================================================================
# Usage: ./run_batch.sh
# description: Runs experiments sequentially for defined Model/Experiment pairs.
#              Supports automatic Hugging Face upload (via autoexp.py) and
#              local checkpoint cleanup to save disk space.
# ==============================================================================

# --- Configuration Section ----------------------------------------------------

# List of Models to Experiment With
MODELS=(
    "NCSOFT/Llama-VARCO-8B-Instruct"
)

# List of Experiment Names (Must match the order of MODELS)
EXPERIMENTS=(
    "llama-varco-8b-lora-v1"
)

# Set to true to delete local checkpoints after successful run/upload
# This helps prevent filling up disk space when running many experiments.
DELETE_LOCAL_CHECKPOINTS=false

# ------------------------------------------------------------------------------

# Get valid directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Validation
if [ "${#MODELS[@]}" -ne "${#EXPERIMENTS[@]}" ]; then
    echo "Error: The number of MODELS (${#MODELS[@]}) does not match the number of EXPERIMENTS (${#EXPERIMENTS[@]})."
    exit 1
fi

echo "================================================================================"
echo "Starting Batch Execution"
echo "Found ${#MODELS[@]} experiments to run."
echo "Delete Local Checkpoints: $DELETE_LOCAL_CHECKPOINTS"
echo "================================================================================"

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    EXP="${EXPERIMENTS[$i]}"
    
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Running Experiment $((i+1)) / ${#MODELS[@]}"
    echo "Model:      $MODEL"
    echo "Experiment: $EXP"
    echo "Timestamp:  $(date)"
    echo "--------------------------------------------------------------------------------"
    
    # Execute the python script
    # Note: We assume autoexp.py is in the same directory as this script
    python "$SCRIPT_DIR/autoexp.py" --model_name "$MODEL" --experiment_name "$EXP"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ">>> Experiment '$EXP' completed successfully."
        
        # Cleanup Logic
        if [ "$DELETE_LOCAL_CHECKPOINTS" = true ]; then
            # Define output dir (Must match autoexp.py logic: outputs/T8091)
            # CAUTION: This matches the hardcoded path in autoexp.py
            OUTPUT_DIR="$PROJECT_ROOT/outputs/T8091"
            
            echo ">>> Cleaning up checkpoints in $OUTPUT_DIR..."
            
            # Remove checkpoint folders (checkpoint-*)
            # We use 'find' to be safe or simple glob expansion
            rm -rf "$OUTPUT_DIR"/checkpoint-*
            
            # Remove best model folder (optional, maybe keep it?)
            # Usually uploaded to HF, safest to delete if space is tight.
            rm -rf "$OUTPUT_DIR"/best_model
            
            echo ">>> Local checkpoints deleted."
        fi
        
    else
        echo "!!! Experiment '$EXP' FAILED with exit code $EXIT_CODE."
        echo "!!! Aborting batch execution."
        exit $EXIT_CODE
    fi
    
    # Optional: Delay between runs
    sleep 3
done

echo ""
echo "================================================================================"
echo "Batch Execution Finished Successfully"
echo "Timestamp: $(date)"
echo "================================================================================"

#!/bin/bash

# Change to the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
echo "Running experiments from: $(pwd)"

# Create output directory if it doesn't exist
mkdir -p output

# Download and extract dataset if not already present
DATASET_DIR="ccle_vs_gdsc"
DATASET_ARCHIVE="ccle_vs_gdsc.tgz"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset directory '$DATASET_DIR' not found. Downloading and extracting..."
    if [ ! -f "$DATASET_ARCHIVE" ]; then
        echo "Downloading dataset archive..."
        curl -O https://bimsbstatic.mdc-berlin.de/akalin/buyar/flexynesis-benchmark-datasets/ccle_vs_gdsc.tgz
    else
        echo "Dataset archive already exists, skipping download."
    fi
    
    echo "Extracting dataset..."
    tar -xzvf ccle_vs_gdsc.tgz
    echo "Dataset extraction completed."
else
    echo "Dataset directory '$DATASET_DIR' already exists, skipping download and extraction."
fi

# Ensure the correct flexynesis-mps wheel is installed in the current Python environment
PYTHON_BIN=${PYTHON_BIN:-python3}
WHEEL_PATH="/Users/hc/Documents/flexxy/flexynesis-mps/dist/flexynesis_mps-1.0.8-py3-none-any.whl"
${PYTHON_BIN} -m pip install "$WHEEL_PATH"

# Base command (use python -m flexynesis to ensure the wheel is used)
BASE_CMD="$PYTHON_BIN -m flexynesis --use_gpu --data_path $SCRIPT_DIR/ccle_vs_gdsc --variance_threshold 50 --features_top_percentile 20 --target_variables Erlotinib --early_stop_patience 10 --hpo_iter 1 --outdir $SCRIPT_DIR/output --safetensors"
# Define options
MODELS=("supervised_vae" "GNN")
DATA_TYPES=("mutation")
FUSION_METHODS=("early" "intermediate")
GNN_CONV_TYPES=("GC" "SAGE")

# Output log file
LOG_FILE="output/experiment_log.tsv"
echo -e "experiment_id\tmodel\tdata_types\tfusion\tgnn_conv" > "$LOG_FILE"

# Experiment counter
experiment_counter=1

# Loop over models
for model in "${MODELS[@]}"; do
    for data_type in "${DATA_TYPES[@]}"; do
        if [[ "$model" == "GNN" ]]; then
            # GNN only supports early fusion, but we vary the GNN convolution type
            for gnn_type in "${GNN_CONV_TYPES[@]}"; do
                prefix="experiment${experiment_counter}"
                stats_file="output/${prefix}.stats.csv"
                
                if [[ -f "$stats_file" ]]; then
                    echo "Skipping $prefix as stats.csv already exists."
                else
                    echo "Running experiment $experiment_counter with $model, $data_type, early, $gnn_type"
                    $BASE_CMD --data_types "$data_type" --fusion early --model_class "$model" --gnn_conv_type "$gnn_type" --prefix "$prefix"
                    echo -e "$prefix\t$model\t$data_type\tearly\t$gnn_type" >> "$LOG_FILE"
                fi
                ((experiment_counter++))
            done
        else
            # Skip intermediate fusion for single data type
            if [[ "$data_type" == *","* ]]; then
                fusion_methods=("early" "intermediate")
            else
                fusion_methods=("early")
            fi
            
            for fusion in "${fusion_methods[@]}"; do
                prefix="experiment${experiment_counter}"
                stats_file="output/${prefix}.stats.csv"
                
                if [[ -f "$stats_file" ]]; then
                    echo "Skipping $prefix as stats.csv already exists."
                else
                    echo "Running experiment $experiment_counter with $model, $data_type, $fusion"
                    $BASE_CMD --data_types "$data_type" --fusion "$fusion" --model_class "$model" --prefix "$prefix"
                    echo -e "$prefix\t$model\t$data_type\t$fusion\tNA" >> "$LOG_FILE"
                fi
                ((experiment_counter++))
            done
        fi
    done
done

echo "All experiments completed. Log saved to $LOG_FILE."


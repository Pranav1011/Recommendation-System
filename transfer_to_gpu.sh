#!/bin/bash

# Transfer files to RTX 4090
REMOTE="root@109.198.107.223"
PORT="42546"
REMOTE_DIR="/workspace/Recommend"

echo "Creating remote directory structure..."
ssh -p $PORT $REMOTE "mkdir -p $REMOTE_DIR/{src/{training,models,data},configs,data/processed}"

echo "Transferring code files..."
scp -P $PORT src/training/train_lightgcn.py $REMOTE:$REMOTE_DIR/src/training/
scp -P $PORT src/models/lightgcn.py $REMOTE:$REMOTE_DIR/src/models/
scp -P $PORT src/data/graph_builder.py $REMOTE:$REMOTE_DIR/src/data/
scp -P $PORT src/training/metrics.py $REMOTE:$REMOTE_DIR/src/training/
scp -P $PORT configs/train_config_lightgcn.json $REMOTE:$REMOTE_DIR/configs/

echo "Transferring data files..."
scp -P $PORT data/processed/train_ratings.parquet $REMOTE:$REMOTE_DIR/data/processed/
scp -P $PORT data/processed/test_ratings.parquet $REMOTE:$REMOTE_DIR/data/processed/

echo "Creating __init__.py files..."
ssh -p $PORT $REMOTE "touch $REMOTE_DIR/src/__init__.py $REMOTE_DIR/src/training/__init__.py $REMOTE_DIR/src/models/__init__.py $REMOTE_DIR/src/data/__init__.py"

echo "Done! Files transferred to GPU."

#!/bin/bash

echo 'Starting mlflow server'
nohup mlflow server --backend-store-uri ./data/mlruns --default-artifact-root ./data/mlruns --host 0.0.0.0  >> mlflow.log 2>&1 &

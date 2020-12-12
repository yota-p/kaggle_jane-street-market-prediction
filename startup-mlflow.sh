#!/bin/bash

echo 'Starting mlflow server'
nohup mlflow server --host 0.0.0.0 & >> mlflow.log 2>&1 &


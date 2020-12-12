#!/bin/bash

echo 'Starting mlflow server'
nohup mlflow server --host 0.0.0.0 & >> mlflow.log 2>&1 &

echo 'Hit below at local to establish port forwarding connection to server:'
echo '    $ gcloud compute ssh HOST -- -N -L 8080:localhost:8080'

#!/bin/bash

curDir=$(dirname "$(realpath "$0")")
BACKENDSTOREDIR=$curDir
ARTIFACTDIR=$curDir"/../../model/MLflow"

mlflow server --backend-store-uri sqlite:///$BACKENDSTOREDIR/mlflow.db --default-artifact-root $ARTIFACTDIR/mlruns

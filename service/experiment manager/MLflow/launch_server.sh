#!/bin/bash

curDir=$(dirname $(dirname $(dirname $(dirname "$(realpath "$0")"))))
BACKENDSTOREDIR=$curDir"/code/output/model/MLflow"
ARTIFACTDIR=$curDir"/code/output/model/MLflow"

mlflow server --backend-store-uri sqlite:///$BACKENDSTOREDIR/mlflow.db --default-artifact-root $ARTIFACTDIR/mlruns

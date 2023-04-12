#!/bin/bash

TYPE=$1             # type one between demo, cpu, gpu
CONTAINER_NAME=$2   # name of the container
MODEL_PATH=$3       # path to the model directory

rm -r model
mkdir model
cp "$MODEL_PATH"/* model
docker build -f dockerfiles/Dockerfile."$TYPE" --build-arg MODEL_PATH="model" -t "$CONTAINER_NAME" .
rm -r model

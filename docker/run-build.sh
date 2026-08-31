#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

debug="true"

docker build -t rosrviz .

if [ "$debug" = "true" ]; then
    docker build --target dev -t rosaudioclip -f Dockerfile.phd .
else
    docker build --target production -t rosaudioclip -f Dockerfile.phd .
fi
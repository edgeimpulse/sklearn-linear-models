#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

TAG=$(find $SCRIPTPATH/*.py "$SCRIPTPATH/Dockerfile" "$SCRIPTPATH/requirements.txt" -type f | sort | xargs shasum | shasum | awk -F ' ' '{ print $1 }')
echo $TAG

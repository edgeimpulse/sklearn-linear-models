#!/bin/bash
set -e

ARGS=${@:1}

echo $ARGS

python3 -u $ARGS

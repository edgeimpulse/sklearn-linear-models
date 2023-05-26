#!/bin/bash
set -e

UNAME=`uname -m`

if [ "$UNAME" == "aarch64" ]; then
    wget https://github.com/yoziru/jax/releases/download/jaxlib-v0.3.15/jaxlib-0.3.15-cp310-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
    pip3 install jaxlib-0.3.15-cp310-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
    rm *.whl
else
    pip3 install jaxlib==0.3.15
fi

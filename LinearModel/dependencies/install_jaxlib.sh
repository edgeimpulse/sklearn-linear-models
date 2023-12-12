#!/bin/bash
set -e

UNAME=`uname -m`

if [ "$UNAME" == "aarch64" ]; then
wget https://cdn.edgeimpulse.com/build-system/wheels/aarch64/jaxlib-0.4.1-cp310-cp310-manylinux2014_aarch64.whl
    pip3 install jaxlib-0.4.1-cp310-cp310-manylinux2014_aarch64.whl
    rm jaxlib-0.4.1-cp310-cp310-manylinux2014_aarch64.whl
else
    pip3 install jaxlib==0.4.1
fi

# syntax = docker/dockerfile:experimental
FROM ubuntu:20.04
WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

# Install base packages
RUN apt update && apt install -y curl zip git lsb-release software-properties-common apt-transport-https vim wget

# Install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.10 python3.10-distutils

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py "pip==21.3.1" "setuptools==62.6.0" && \
    rm get-pip.py

# Symlink so we use python3/pip3 calls
RUN rm -f /usr/bin/python3 /usr/bin/pip3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/pip3.10 /usr/bin/pip3

# Copy other Python requirements in and install them
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Copy the rest of your training scripts in
COPY . ./

# And tell us where to run the pipeline
ENTRYPOINT ["./run.sh", "train.py"]

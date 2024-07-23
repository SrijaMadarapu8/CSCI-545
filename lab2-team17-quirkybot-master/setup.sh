#!/usr/bin/env bash
set -euxo pipefail

sudo apt-get update
sudo apt-get install -y \
      python-catkin-tools \
      python-pytest \
      python-skimage \
      python-pip \
      ros-kinetic-rviz \
      ros-kinetic-tf2-ros \
      ros-kinetic-map-server

pip install -U 'singledispatch==3.7.0' 'llvmlite==0.19.0' 'numba==0.34.0' 'numpy==1.13.3' 'scikit-learn==0.19.2'

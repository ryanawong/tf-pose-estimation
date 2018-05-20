#!/usr/bin/env sh
set -e

python webcam.py --model=mobilenet_thin --resolution=432x368 --camera=0

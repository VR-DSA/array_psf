#!/bin/bash
#

nvcc $1 -o run -ccbin=g++ -lm --gpu-architecture=sm_75 -O3

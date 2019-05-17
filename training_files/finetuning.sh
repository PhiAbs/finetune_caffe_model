#!/usr/bin/env sh
set -e

./../../caffe/build/tools/caffe train \
    --solver=/home/pussycat/finetune_caffe_model/models/caffenet/solver.prototxt \
    --weights=/home/pussycat/finetune_caffe_model/models/weights/bvlc_reference_caffenet.caffemodel $@

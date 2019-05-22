#!/usr/bin/env sh
set -e

./../../caffe/build/tools/caffe train \
    --solver=/home/pussycat/finetune_caffe_model/models/caffenet/solver.prototxt \
    --weights=/home/pussycat/finetune_caffe_model/models/weights/bvlc_reference_caffenet.caffemodel \
    2>&1 | tee /home/pussycat/finetune_caffe_model/logging/finetune_on_custom_classes_02.log $@

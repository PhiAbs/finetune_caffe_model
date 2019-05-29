#!/usr/bin/env sh
set -e
./../../caffe/build/tools/caffe train \
    --solver=../models/caffenet/solver.prototxt \
    --snapshot=../models/caffenet/run2b_dog_cat_female_male_ball_continued/solver_iter_80000.solverstate \
    2>&1 | tee /home/pussycat/finetune_caffe_model/logging/finetune_on_2b_dog_cat_female_male_ball_continued_02.log
    $@

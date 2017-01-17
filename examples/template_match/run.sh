#!/bin/bash

ROOT=/run/media/larry/fafb882a-0878-4e0a-9ccb-2fb979b7f717/e3dengine/caffe
TOOLS=$ROOT/build/tools

$TOOLS/caffe train -solver $ROOT/examples/template_match/solver.prototxt 

$TOOLS/caffe train -solver solver.prototxt \
-snapshot triplet_iter_230000.solverstate \
2>&1 | tee /home/zxluo/learned_descriptor/net/log.txt
echo "Done."
#-weights matchnet_baseline_iter_300000.caffemodel \

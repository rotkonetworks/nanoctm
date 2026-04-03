#!/bin/bash
# Train CTM v2 (4 brain regions) from scratch on ClimbMix-400B
#
# d6 model (384-dim, 6 layers) fits on 8GB AMD GPU
# CTM v2 replaces last layer's MLP with 4-region brain architecture
# K=4 thinking iterations per token
#
# Expected: ~3-5 min/1000 steps, bpb dropping below 2.0 by step 5k

NANOCHAT_NO_COMPILE=1 python3 -u -m scripts.base_train \
    --use-ctm \
    --ctm-v2 \
    --depth=6 \
    --ctm-iterations=4 \
    --ctm-memory-length=8 \
    --ctm-synapse-depth=2 \
    --ctm-layers=last \
    --device-batch-size=2 \
    --total-batch-size=32768 \
    --max-seq-len=256 \
    --cache-aware-ratio=0.30 \
    --num-iterations=10000 \
    "$@"

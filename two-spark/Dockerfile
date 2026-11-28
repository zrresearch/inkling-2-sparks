# Two-Spark build: the tuned GB10 image plus the expert cache.
#   gb10_cache.py   the expert cache (slots, pack loads with O_DIRECT, grouped passes)
#   patch_cache.py  slot-map GEMM kernels, MXFP8 expert split at TP=2, cache hooks in moe.py / model.py
#   patches/*.diff  the forward pass for the cache (exact step experts, grouped passes for long prompts, bmm out=)
FROM inkling2-gb10
COPY gb10_cache.py patch_cache.py /opt/inkling2-gb10/
COPY patches /opt/inkling2-gb10/patches
RUN python3 /opt/inkling2-gb10/patch_cache.py && \
    cd /usr/local/lib/python3.12/dist-packages && \
    for p in moe gb10_bmm model; do patch -p1 --forward < /opt/inkling2-gb10/patches/$p.diff || exit 1; done && \
    python3 -c "import vllm.models.inkling2.moe, vllm.models.inkling2.ops.gb10_cache" && \
    md5sum vllm/models/inkling2/moe.py vllm/models/inkling2/ops/gb10_bmm.py vllm/models/inkling2/model.py vllm/models/inkling2/ops/gb10_cache.py

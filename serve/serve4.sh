#!/usr/bin/env bash
# Inkling 2 Small on four DGX Sparks: vLLM tensor parallel 4 across the nodes, one container per node, with the GB10
# image from ../build. Run on the head node (rank 0). Workers start headless first; the head serves the API.
#   serve4.sh start [extra vllm args...] | stop | logs [rank]
# Settings come from nodes.conf next to this script (copy nodes.conf.example) and the environment.
set -euo pipefail
cd "$(dirname "$0")"
[[ -f nodes.conf ]] || { echo "copy nodes.conf.example to nodes.conf and fill it in"; exit 2; }
. ./nodes.conf
IMAGE=${IMAGE:-inkling2-gb10}
NAME=${NAME:-Inkling-2-Small}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8899}
CTX=${CTX:-262144}
MASTER_PORT=${MASTER_PORT:-29701}
PROF=${PROF:-/tmp/inkling-prof}   # mounted at /prof (profiler output, INKLING_EXPERT_STATS files)
# The tuned build (see README): experts split across the Sparks on the NVFP4 layers, FP8 attention weights, Triton
# matrix-vector kernel for attention weights. Set EXTRA_ENV="" for the reference build (exact arithmetic).
EXTRA_ENV=${EXTRA_ENV--e INKLING_ATTN_GEMV=1 -e INKLING_MOE_TP=1 -e INKLING_ATTN_FP8=1}
# CUDA graphs for decode-only steps: required with prefix caching on (README, "Known faults").
DEFAULT_ARGS=${DEFAULT_ARGS:---gpu-memory-utilization 0.93 --max-num-seqs 16 --compilation-config.cudagraph_mode=FULL_DECODE_ONLY}

on() { if [[ $1 == local ]]; then bash -c "$2"; else ssh -n -o BatchMode=yes "$1" "$2"; fi; }

run_rank() {
  local r=$1; shift
  on "${NODES[$r]}" "mkdir -p $PROF"
  local extra=""
  [[ $r -gt 0 ]] && extra="--headless"
  [[ $r -eq 0 ]] && extra="--host $HOST --port $PORT --served-model-name $NAME"
  local ip
  ip=$(on "${NODES[$r]}" "ip -o -4 addr show $IFNAME | awk '{print \$4}' | cut -d/ -f1")
  on "${NODES[$r]}" "docker rm -f inkling-r$r >/dev/null 2>&1; docker run -d --name inkling-r$r --gpus all --ipc host --network host \
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 -v $MODEL:$MODEL:ro -v $PROF:/prof $EXTRA_ENV \
    -e NCCL_SOCKET_IFNAME=$IFNAME -e GLOO_SOCKET_IFNAME=$IFNAME -e NCCL_IB_HCA=${HCAS[$r]} -e NCCL_IB_GID_INDEX=${GID_INDEX:-3} \
    -e VLLM_HOST_IP=$ip \
    $IMAGE $MODEL --tensor-parallel-size 4 --nnodes 4 --node-rank $r --master-addr $MASTER --master-port $MASTER_PORT \
    --max-model-len $CTX --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
    --limit-mm-per-prompt '{\"image\":0,\"video\":0}' $DEFAULT_ARGS $extra $* >/dev/null"
}

case ${1:-} in
  start) shift; for r in 3 2 1 0; do run_rank $r "$@"; done; echo "started; follow with: $0 logs" ;;
  stop) for r in 0 1 2 3; do on "${NODES[$r]}" "docker rm -f inkling-r$r >/dev/null 2>&1 || true"; done; echo stopped ;;
  logs) r=${2:-0}; on "${NODES[$r]}" "docker logs --tail 40 inkling-r$r 2>&1" ;;
  *) echo "usage: $0 start [vllm args] | stop | logs [rank]"; exit 2 ;;
esac

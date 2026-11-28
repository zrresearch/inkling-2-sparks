#!/usr/bin/env bash
# Inkling 2 Small on TWO Sparks with the expert cache: experts split over the two (INKLING_MOE_TP, all 48 layers at
# TP=2), each Spark keeping SLOTS experts per layer in memory and reading the rest from its NVMe pack with O_DIRECT.
# Eager mode (the cache syncs to the CPU inside each MoE layer). Settings: nodes2.conf (see nodes2.conf.example).
#   serve2.sh start [extra vllm args] | stop | logs [rank]
set -euo pipefail
cd "$(dirname "$0")"; . ./nodes2.conf
IMAGE=${IMAGE:-inkling2-gb10:tp2}
NAME=${NAME:-Inkling-2-Small}
HOST=${HOST:-0.0.0.0}; PORT=${PORT:-8899}; CTX=${CTX:-131072}
SLOTS=${SLOTS:-176}
MASTER_PORT=${MASTER_PORT:-29711}
on() { if [[ $1 == local ]]; then bash -c "$2"; else ssh -n -o BatchMode=yes "$1" "$2"; fi; }
run_rank() {
  local r=$1; shift
  local extra=""; [[ $r -gt 0 ]] && extra="--headless"; [[ $r -eq 0 ]] && extra="--host $HOST --port $PORT --served-model-name $NAME"
  local ifn=${IFNAMES[$r]}
  local ip; ip=$(on "${NODES[$r]}" "ip -o -4 addr show $ifn | awk '{print \$4}' | cut -d/ -f1")
  on "${NODES[$r]}" "docker rm -f inkling2x-r$r >/dev/null 2>&1; docker run -d --name inkling2x-r$r --gpus all --ipc host --network host \
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 \
    -v $BASE/slim:/model:ro -v $BASE/packs:/packs:ro -v $BASE/expert-usage.npy:/stats/expert-usage.npy:ro \
    -e INKLING_ATTN_GEMV=1 -e INKLING_MOE_TP=1 -e INKLING_ATTN_FP8=1 \
    -e INKLING_EXPERT_CACHE=$SLOTS -e INKLING_EXPERT_PACK=/packs -e INKLING_EXPERT_STATS_FILE=/stats/expert-usage.npy \
    -e NCCL_SOCKET_IFNAME=$ifn -e GLOO_SOCKET_IFNAME=$ifn -e NCCL_IB_HCA=${HCAS[$r]} -e NCCL_IB_GID_INDEX=$GID_INDEX -e VLLM_HOST_IP=$ip \
    $IMAGE /model --tensor-parallel-size 2 --nnodes 2 --node-rank $r --master-addr $MASTER --master-port $MASTER_PORT \
    --max-model-len $CTX --enforce-eager --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
    --limit-mm-per-prompt '{\"image\":0,\"video\":0}' $extra $* >/dev/null"
}
case ${1:-} in
  start) shift; for r in 1 0; do run_rank $r "$@"; done; echo "started; follow with: $0 logs" ;;
  stop) for r in 0 1; do on "${NODES[$r]}" "docker rm -f inkling2x-r$r >/dev/null 2>&1 || true"; done; echo stopped ;;
  logs) r=${2:-0}; on "${NODES[$r]}" "docker logs --tail 40 inkling2x-r$r 2>&1" ;;
  *) echo "usage: $0 start|stop|logs [rank]"; exit 2 ;;
esac

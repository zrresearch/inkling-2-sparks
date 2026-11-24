# Thinking Machines' Inkling 2 Small image (vLLM), pinned to the digest this recipe was built and measured on.
FROM vllm/vllm-openai@sha256:a710ed3b5eba4e205fa1904cae57444446ead088bac575423f9b4a392a9a80a5
COPY gb10_bmm.py gb10_attention.py gb10_fp8.py expert_stats.py patch_vllm.py /opt/inkling2-gb10/
RUN python3 /opt/inkling2-gb10/patch_vllm.py

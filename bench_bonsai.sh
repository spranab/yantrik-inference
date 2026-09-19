#!/bin/bash
# Same card, cooled before every run, models alternated, two rounds.
cd /c/Users/sync/codes/bonsai/bin
Q4=$(find ~/.cache/huggingface/hub -name "Qwen3.8-27B-UD-Q4_K_M.gguf" | head -1)
declare -A M=( [Q4_K_M]="$Q4" [PQ2_0]="../models/Ternary-Bonsai-2-27B-PQ2_0.gguf" [PTQ1_0]="../models/Ternary-Bonsai-2-27B-PTQ1_0.gguf" )
for round in 1 2; do
  for name in Q4_K_M PQ2_0 PTQ1_0; do
    until [ "$(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader)" -lt 50 ]; do sleep 5; done
    t0=$(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader)
    out=$(CUDA_VISIBLE_DEVICES=0 ./llama-bench.exe -m "${M[$name]}" -ngl 99 -fa 1 -p 512,2048 -n 128 -r 3 2>&1 | grep -E "^\| qwen")
    t1=$(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader)
    echo "$out" | awk -v n="$name" -v r="$round" -v a="$t0" -v b="$t1" -F'|' '{gsub(/ /,"",$8); split($9,v," "); print "round " r "  " n "  " $8 "  " v[1] " tok/s  (card " a "C->" b "C)"}'
  done
done

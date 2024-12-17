PRETRAINED_MODEL_PATH="TraceLLM/checkpoints/llama-2-7b"
MERGED_MODEL_PATH="${PRETRAINED_MODEL_PATH}/merged"
CKPT_PATH="baseline"
GEN_TYPE="recursive_instruction"  # {baseline, recursive, recursive_instruction}
GPU=0
PORT=8000

if [ ! -d "${MERGED_MODEL_PATH}/${CKPT_PATH}" ]; then
  # Merge lora adapters
  CUDA_VISIBLE_DEVICES=$GPU python generate/load_and_merge.py --model_name_or_path=${PRETRAINED_MODEL_PATH} --lora_path=${PRETRAINED_MODEL_PATH}/${CKPT_PATH} --output_path=${MERGED_MODEL_PATH}/${CKPT_PATH}
  cp ${PRETRAINED_MODEL_PATH}/tokenizer* ${MERGED_MODEL_PATH}/${CKPT_PATH}
  cp ${PRETRAINED_MODEL_PATH}/special_tokens_map.json ${MERGED_MODEL_PATH}/${CKPT_PATH}
fi

CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.api_server --port $PORT --model ${MERGED_MODEL_PATH}/${CKPT_PATH} --gpu-memory-utilization 0.95 &
sleep 30
for temp in 0.0 0.2 0.4 0.6 0.8 1.0
do
  python generate/trace_oracle/run_validation_heatmap.py --port $PORT --summary-path heatmap_${CKPT_PATH}_temp_${temp}.summary --temperature ${temp} --tag heatmap_${CKPT_PATH} --gen-type $GEN_TYPE
done
kill %%

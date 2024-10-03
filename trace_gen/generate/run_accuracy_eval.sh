PRETRAINED_MODEL_PATH="<BASE MODEL PATH w>"
MERGED_MODEL_PATH="<FILLME>"
ADAPTER_PATH="<FILLME>"
CKPT="checkpoint"
GPU=1
PORT=8001

CKPT_PATH="checkpoint-name"
MERGED_MODEL_NAME=${CKPT}_${CKPT_PATH}
if [ ! -d "${MERGED_MODEL_PATH}/${MERGED_MODEL_NAME}" ]; then
  CUDA_VISIBLE_DEVICES=$GPU python generate/load_and_merge.py --model_name_or_path=${PRETRAINED_MODEL_PATH} --lora_path=${ADAPTER_PATH}/${CKPT}/${CKPT_PATH} --output_path=${MERGED_MODEL_PATH}/${MERGED_MODEL_NAME}
  echo "merged adapter"
  cp ${PRETRAINED_MODEL_PATH}/tokenizer* ${MERGED_MODEL_PATH}/${MERGED_MODEL_NAME}
  cp ${PRETRAINED_MODEL_PATH}/special_tokens_map.json ${MERGED_MODEL_PATH}/${MERGED_MODEL_NAME}
  echo "copied tokenizer"
fi
CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.api_server --port $PORT --model ${MERGED_MODEL_PATH}/${MERGED_MODEL_NAME} --gpu-memory-utilization 0.97 &
sleep 15
for temp in 0.0 0.2 0.4 0.6 0.8 1.0
do
  python generate/trace_oracle/run_validation_heatmap.py --port $PORT --summary-path heatmap_${MERGED_MODEL_NAME}_temp_${temp}.summary --temperature ${temp} --tag heatmap_${MERGED_MODEL_NAME} --gen-type recursive_instruction
done
kill %%

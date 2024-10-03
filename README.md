# Microservice Trace Generators with Large Language Models

## Dependencies
We use python language where dependencies can be found in `trace_gen/requirements.txt`.
Also, we install our codes as a python package we use `poetry`.
Use the following commands:
```bash
pip install poetry
# In the root directory
poetry install
cd trace_gen
pip install -r requirements.txt
```

## Data Preprocessing

We use `CallGraph` data in [Alibaba microservice v2022 traces](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022) as our training data.
Fetch the first 20 files of `CallGraph` using the scripts in Alibaba's repo and preprocess the data using our scripts.


To convert the separate API calls into call graphs, use the following command.
Make sure to change file directories before you execute the command.
```bash
> python trace_gen/preprocess/trace_to_training_data.py
```

To collect call graph stats required to generate instructions, run the following commands:
```bash
> python trace_gen/preprocess/trace_to_cg_stats.py
> python trace_gen/preprocess/merge_cg_stats.py
```

To convert the call graphs to text representations, run the following command.
Make sure to change file directories before you execute the command.

Also, make sure to set the `task_type` correctly depending on your use cases:
* `TraceGenTaskType.graph_gen_non_recursive`: tabular format
* `TraceGenTaskType.graph_gen_recursive`: recursive format
* `TraceGenTaskType.graph_gen`: instruction-tuning

```bash
> python trace/preprocess/training_data_to_text_representations.py
```

## Data Examples
We include part of training datasets in the `dataset_examples` folder:
* `tabular_dataset.txt`: dataset in tabular format
* `recursive_dataset.txt`: dataset in recursive format
* `recursive_instruction_dataset.txt`: dataset in recursive format with instructions

## Training
Model training scripts can be found in `trace_gen/train`.
* `pretraining.py`: For pretraining LLaMA-7B with trace data.
* `sft.py`: For supervised-fine-tuning the model with instruction datasets.

## Generation
To get the accuracy report, follow the script in `trace_gen/generate/run_accuracy_eval.sh`.
To run the script, prompt files and lora adapaters after training are required.

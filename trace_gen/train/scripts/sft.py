from pathlib import Path

from argparse import Namespace
import pickle
import torch
import datasets
import glob
from multiprocessing import Pool
from datasets import Dataset, Features, Value

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from huggingface_hub import Repository

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import logging

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

import arguments
from trace_gen.utils.models.lora_transformer import get_lora_config

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def setup_logging(args):
    project_name = args.model_ckpt.split("/")[-1]
    logger = logging.getLogger(__name__)
    log_dir = Path(args.save_dir) / "log/"
    log_dir.mkdir(exist_ok=True)
    filename = f"debug_{accelerator.process_index}.log"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_dir / filename), logging.StreamHandler()],
    )
    if accelerator.is_main_process:  # we only want to setup logging once
        accelerator.init_trackers(project_name, vars(args))
        run_name = accelerator.trackers[0].run.name
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        run_name = ""
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, run_name

def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        accelerator.log(metrics, step)

# Settings
parser = HfArgumentParser(arguments.TrainingArguments)
args = parser.parse_args()

# Accelerator
config = ProjectConfiguration(project_dir=args.save_dir, logging_dir="log")
accelerator = Accelerator(log_with=["wandb"], project_config=config)
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

args = Namespace(**vars(args), **acc_state)
samples_per_step = accelerator.state.num_processes * args.train_batch_size
set_seed(args.seed)

# Clone model repository
if accelerator.is_main_process:
    hf_repo = Repository(args.save_dir)

# Logging
logger, run_name = setup_logging(args)
logger.info(accelerator.state)

# Checkout new branch on repo
if accelerator.is_main_process:
    hf_repo.git_checkout(run_name, create_branch_ok=True)

# Load model and tokenizer
def create_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        args.resume_from_checkpoint if args.resume_from_checkpoint else args.save_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.save_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False
if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

### Dataset Load and Preprocessing ###

def read_pickle_file(csv_file):
    with open(csv_file, "rb") as f:
        data = pickle.load(f)
    return data
 
def load_local_dataset(csv_files):
    with Pool(processes=16) as pool: # or whatever your hardware can support
        # have your pool map the file names to dataframes
        item_list = pool.map(read_pickle_file, csv_files)
    return item_list

def load_iterable_dataset_from_csv_path(file_path):
    csv_files = sorted(glob.glob(f"{file_path}/*.pkl"))
    context_feat = Features({'instruction': Value(dtype='string', id=None), 'output': Value(dtype='string', id=None)})
    def my_gen():
        for ds in load_local_dataset(csv_files):
            for i in range(len(ds['instruction'])):
                yield {"instruction": ds["instruction"][i], "output": ds["output"][i]}
    return Dataset.from_generator(my_gen, features=context_feat)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: \n{example['output'][i]}\n ### Finish"
        output_texts.append(text)
    return output_texts

def formatting_prompts_func_wo_description(example):
    return f"{example['instruction']}\n{example['output']}"


response_template = "\n ### Answer:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

### Dataset ###
dirname = "data/CallGraph"
train_data = load_iterable_dataset_from_csv_path(dirname + "/train_data_sft")
train_data = train_data.shuffle(seed=args.seed)
validation_data = load_iterable_dataset_from_csv_path(dirname + "/validation_data_sft")

### Train ###
training_arguments = TrainingArguments(
    per_device_train_batch_size=args.train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    logging_steps=1,
    learning_rate=args.learning_rate,
    max_grad_norm=1,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    warmup_ratio=0.05,
    output_dir=args.save_dir,
    save_safetensors=True,
    group_by_length=True,
    lr_scheduler_type="cosine",
    seed=args.seed,
    report_to="wandb",
    resume_from_checkpoint=args.resume_from_checkpoint,
    save_strategy="epoch",
    # save_steps=args.save_checkpoint_steps,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=validation_data,
    peft_config=get_lora_config(),
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=formatting_prompts_func,
    # data_collator=collator,
)

trainer.train()
trainer.save_model(output_dir=args.save_dir)

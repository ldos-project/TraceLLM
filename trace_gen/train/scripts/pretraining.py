import logging
import os
import time
from argparse import Namespace
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import ProjectConfiguration
from arguments import TrainingArguments
from huggingface_hub import Repository
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, get_scheduler, set_seed
import glob
from multiprocessing import Pool
from datasets import Dataset,Features,Value

from trace_gen.utils.models.lora_transformer import add_lora_adapter
from trace_gen.train.dataset.call_graph_dataset import CallGraphDataset
from trace_gen.utils.preprocessing.read_file import file_to_samples

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

def read_csv(csv_file):
    if "merged_CallGraph_edges" in csv_file:
        return file_to_samples(csv_file, lambda x: True if x.startswith("<edges of S_") else False)
    elif "prompt_CallGraph" in csv_file:
        return file_to_samples(csv_file)
    else:
        return file_to_samples(csv_file, lambda x: True if "[GENERATE GRAPH]" in x else False)
 
def load_dataset_from_csv_path(file_path):
    csv_files = sorted(glob.glob(f"{file_path}/*.csv"))
    context_feat = Features({'content': Value(dtype='string', id=None)})
    def my_gen():
        for i in load_local_dataset(csv_files):
            yield {"content": i}
    return Dataset.from_generator(my_gen, features=context_feat)

def load_local_dataset(csv_files):
    with Pool(processes=16) as pool: # or whatever your hardware can support
        # have your pool map the file names to dataframes
        item_list = pool.map(read_csv, csv_files)

    return iter([item for sublist in item_list for item in sublist])

def create_dataloaders(args):
    dirname = "<REPLACE ME>"

    train_data = load_dataset_from_csv_path(dirname + "/training_trace_text_representaion")
    train_data = train_data.shuffle(seed=args.seed)
    valid_data = load_dataset_from_csv_path(dirname + "/validation_trace_text_representation")

    config_model = accelerator.unwrap_model(model).config

    train_dataset = CallGraphDataset(
        tokenizer, train_data, infinite=True, seq_length=config_model.max_position_embeddings,
    )
    valid_dataset = CallGraphDataset(
        tokenizer, valid_data, infinite=False, seq_length=config_model.max_position_embeddings,
    )
    train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader


def get_grouped_params(model, args, no_decay=["bias", "ln_1.weight", "ln_2.weight", "ln_f.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        accelerator.log(metrics, step)


def compute_tflops(elapsed_time, accelerator, args):
    # TFLOPs formula (from Equation 3 in Section 5.1 of https://arxiv.org/pdf/2104.04473.pdf).
    config_model = accelerator.unwrap_model(model).config
    checkpoint_factor = 4 if args.gradient_checkpointing else 3
    batch_size = args.train_batch_size * accelerator.state.num_processes * args.gradient_accumulation_steps
    factor = 24 * checkpoint_factor * batch_size * config_model.max_position_embeddings * config_model.num_hidden_layers * (config_model.hidden_size**2)
    flops_per_iteration = factor * (
        1.0
        + (config_model.max_position_embeddings / (6.0 * config_model.hidden_size))
        + (tokenizer.vocab_size / (16.0 * config_model.num_hidden_layers * config_model.hidden_size))
    )
    tflops = flops_per_iteration / (elapsed_time * accelerator.state.num_processes * (10**12))
    return tflops


def evaluate(args):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break
    losses = torch.cat(losses)
    loss = losses[: eval_dataloader.dataset.current_size].mean()
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


# Settings
parser = HfArgumentParser(TrainingArguments)
args = parser.parse_args()

# Accelerator
config = ProjectConfiguration(project_dir=args.save_dir, logging_dir="log")
accelerator = Accelerator(log_with=["wandb", "tensorboard"], project_config=config)
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
model = AutoModelForCausalLM.from_pretrained(
    args.save_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(args.save_dir)

# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders(args)

# load in the weights and states from a previous save
if args.lora:
    if args.resume_from_checkpoint:
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        # accelerator.load_state(args.resume_from_checkpoint)
        model.load_adapter(args.resume_from_checkpoint, "trained_adapter")
        model.set_adapter(["trained_adapter"])
        path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]
        if "phase_1" in training_difference:
            resume_step = 0
        else:
            resume_step = int(training_difference.replace("step_", ""))
    else:
        add_lora_adapter(model)
else:
    raise ValueError("'lora' option is required to run this script.")

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.max_train_steps,
)
accelerator.register_for_checkpointing(lr_scheduler)

def get_lr():
    return optimizer.param_groups[0]["lr"]

# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Train model
model.train()
completed_steps = 0
t_start = time.time()
loss_tracking = 0
current_step = 0

logger.info("<Training start>")
max_train_steps = args.max_train_steps + completed_steps
for step, batch in enumerate(train_dataloader, start=1):
    if args.resume_from_checkpoint and step < resume_step:
        continue  # we need to skip steps until we reach the resumed step
    loss = model(batch, labels=batch, use_cache=False).loss
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
    loss_tracking += avg_loss.item() / args.gradient_accumulation_steps
    loss = loss / args.gradient_accumulation_steps
    if step % args.gradient_accumulation_steps != 0:
        # Prevent backward from doing gradient all_reduce in every step
        if accelerator.distributed_type == DistributedType.MULTI_GPU:
            with model.no_sync():
                accelerator.backward(loss)
        else:
            accelerator.backward(loss)
    else:
        lr = get_lr()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        elapsed_time = time.time() - t_start
        tflops = compute_tflops(elapsed_time, accelerator, args)
        log_metrics(
            current_step + step,
            {
                "steps": completed_steps,
                "loss/train": loss_tracking,
                "lr": lr,
                "tflops": tflops,
                "time_per_iteration": elapsed_time,
            },
        )
        t_start = time.time()
        loss_tracking = 0
    completed_steps += 1
    if step % args.save_checkpoint_steps == 0:
        logger.info("Evaluating and saving model checkpoint")
        ckpt_name = f"{args.resume_from_checkpoint.split('/')[-1]}_step_{step}" if args.resume_from_checkpoint else f"step_{step}"
        save_dir = os.path.join(args.save_dir, "baseline_" + ckpt_name)
        os.makedirs(save_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)

        model.train()
    if completed_steps >= max_train_steps:
        break

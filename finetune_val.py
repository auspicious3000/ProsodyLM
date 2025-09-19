import os
import time
import fire
import torch
import wandb
import numpy as np
from typing import List
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
import inspect

from peft import (
    LoraConfig,
    get_peft_model
)
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from transformers.integrations import WandbCallback
from llm.models_llama import LlamaForCausalLM_val as ModelForCausalLM
from llm.models_llama import FreezableEmbedding, FreezableLMHead
from trainer_3l import TrainerCustom as Trainer
from torch.optim import AdamW

os.environ["TOKENIZERS_PARALLELISM"] = "false"
rng = np.random.default_rng(1)


def print_arguments():
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    arg_info = inspect.getargvalues(frame)
    
    print(f"Arguments for {func_name}:")
    for name in arg_info.args:
        value = arg_info.locals[name]
        print(f"{name}: {value}")
    if arg_info.varargs:
        for i, value in enumerate(arg_info.locals[arg_info.varargs]):
            print(f"{arg_info.varargs}[{i}]: {value}")
    if arg_info.keywords:
        for name, value in arg_info.locals[arg_info.keywords].items():
            print(f"{name}: {value}")


class TimeLoggingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = (time.time() - self.epoch_start_time) / 60
        #if control.should_log:
        if state.is_world_process_zero:
            print(f"TimeLoggingCallback: Epoch {int(state.epoch)} took {epoch_time:.3f} minutes")


class DelayedCheckpointCallback(TrainerCallback):
    def __init__(self, delay_steps):
        self.delay_steps = delay_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step < self.delay_steps:
            control.should_save = False


class GenerateTokenizePrompt:
    def __init__(self, tokenizer, n_prs_bins, train_on_inputs):
        self.tokenizer = tokenizer
        self.len_tokenizer = len(tokenizer)
        self.n_prs_bins = n_prs_bins
        self.colon_token_id = tokenizer.convert_tokens_to_ids(':')
        self.train_on_inputs = train_on_inputs

    def tokenize(self, prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )
        assert result["input_ids"][-1] == self.tokenizer.eos_token_id
        result["labels"] = result["input_ids"].copy()

        return result

    def process_prompt(self, data_point):
        
        model_content = data_point['instruction']+' '+data_point['input_cur']
        model_message = {'role': 'assistant', 'content': model_content}
        curr_messages = [model_message]
    
        full_prompt = self.tokenizer.apply_chat_template(curr_messages, tokenize=False)
        
        tokenized_full_prompt = self.tokenize(full_prompt)
        tokenized_full_prompt["num_tokens"] = len(tokenized_full_prompt["input_ids"])
    
        labels = np.array(tokenized_full_prompt["labels"])
        response_token_ids_end_idx = None
        spk_bins_idx = np.where(labels >= self.len_tokenizer-self.n_prs_bins)[0]        
        assert labels[spk_bins_idx[0]+1] == self.colon_token_id
        response_token_ids_end_idx = spk_bins_idx[0]+1
        assert response_token_ids_end_idx is not None            
        labels[:response_token_ids_end_idx+1] = -100
        tokenized_full_prompt["labels"] = labels.tolist()
            
        return tokenized_full_prompt


# Custom grouped parameters
def get_lora_embed_head_parameters(model, base_lr, embed_lr_scale=0.2, weight_decay=0.01):
    lora_params = []
    original_embed_head_params = []
    new_embed_head_params = []

    trainable_params = set()
    matched_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable_params.add(name)

        if "lora_A" in name or "lora_B" in name:
            lora_params.append(param)
            matched_params.add(name)
        elif "original_embed" in name or "original_head" in name:
            original_embed_head_params.append(param)
            matched_params.add(name)
        elif "new_embed" in name or "new_head" in name:
            new_embed_head_params.append(param)
            matched_params.add(name)

    assert matched_params == trainable_params, (
        f"Unmatched trainable parameters found: {trainable_params.symmetric_difference(matched_params)}"
    )

    assert lora_params, "Expected LoRA parameters (lora_A / lora_B), but found none."
    assert original_embed_head_params, "Expected original_embed and/or original_head parameters, but found none."
    assert new_embed_head_params, "Expected new_embed and/or new_head parameters, but found none."

    return [
        {
            "params": lora_params,
            "weight_decay": weight_decay,
            "lr": base_lr,
        },
        {
            "params": original_embed_head_params,
            "weight_decay": 0.0,
            "lr": base_lr * embed_lr_scale,
        },
        {
            "params": new_embed_head_params,
            "weight_decay": 0.0,
            "lr": base_lr,
        },
    ]


def train(
    # model/data params
    base_model: str = "./",
    data_path: str = "./",
    data_val_path: str = "./",
    output_dir: str = "./",
    # training hyperparams
    warmup_ratio: float = 0.1,
    logging_steps: int = 100,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4000,
    val_set_size: int = 2000,
    num_proc: int = 128,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = True, 
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "false",  # options: false | gradients | all
    wandb_log_model: str = "false",  # options: false | true
    wandb_last_run_id: str = None,
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    eval_steps: int = 100,
    save_steps: int = 100, 
    save_start_steps: int = 100,
    save_total_limit: int = 50,
    disable_tqdm: bool = False,
    # tokenizer params
    n_prs_bins = 513,
):
    if int(os.environ.get("RANK", 0)) == 0:
        print_arguments()
    assert (
        base_model
    ), "Please specify a --base_model"
    assert num_proc > 0
    
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    world_size = int(os.environ.get("WORLD_SIZE"))
    gradient_accumulation_steps = batch_size // (micro_batch_size * world_size)
    ddp = (world_size != 1)        
    
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )        
    
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_API_KEY"] = ''
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if resume_from_checkpoint is not None and wandb_last_run_id is not None:
        if int(os.environ["RANK"]) == 0:
            wandb_run = wandb.init(
                project=os.environ["WANDB_PROJECT"],
                id=wandb_last_run_id,
                resume="must",
            )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"  # Allow batched inference
    tokenizer.init_kwargs["padding_side"] = tokenizer.padding_side
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ['<SIL>','[SEP_1]','[SEP_2]']}, 
        replace_additional_special_tokens=False
    )
    num_tks = [f"<|num_tk_{i}|>" for i in range(n_prs_bins)]
    tokenizer.add_special_tokens(
        {"additional_special_tokens": num_tks}, 
        replace_additional_special_tokens=False
    )
    tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))

    generate_and_tokenize_prompt = GenerateTokenizePrompt(tokenizer, n_prs_bins, train_on_inputs)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)

    if disable_tqdm: disable_progress_bar()
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=1638
        )
        train_data = (
            train_val["train"].map(generate_and_tokenize_prompt.process_prompt, 
                                   load_from_cache_file=True,
                                   num_proc=num_proc)
        )
        val_data = (
            train_val["test"].map(generate_and_tokenize_prompt.process_prompt, 
                                  load_from_cache_file=True,
                                  num_proc=num_proc)
        )
        all_ntokens_val = [len_data for len_data in val_data['num_tokens']]
        if int(os.environ.get("RANK", 0)) == 0:
            print("Max number of valid tokens: ", max(all_ntokens_val))
    else:
        train_data = data["train"].map(generate_and_tokenize_prompt.process_prompt, 
                                       load_from_cache_file=True, 
                                       num_proc=num_proc)

        if data_val_path.endswith(".json") or data_val_path.endswith(".jsonl"):
            data_val = load_dataset("json", data_files=data_val_path)
            val_data = data_val["train"].map(generate_and_tokenize_prompt.process_prompt, 
                                             load_from_cache_file=True, 
                                             num_proc=num_proc)
            all_ntokens_val = [len_data for len_data in val_data['num_tokens']]
            assert max(all_ntokens_val) < cutoff_len
            if int(os.environ.get("RANK", 0)) == 0:
                print("Max number of valid tokens: ", max(all_ntokens_val))

    all_ntokens = [len_data for len_data in train_data['num_tokens']]
    if int(os.environ.get("RANK", 0)) == 0:
        print("Max number of train tokens: ", max(all_ntokens))
    if max(all_ntokens) > cutoff_len:
        pct = np.sum(np.array(all_ntokens) > cutoff_len) / len(all_ntokens) * 100
        print(f"{pct:.4f}% overlength train tokens")
        train_data = train_data.filter(lambda x: x['num_tokens'] < cutoff_len, 
                                       load_from_cache_file=True, num_proc=num_proc)
    if max(all_ntokens_val) > cutoff_len:
        pct = np.sum(np.array(all_ntokens_val) > cutoff_len) / len(all_ntokens_val) * 100
        print(f"{pct:.4f}% overlength valid tokens")
        val_data = val_data.filter(lambda x: x['num_tokens'] < cutoff_len, 
                                   load_from_cache_file=True, num_proc=num_proc)

    len_tokenizer = len(tokenizer)
    llama_config = AutoConfig.from_pretrained(base_model)
    llama_config.num_start_id = len_tokenizer - n_prs_bins
    llama_config.sil_token_id = tokenizer.convert_tokens_to_ids('<SIL>')
    llama_config.sep1_token_id = tokenizer.convert_tokens_to_ids('[SEP_1]')
    llama_config.sep2_token_id = tokenizer.convert_tokens_to_ids('[SEP_2]')
    llama_config.invalid_num_id = tokenizer.convert_tokens_to_ids(f'<|num_tk_{n_prs_bins-1}|>')
    llama_config.lambda_num = 1
    
    llama_config.save_pretrained(output_dir)
    model = ModelForCausalLM.from_pretrained(
        base_model,
        config=llama_config,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    original_vocab_size = model.config.vocab_size
    model.resize_token_embeddings(len_tokenizer)

    # Get the expanded embedding layer
    embed_layer = model.get_input_embeddings()
    original_embeddings = embed_layer.weight[:original_vocab_size].detach().clone()
    model.set_input_embeddings(
        FreezableEmbedding(original_embeddings, (len_tokenizer-original_vocab_size))
    )
    lm_head = model.get_output_embeddings()
    original_lm_head = lm_head.weight[:original_vocab_size].detach().clone()
    model.set_output_embeddings(
        FreezableLMHead(original_lm_head, (len_tokenizer-original_vocab_size))
    )
    
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        modules_to_save=["embed_tokens", "lm_head"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )
    model = get_peft_model(model, config)
    
    
    optimizer = AdamW(
        get_lora_embed_head_parameters(model, learning_rate, embed_lr_scale=0.1, weight_decay=0.01)
    )

    if int(os.environ.get("RANK", 0)) == 0:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        optimizers=(optimizer, None),
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            bf16=True,
            logging_steps=logging_steps,
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            dataloader_num_workers=4,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else "none",
            run_name=wandb_run_name if use_wandb else None,
            remove_unused_columns=True,
            disable_tqdm=disable_tqdm,
            log_on_each_node=False,
            length_column_name="num_tokens",
            seed=1638,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[
            TimeLoggingCallback(), 
            DelayedCheckpointCallback(save_start_steps),
        ],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
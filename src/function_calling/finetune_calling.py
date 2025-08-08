"""Fine-tuning for Function-Calling (script form of Bonus Unit 1).

This script reproduces the notebook flow as CLI commands:
- train: preprocess dataset, add special tokens, LoRA fine-tune, save locally
- push: push the trained adapter and tokenizer to the Hub
- test: load base model + adapter and run a short generation

Environment variables:
- HF_TOKEN: Required for pushing to the Hub (and recommended for model pulls)

Examples:
    # 1) Train a small run (CPU or GPU). Adjust limits for real training
    python -m src.function_calling.bonus_unit1 train \
        --model google/gemma-2-2b-it \
        --dataset Jofthomas/hermes-function-calling-thinking-V1 \
        --output-dir gemma-2b-fc-demo \
        --train-limit 100 --test-limit 10

    # 2) Push to your namespace on the Hub (requires HF_TOKEN)
    python -m src.function_calling.bonus_unit1 push \
        --repo your-username/gemma-2b-fc-demo

    # 3) Test the adapter from Hub
    python -m src.function_calling.bonus_unit1 test \
        --peft-repo your-username/gemma-2b-fc-demo
"""

from __future__ import annotations

import argparse
import os
from enum import Enum
from typing import Dict, Tuple

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


def login_hf_if_token_present() -> None:
    """Login to Hugging Face using HF_TOKEN env var if available (best effort)."""

    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] HF login skipped: {exc}")


class ChatmlSpecialTokens(str, Enum):
    """Special tokens used by the custom chat template and tags."""

    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call = "<tool_call>"
    eotool_call = "</tool_call>"
    tool_response = "<tool_response>"
    eotool_response = "</tool_response>"
    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def preprocess_dataset(model_name: str, dataset_name: str, train_limit: int, test_limit: int):
    """Load and preprocess dataset; return (dataset_dict, tokenizer).

    - Renames column to `messages`
    - Adjusts chat_template on tokenizer
    - Flattens messages into a single `text` using tokenizer.apply_chat_template
    - Limits train/test sizes for quick experiments
    """

    try:
        from transformers import AutoTokenizer, set_seed
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Install transformers and datasets.") from exc

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = (
        "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}"
        "{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    )

    def preprocess(sample: Dict) -> Dict:
        messages = list(sample["messages"])  # copy
        if messages and messages[0].get("role") == "system":
            system_message_content = messages[0]["content"]
            messages[1]["content"] = (
                system_message_content
                + "Also, before making a call to a function take the time to plan the function to take. "
                  "Make that thinking process between <think>{your thoughts}</think>\n\n"
                + messages[1]["content"]
            )
            messages.pop(0)
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    ds = load_dataset(dataset_name)
    ds = ds.rename_column("conversations", "messages") if "conversations" in ds["train"].features else ds
    ds = ds.map(preprocess, remove_columns=[c for c in ds["train"].column_names if c != "text"])
    ds = ds["train"].train_test_split(0.1, seed=42)
    if train_limit:
        ds["train"] = ds["train"].select(range(min(train_limit, len(ds["train"]))))
    if test_limit:
        ds["test"] = ds["test"].select(range(min(test_limit, len(ds["test"]))))
    return ds, tokenizer


def load_base_model_and_tokenizer(model_name: str):
    """Load base model and tokenizer, extend with special tokens, return (model, tokenizer)."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Install transformers and torch.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list(),
    )
    tokenizer.chat_template = (
        "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}"
        "{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    try:
        model.to(torch.bfloat16)
    except Exception:
        pass
    return model, tokenizer


def build_peft_config(r: int, alpha: int, dropout: float):
    """Construct a PEFT LoRA config with default target modules for decoder-only LMs."""

    try:
        from peft import LoraConfig, TaskType
    except Exception as exc:
        raise RuntimeError("Install peft.") from exc

    target_modules = [
        "gate_proj",
        "q_proj",
        "lm_head",
        "o_proj",
        "k_proj",
        "embed_tokens",
        "down_proj",
        "up_proj",
        "v_proj",
    ]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )


def train(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    train_limit: int,
    test_limit: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    num_train_epochs: int,
    max_seq_length: int,
) -> str:
    """Fine-tune with LoRA and save artifacts to output_dir.

    Returns the output_dir path.
    """

    login_hf_if_token_present()

    ds, _ = preprocess_dataset(model_name, dataset_name, train_limit, test_limit)
    model, tokenizer = load_base_model_and_tokenizer(model_name)

    try:
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise RuntimeError("Install trl.") from exc

    peft_config = build_peft_config(lora_r, lora_alpha, lora_dropout)
    training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        save_strategy="no",
        eval_strategy="epoch",
        logging_steps=5,
        learning_rate=learning_rate,
        max_grad_norm=1.0,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        bf16=True,
        hub_private_repo=False,
        push_to_hub=False,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=True,
        max_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model()
    # Save tokenizer (includes special tokens + chat template)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def push_to_hub(repo_id: str, output_dir: str) -> Tuple[str, str]:
    """Push local `output_dir` (adapter) and tokenizer to the Hub under repo_id.

    Returns (model_repo, tokenizer_repo) which are identical in this layout.
    """

    login_hf_if_token_present()
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except Exception as exc:
        raise RuntimeError("Install huggingface_hub.") from exc

    api = HfApi()
    create_repo(repo_id, exist_ok=True, repo_type="model")
    # Upload entire output_dir structure (adapter weights and config)
    upload_folder(repo_id=repo_id, folder_path=output_dir, repo_type="model")
    return repo_id, repo_id


def test_adapter(peft_repo_id: str, max_new_tokens: int) -> str:
    """Load base model and LoRA adapter; generate from a small prompt.

    Returns the decoded text.
    """

    login_hf_if_token_present()
    try:
        import torch
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Install peft and transformers.") from exc

    config = PeftConfig.from_pretrained(peft_repo_id)
    base_name = config.base_model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(base_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(peft_repo_id)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, peft_repo_id)
    try:
        model.to(torch.bfloat16)
    except Exception:
        pass
    model.eval()

    prompt = (
        "<bos><start_of_turn>human\n"
        "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags."
        "You may call one or more functions to assist with the user query.\n"
        "Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number'}, 'from_currency': {'type': 'string'}, 'to_currency': {'type': 'string'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}] </tools>\n"
        "Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}\n"
        "For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{tool_call}\n</tool_call>\n"
        "Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n"
        "Hi, I need to convert 500 USD to Euros. Can you help me with that?<end_of_turn><eos>\n"
        "<start_of_turn>model\n<think>"
    )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    try:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    except Exception:
        pass
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.01,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output[0])
    print(text)
    return text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune for Function-Calling (LoRA)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Run preprocessing and LoRA fine-tuning")
    p_train.add_argument("--model", default="google/gemma-2-2b-it")
    p_train.add_argument("--dataset", default="Jofthomas/hermes-function-calling-thinking-V1")
    p_train.add_argument("--output-dir", default="gemma-2-2b-fc-lora")
    p_train.add_argument("--train-limit", type=int, default=100)
    p_train.add_argument("--test-limit", type=int, default=10)
    p_train.add_argument("--lora-r", type=int, default=16)
    p_train.add_argument("--lora-alpha", type=int, default=64)
    p_train.add_argument("--lora-dropout", type=float, default=0.05)
    p_train.add_argument("--learning-rate", type=float, default=1e-4)
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--max-seq-length", type=int, default=1500)

    p_push = sub.add_parser("push", help="Push the trained adapter/tokenizer to the Hub")
    p_push.add_argument("--repo", required=True, help="e.g., your-username/gemma-2b-fc-demo")
    p_push.add_argument("--output-dir", default="gemma-2-2b-fc-lora")

    p_test = sub.add_parser("test", help="Load base+adapter and generate")
    p_test.add_argument("--peft-repo", required=True)
    p_test.add_argument("--max-new-tokens", type=int, default=300)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "train":
        path = train(
            model_name=args.model,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            max_seq_length=args.max_seq_length,
        )
        print(f"Saved to {path}")
        return

    if args.command == "push":
        model_repo, _ = push_to_hub(args.repo, args.output_dir)
        print(f"Pushed to {model_repo}")
        return

    if args.command == "test":
        _ = test_adapter(args.peft_repo, args.max_new_tokens)
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()



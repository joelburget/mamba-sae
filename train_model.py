import argparse
import os

import torch
from datasets import load_dataset

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    MambaConfig,
    MambaForCausalLM,
)
from params import d_model, dataset_path, model_dir, model_path
from tokenizer import Tokenizer

os.environ["WANDB_PROJECT"] = "mamba-1l"
os.environ["WANDB_LOG_MODEL"] = "false"


class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
        )

        return lm_loss

    def on_train_end(self, args):
        self.save_model(model_dir, False)

    def save_model(self, output_dir: str, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), model_path)


def run(args):
    model = MambaForCausalLM(MambaConfig(n_layer=1, d_model=d_model))
    tokenizer = Tokenizer()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = load_dataset(dataset_path, streaming=True)
    dataset = dataset.map(lambda example: tokenizer(example["text"]))
    # GPU not big enough to handle larger!
    dataset = dataset.filter(lambda example: len(example["input_ids"]) < 7_500)

    trainer = MambaTrainer(
        model=model,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=model_dir,
            logging_steps=50,
            save_steps=500,
            save_total_limit=8,
            max_steps=100_000,
            report_to="wandb",
        ),
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(model_dir, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    run(args)

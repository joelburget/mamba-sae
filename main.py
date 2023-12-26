import argparse
import os

import torch
from datasets import load_dataset
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)


class MambaTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                           labels.view(-1))

        return lm_loss

    def save_model(self, output_dir: str, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)


def run(args):

    model = MambaLMHeadModel(MambaConfig(n_layer=1))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)

    dataset = load_dataset("monology/pile-uncopyrighted")

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
            output_dir="output",
            logging_steps=50,
            save_steps=500,
        ),
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer",
                        type=str,
                        default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    run(args)

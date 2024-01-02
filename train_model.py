import argparse
import os

import torch
from datasets import load_dataset
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

os.environ["WANDB_PROJECT"] = "mamba-1l"
os.environ["WANDB_LOG_MODEL"] = "false"

d_model = 640

MambaConfig.to_dict = lambda self: dict(
    d_model=self.d_model,
    n_layer=self.n_layer,
    vocab_size=self.vocab_size,
    ssm_cfg=self.ssm_cfg,
    rms_norm=self.rms_norm,
    residual_in_fp32=self.residual_in_fp32,
    fused_add_norm=self.fused_add_norm,
    pad_vocab_size_multiple=self.pad_vocab_size_multiple,
)


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

    def on_train_end(self, args):
        self.save_model(args.output_dir, False)

    def save_model(self, output_dir: str, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)


def run(args):
    model = MambaLMHeadModel(MambaConfig(n_layer=1, d_model=d_model))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)

    dataset = load_dataset("/mnt/hddraid/pile-uncopyrighted", streaming=True)
    dataset = dataset.map(lambda example: tokenizer(example["text"]))
    # GPU not big enough to handle larger!
    dataset = dataset.filter(lambda example: len(example["input_ids"]) < 7_500)

    output_dir = args.output_dir
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
            output_dir=output_dir,
            logging_steps=50,
            save_steps=500,
            max_steps=51_001,
            report_to="wandb",
        ),
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir, False)


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
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    run(args)

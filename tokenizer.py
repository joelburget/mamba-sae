from transformers import AutoTokenizer

# Turn off "Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained."
# import logging
# logging.getLogger("transformers.tokenization_utils_base").disabled = True


class Tokenizer:
    def __init__(self):
        self.wrapped = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b", padding_side="left"
        )
        self.wrapped.eos_token = "<|endoftext|>"
        self.wrapped.pad_token = self.wrapped.eos_token
        self.bos_token_id = self.wrapped.bos_token_id
        self.pad_token_id = self.wrapped.pad_token_id

    def __call__(self, *args, **kwargs):
        result = self.wrapped(*args, **kwargs)
        del result["attention_mask"]
        return result

    def add_tokens(self, *args, **kwargs):
        return self.wrapped.add_tokens(*args, **kwargs)

    def add_special_tokens(self, *args, **kwargs):
        return self.wrapped.add_special_tokens(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.wrapped.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.wrapped.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.wrapped.decode(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.wrapped.encode(*args, **kwargs)

    def convert_ids_to_tokens(self, *args, **kwargs):
        return self.wrapped.convert_ids_to_tokens(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self.wrapped.convert_tokens_to_ids(*args, **kwargs)

    def get_added_vocab(self, *args, **kwargs):
        return self.wrapped.get_added_vocab(*args, **kwargs)

    def num_special_tokens_to_add(self, *args, **kwargs):
        return self.wrapped.num_special_tokens_to_add(*args, **kwargs)

    def set_truncation_and_padding(self, *args, **kwargs):
        return self.wrapped.set_truncation_and_padding(*args, **kwargs)

    def train_new_from_iterator(self, *args, **kwargs):
        return self.wrapped.train_new_from_iterator(*args, **kwargs)

    def pad(self, *args, **kwargs):
        result = self.wrapped.pad(*args, **kwargs)
        del result["attention_mask"]
        return result

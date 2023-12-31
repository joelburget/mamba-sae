from transformers import AutoTokenizer


class Tokenizer:

    def __init__(self):
        self.wrapped = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b",
                                                     padding_side='left')
        self.wrapped.eos_token = "<|endoftext|>"
        self.wrapped.pad_token = self.wrapped.eos_token

    def __getattr__(self, name):
        """
        Delegate to the wrapped object for any method or attribute not defined in this class.
        """

        def method(*args, **kwargs):
            result = getattr(self.wrapped, name)(*args, **kwargs)
            if name == 'pad':
                del result['attention_mask']
            return result

        return method

    def __call__(self, *args, **kwargs):
        result = self.wrapped(*args, **kwargs)
        del result['attention_mask']
        return result

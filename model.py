from typing import Any
import random
import logging

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_distilgpt2() -> Any:
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2', pad_token_id=tokenizer.eos_token_id)
    return model, tokenizer


def generate_distilgpt2(model: Any, tokenizer: Any, input_text: str) -> str:
    input_text = '[QUESTION]: ' + input_text + '[ANSWER]:'
    input_ids = tokenizer.encode(input_text)
    sample_outputs = model.generate(
        torch.tensor([input_ids]),
        do_sample=True, 
        max_length=len(input_ids) + 50, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=5
    )
    output_ids = random.choice(sample_outputs)
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    answer = output[len(input_text):]
    return answer


def load_gpt2() -> Any:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    return model, tokenizer


class Model():
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model, self.tokenizer = self._load()

    def _load(self) -> Any:
        if self.model_name == 'distilgpt2':
            return load_distilgpt2()
        if self.model_name == 'gpt2':
            return load_gpt2()

    def generate_text(self, title: str = '', text: str = '') -> str:
        input_text = title + ' ' + text
        if self.model_name == 'distilgpt2':
            return generate_distilgpt2(self.model, self.tokenizer, input_text)
        if self.model_name == 'gpt2':
            return generate_distilgpt2(self.model, self.tokenizer, input_text)


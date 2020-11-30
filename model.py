from typing import Any
import random
import logging
import pathlib
import re
import gdown
import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def load_distilgpt2_custom_1() -> Any:
    bos_token = '<|start|>'
    ans_token = ' [ANSWER] '
    eos_token = '<|end|>'
    pad_token = '<|pad|>'
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', 
                                                bos_token=bos_token, 
                                                eos_token=eos_token, 
                                                pad_token=pad_token,)
    model = GPT2LMHeadModel.from_pretrained("distilgpt2",)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    path_to_checkpoint = f'{CURRENT_DIR}/distilgpt2_ep1.ph'
    if not os.path.exists(path_to_checkpoint):
        gdown.download('https://drive.google.com/uc?id=1a9bAXF6lmwwjN1-VwVqCfUu21vXVTme2', path_to_checkpoint, quiet=False)
    model.load_state_dict(torch.load(path_to_checkpoint, map_location=device))
    model.to(device)
    return model, tokenizer

def generate_distilgpt2_custom_1(model: Any, tokenizer: Any, input_text: str, num_return_sequences: int=1) -> [str]:
    bos_token = '<|start|>'
    ans_token = ' [ANSWER] '
    eos_token = '<|end|>'
    pad_token = '<|pad|>'
    input_text = bos_token + input_text + ans_token
    encodings_dict = tokenizer(input_text, truncation=True, max_length=1024)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    input_ids = torch.tensor(encodings_dict['input_ids']).to(device)
    greedy_output = model.generate(input_ids.unsqueeze(0),                             
                                max_length=len(input_ids) + 140,
                                num_beams=10,
                                no_repeat_ngram_size=2, 
                                num_return_sequences=num_return_sequences,
                                early_stopping=True)
    answer = []
    for output in greedy_output:
        answer.append(tokenizer.decode(output[len(input_ids):], skip_special_tokens=True))
    return answer

def load_distilgpt2() -> Any:
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2', pad_token_id=tokenizer.eos_token_id)
    return model, tokenizer


def generate_distilgpt2(model: Any, tokenizer: Any, input_text: str, num_return_sequences: int=1) -> [str]:
    input_text = '[QUESTION]: ' + input_text + '[ANSWER]:'
    input_ids = tokenizer.encode(input_text)
    sample_outputs = model.generate(
        torch.tensor([input_ids]),
        do_sample=True, 
        max_length=len(input_ids) + 140, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=num_return_sequences
    )
    for output in sample_outputs:
        answer.append(tokenizer.decode(output[len(input_ids):], skip_special_tokens=True))
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
        if self.model_name == 'distil_gpt2_1':
            return load_distilgpt2_custom_1()

    def _process_output(self, text: str) -> str:
        # clear end
        end_punc = {'.', '!', '?'}
        if set(text) & end_punc:
            end_index = len(text) - 1
            while text[end_index] not in end_punc:
                end_index -= 1
            text = text[:end_index + 1]
        
        # remove ANSWER
        text = text.replace('[ANSWER]:', '').replace('[ANSWER]', '').replace('[ANSWER:', '')

        # remove multiple spaces
        text = re.sub(' +', ' ', text)

        return text

    def generate_text(self, title: str = '', text: str = '', num_return_sequences: int=1) -> [str]:
        input_text = title + ' ' + text
        output = ''
        if self.model_name == 'distilgpt2':
            output = generate_distilgpt2(self.model, self.tokenizer, input_text, num_return_sequences)
        if self.model_name == 'gpt2':
            output = generate_distilgpt2(self.model, self.tokenizer, input_text, num_return_sequences)
        if self.model_name == 'distil_gpt2_1':
            output = generate_distilgpt2_custom_1(self.model, self.tokenizer, input_text, num_return_sequences)
        output = [self._process_output(text) for text in output]
        return output


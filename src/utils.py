import pathlib
from typing import Any
import os
import re

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gdown

PARENT_DIR = pathlib.Path(__file__).parent.parent.absolute()

def contains_stop_words(text: str) -> bool:
    stop_words = [
        '[serious]', 'blm', 'abuser', 'abusing', 'racist', 
        'racism', 'shooting', 'shooter', 'police',
        'sexism', 'feminis', 'of colour', 'black people',
        'black wom', 'black man', 'black men', 'weapon',
        'muslim', 'islam', 'terroris', ' rape', 'rapist',
        'black lives', 'covid', 'coronavirus', 'pandemic'
    ]
    for stop_word in stop_words:
        if stop_word in text.lower():
            return True
    return False

def process_output(text: str) -> str:
        # Отрезаем, если модель сама начала вопросы генерить
        # Еще отрезаем все, что генерится в новом абзаце, часто оно плохое
        text = text.strip(' ')
        text = text.strip('\n')
        stop_pattern = re.compile(r'[A-Z]:|\n\n')
        match = stop_pattern.search(text)
        stop_position = match.start() if match else len(text)
        text = text[:stop_position]
        # Отрезание незаконченных фраз
        end_punc = {'.', '!', '?'}
        if set(text) & end_punc:
            end_index = len(text) - 1
            while text[end_index] not in end_punc:
                end_index -= 1
            text = text[:end_index + 1]
        # Удаление технических токенов
        text = text.replace('[ANSWER]:', '').replace('[ANSWER]', '').replace('[ANSWER:', '')
        text = text.replace(' Q:', '').replace(' A:', '')
        # Удаление повторяющихся пробелов
        text = re.sub(' +', ' ', text)
        # Почти всегда ответы с таким текстом плохие
        if '&amp,#x200B' in text:
            return None
        # Удаление строк с "Edit:"
        lines = text.split('\n')
        lines_filtered = []
        for line in lines:
            if line.lower().startswith('edit:'):
                break
            lines_filtered.append(line)
        text = '\n'.join(lines_filtered)
        # Проверка на запрещенные слова
        if contains_stop_words(text):
            return None
            
        return text

def load_distilgpt2_custom() -> Any:
    bos_token = '<|start|>'
    eos_token = '<|end|>'
    pad_token = '<|pad|>'
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2',
                                              bos_token=bos_token,
                                              eos_token=eos_token,
                                              pad_token=pad_token)
    model = GPT2LMHeadModel.from_pretrained('distilgpt2',)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    path_to_checkpoint = f'{PARENT_DIR}/models/distilgpt2_mln1train_ep3.ph'
    if not os.path.exists(path_to_checkpoint):
        gdown.download(
            'https://drive.google.com/uc?id=1WuEZRzzCv0AUMGY9x_keBjrR8SKRkkbq',
            path_to_checkpoint, quiet=False)
    model.load_state_dict(torch.load(path_to_checkpoint, map_location=device))
    model.to(device)
    return model, tokenizer

def generate_distilgpt2_custom(model: Any, tokenizer: Any,
                                input_text: str, num_return_sequences: int = 1) -> [str]:
    bos_token = '<|start|>'
    q_token = ' [QUESTION] '
    ans_token = ' [ANSWER] '
    input_text = bos_token + q_token + input_text + ans_token
    encodings_dict = tokenizer(input_text, truncation=True, max_length=1024)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    input_ids = torch.tensor(encodings_dict['input_ids']).to(device)
    greedy_output = model.generate(input_ids.unsqueeze(0),                             
                                   max_length=len(input_ids) + 140,
                                   num_beams=10,
                                   no_repeat_ngram_size=2, 
                                   num_return_sequences=num_return_sequences,
                                   early_stopping=True)
    # sample_output = model.generate(input_ids.unsqueeze(0),                             
    #                                max_length=len(input_ids) + 140,
    #                                do_sample=True, 
    #                                top_p=0.5, 
    #                                num_beams=5,
    #                                no_repeat_ngram_size=4, 
    #                                num_return_sequences=num_return_sequences,
    #                                temperature=0.8,
    #                                early_stopping=True)
    answer = []
    for output in greedy_output:
        answer.append(
            tokenizer.decode(output[len(input_ids):],
            skip_special_tokens=True))
    return answer


def load_distilgpt2() -> Any:
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2', pad_token_id=tokenizer.eos_token_id)
    return model, tokenizer


def generate_distilgpt2(model: Any, tokenizer: Any,
                        input_text: str, num_return_sequences: int = 1) -> [str]:
    input_text = 'Q: ' + input_text + 'A:'
    input_ids = tokenizer.encode(input_text)
    sample_outputs = model.generate(
        torch.tensor([input_ids]),
        do_sample=True, 
        max_length=len(input_ids) + 140, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=num_return_sequences
    )
    answer = []
    for output in sample_outputs:
        answer.append(
            tokenizer.decode(output[len(input_ids):],
            skip_special_tokens=True))
    return answer

def load_gpt2() -> Any:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    return model, tokenizer


def load_gpt2_custom() -> Any:
    bos_token = '<|start|>'
    eos_token = '<|end|>'
    pad_token = '<|pad|>'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                              bos_token=bos_token,
                                              eos_token=eos_token,
                                              pad_token=pad_token)
    model = GPT2LMHeadModel.from_pretrained('gpt2',)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    path_to_checkpoint = f'{PARENT_DIR}/models/gpt2_ep3.ph'
    if not os.path.exists(path_to_checkpoint):
        gdown.download(
            'https://drive.google.com/uc?id=bP7bAVd__zoD-8IBeE4SoGoQJOstVmNi',
            path_to_checkpoint, quiet=False)
    model.load_state_dict(torch.load(path_to_checkpoint, map_location=device))
    model.to(device)
    return model, tokenizer

def generate_gpt2_custom(model: Any, tokenizer: Any,
                                input_text: str, num_return_sequences: int = 1) -> [str]:
    bos_token = '<|start|>'
    q_token = ' Q: '
    ans_token = ' A: '
    input_text = bos_token + q_token + input_text + ans_token
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
        answer.append(
            tokenizer.decode(output[len(input_ids):],
            skip_special_tokens=True))
    return answer

def load_gpt2_medium_custom() -> Any:
    bos_token = '<|start|>'
    eos_token = '<|end|>'
    pad_token = '<|pad|>'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium',
                                              bos_token=bos_token,
                                              eos_token=eos_token,
                                              pad_token=pad_token)
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium',)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    path_to_checkpoint = f'{PARENT_DIR}/models/gpt2-medium_ep3.ph'
    if not os.path.exists(path_to_checkpoint):
        gdown.download(
            'https://drive.google.com/uc?id=19Q4x6xT1eufy-hJq3RK0Tp6WeCs2rw_j',
            path_to_checkpoint, quiet=False)
    model.load_state_dict(torch.load(path_to_checkpoint, map_location=device))
    model.to(device)
    return model, tokenizer

def generate_gpt2_medium_custom(model: Any, tokenizer: Any,
                                input_text: str, num_return_sequences: int = 1) -> [str]:
    bos_token = '<|start|>'
    q_token = ' Q: '
    ans_token = ' A: '
    input_text = bos_token + q_token + input_text + ans_token
    encodings_dict = tokenizer(input_text, truncation=True, max_length=1024)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    input_ids = torch.tensor(encodings_dict['input_ids']).to(device)
    greedy_output = model.generate(input_ids.unsqueeze(0),                             
                                #    max_length=len(input_ids) + 140,
                                   max_length=len(input_ids) + 80,
                                   num_beams=10,
                                   no_repeat_ngram_size=2, 
                                   num_return_sequences=num_return_sequences,
                                   early_stopping=True)
    answer = []
    for output in greedy_output:
        answer.append(
            tokenizer.decode(output[len(input_ids):],
            skip_special_tokens=True))
    return answer


def load_gpt2_large() -> Any:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    return model, tokenizer

def generate_gpt2_large(model: Any, tokenizer: Any,
                                input_text: str, num_return_sequences: int = 1) -> [str]:
    q_token = 'Q: '
    ans_token = ' A: '
    input_text = q_token + input_text + ans_token
    encodings_dict = tokenizer(input_text, truncation=True, max_length=1024)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    input_ids = torch.tensor(encodings_dict['input_ids']).to(device)
    sample_output = model.generate(input_ids.unsqueeze(0),                             
                                   max_length=len(input_ids) + 80,
                                   do_sample=True, 
                                   num_beams=max(num_return_sequences, 5),
                                   top_p=0.75, 
                                #    temperature=1.1,
                                   no_repeat_ngram_size=4, 
                                   num_return_sequences=num_return_sequences,
                                   early_stopping=True)
    answer = []
    for output in sample_output:
        answer.append(
            tokenizer.decode(output[len(input_ids):],
            skip_special_tokens=True))
    return answer
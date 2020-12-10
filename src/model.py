import logging

from utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Model():
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model, self.tokenizer = self._load()

    def _load(self) -> Any:
        if self.model_name == 'distilgpt2_default':
            return load_distilgpt2()
        if self.model_name == 'gpt2_default':
            return load_gpt2()
        if self.model_name == 'distil_gpt2_custom':
            return load_distilgpt2_custom()
        if self.model_name == 'gpt2_custom':
            return load_gpt2_custom()
        if self.model_name == 'gpt2_medium_custom':
            return load_gpt2_medium_custom()
        if self.model_name == 'gpt2_large':
            return load_gpt2_large()

    

    def generate_text(self, title: str = '', text: str = '', num_return_sequences: int = 1) -> [str]:
        input_text = title + ' ' + text
        output = ''
        if self.model_name == 'distilgpt2_default':
            output = generate_distilgpt2(self.model, self.tokenizer, input_text, num_return_sequences)
        if self.model_name == 'gpt2_default':
            output = generate_distilgpt2(self.model, self.tokenizer, input_text, num_return_sequences)
        if self.model_name == 'distil_gpt2_custom':
            output = generate_distilgpt2_custom(self.model, self.tokenizer, input_text, num_return_sequences)
        if self.model_name == 'gpt2_custom':
            output = generate_gpt2_custom(self.model, self.tokenizer, input_text, num_return_sequences)
        if self.model_name == 'gpt2_medium_custom':
            output = generate_gpt2_medium_custom(self.model, self.tokenizer, input_text, num_return_sequences)
        if self.model_name == 'gpt2_large':
            output = generate_gpt2_large(self.model, self.tokenizer, input_text, num_return_sequences)
        return output

"""
Model metadata shared by binaries.
"""

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig
from transformers import RobertaConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import RobertaForMaskedLM, RobertaTokenizer

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),

    # masked LM models
    'roberta': (RobertaForMaskedLM, RobertaTokenizer),
}


"""
Model metadata shared by binaries.
"""

from pytorch_transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
}


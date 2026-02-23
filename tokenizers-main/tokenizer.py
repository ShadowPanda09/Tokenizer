from tokenizers import Tokenizer, models
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from tokenizers.trainers import BpeTrainer
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer

comp = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(comp)


pipe = pipeline(
    "text-generation",
    model = "FacebookAI/roberta-large-mnli",
    device = comp
)

text = input("")
outputs = pipe(text, max_new_tokens = 256)
response = outputs[0]["generated_text"]
print(response)

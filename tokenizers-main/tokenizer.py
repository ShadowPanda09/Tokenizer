from tokenizers import Tokenizer, models
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from tokenizers.trainers import BpeTrainer
import pandas as pd
import torch as tr
from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b") 
text = input("Prompt here")
tokenizer(text, return_tensors = "pt")





"""
df = pd.read_csv("cleanedjeopardy.csv")
text_data = df['Answer'].tolist()

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizers = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])



tokenizer.train(text_data,trainer)



output= tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
"""
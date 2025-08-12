from transformers import BertTokenizer, BertModel # type: ignore
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentence = "I Love cricket"
tokens = tokenizer(sentence, return_tensors='pt')

print(tokens)

print("Token IDs:", tokens['input_ids'][0].tolist())
print("Tokens:", tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]))
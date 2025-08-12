import tiktoken # type: ignore

tokenizer = tiktoken.encoding_for_model("gpt-4")

text = "I Love Cricket"

token_ids = tokenizer.encode(text)

decoded_text = tokenizer.decode(token_ids)

print("Original text:", text)
print("Token IDs:", token_ids)
print("Number of tokens:", len(token_ids))
print("Decoded text:", decoded_text)


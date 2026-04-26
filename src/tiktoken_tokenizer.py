import tiktoken

tokenizer=tiktoken.get_encoding("gpt2")

with open(r"project1\src\data\output.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
ids=tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
print(tokenizer.decode(ids))
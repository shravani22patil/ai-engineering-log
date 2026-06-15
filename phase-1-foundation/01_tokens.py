import tiktoken

# Loading the tokenizer that GPT-4 and Claude-family models use
# cl100k_base = a vocabulary of ~100,000 subword units
encoder = tiktoken.get_encoding("cl100k_base")

def show_tokens(text: str):
    # encoder() converts your strings into a list of integers token IDs
    token_ids = encoder.encode(text)

    # decoding each token IDs back to the text chunks they represent
    # this shows exactly how your model will see your input text
    token_chunks = [encoder.decode([tid]) for tid in token_ids]

    print(f"\nText: {text}")
    print(f"Tokens: {token_ids}")
    print(f"Count: {len(token_ids)} tokens")
    print(f"Cost at $0.003/1k tokens: ${len(token_ids)/ 1000 * 0.003:.5f}")
    print("_" * 50)

# Trying different kinds of text inputs to see how they are tokenized
show_tokens("Hello, world!")
show_tokens("The quick brown fox jumps over the lazy dog.")
show_tokens("I can't believe it's not butter!")
show_tokens("नमस्ते दुनिया")
show_tokens("你好世界")

# Now trying a real prompt I might send to an LLM
real_prompt = """You are a helpful AI assistant. 
Extract the company name, role, and salary from this job posting:
Senior ML Engineer at Anthropic, $180k-$250k base."""

show_tokens(real_prompt)

# Claude's 200k context = roughly 150,000 English words
# But a Hindi document of 150,000 words would use far more tokens
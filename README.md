# HuggingFaceTokenizers.jl

[![Build Status](https://github.com/MurrellGroup/HuggingFaceTokenizers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/HuggingFaceTokenizers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/HuggingFaceTokenizers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/HuggingFaceTokenizers.jl)

Rudimentary Julia bindings for [🤗 Tokenizers](https://github.com/huggingface/tokenizers), providing fast and easy-to-use tokenization through Python interop.

## Installation

From the Julia REPL, enter Pkg mode with `]` and add the package using the URL:

```
add HuggingFaceTokenizers
```

## Usage

### Loading a Tokenizer

You can load a tokenizer either from a pre-trained model or from a saved file:

```julia
using HuggingFaceTokenizers

# Load a pre-trained tokenizer
tokenizer = from_pretrained(Tokenizer, "bert-base-uncased")

# Alternatively specify revision and auth token
tokenizer = from_pretrained(Tokenizer, "bert-base-uncased", "main", nothing)

# Or load from a file
tokenizer = from_file(Tokenizer, "path/to/tokenizer.json")
```

### Basic Operations

#### Single Text Processing

```julia
# Encode a single text
text = "Hello, how are you?"
result = encode(tokenizer, text)
println("Tokens: ", result.tokens)
println("IDs: ", result.ids)

# Decode back to text
decoded_text = decode(tokenizer, result.ids)
println("Decoded: ", decoded_text)
```

#### Batch Processing

```julia
# Encode multiple texts at once
texts = ["Hello, how are you?", "I'm doing great!"]
batch_results = encode_batch(tokenizer, texts)

# Each result contains tokens and ids
for (i, result) in enumerate(batch_results)
    println("Text $i:")
    println("  Tokens: ", result.tokens)
    println("  IDs: ", result.ids)
end

# Decode multiple sequences at once
ids_batch = [result.ids for result in batch_results]
decoded_texts = decode_batch(tokenizer, ids_batch)
```

### Saving a Tokenizer

```julia
# Save the tokenizer to a file
save(tokenizer, "path/to/save/tokenizer.json")
```

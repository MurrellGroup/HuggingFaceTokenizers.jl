"""
    Tokenizer

A wrapper around a Python tokenizer.
"""
struct Tokenizer
    py_tokenizer::Py
end

"""
    from_file(::Type{Tokenizer}, path::String)

Create a tokenizer from a saved tokenizer file.

```julia
tokenizer = from_file(Tokenizer, "path/to/tokenizer.json")
```
"""
function from_file(::Type{Tokenizer}, path::String)
    py_tokenizer = tokenizers.Tokenizer.from_file(path)
    return Tokenizer(py_tokenizer)
end

"""
    from_pretrained(::Type{Tokenizer}, name::String)

Create a tokenizer from a pretrained tokenizer.

```julia
tokenizer = from_pretrained(Tokenizer, "bert-base-uncased")
```
"""
function from_pretrained(::Type{Tokenizer}, name::String)
    py_tokenizer = tokenizers.Tokenizer.from_pretrained(name)
    return Tokenizer(py_tokenizer)
end

"""
    save(tokenizer::Tokenizer, path::String)

Save the tokenizer to a file.
"""
function save(tokenizer::Tokenizer, path::String)
    tokenizer.py_tokenizer.save(path)
    return nothing
end

"""
    encode(tokenizer::Tokenizer, text::String) -> (tokens::Vector{String}, ids::Vector{Int})

Encode a single text string into tokens and their corresponding IDs.
"""
function encode(tokenizer::Tokenizer, text::String)
    output = tokenizer.py_tokenizer.encode(text)
    tokens = pyconvert(Vector{String}, output.tokens)
    ids = pyconvert(Vector{Int}, output.ids)
    return (; tokens, ids)
end

"""
    decode(tokenizer::Tokenizer, ids::Vector{Int}) -> String

Decode a sequence of token IDs back into text.
"""
function decode(tokenizer::Tokenizer, ids::Vector{Int})
    return pyconvert(String, tokenizer.py_tokenizer.decode(ids))
end

"""
    encode_batch(tokenizer::Tokenizer, texts::Vector{String}) -> Vector{Tuple{Vector{String}, Vector{Int}}}

Encode multiple texts in batch.
"""
function encode_batch(tokenizer::Tokenizer, texts::Vector{String})
    return map(tokenizer.py_tokenizer.encode_batch(texts)) do output
        (; tokens = pyconvert(Vector{String}, output.tokens), ids = pyconvert(Vector{Int}, output.ids))
    end
end

"""
    decode_batch(tokenizer::Tokenizer, batch_ids::Vector{Vector{Int}}) -> Vector{String}

Decode multiple sequences of token IDs in batch.
"""
function decode_batch(tokenizer::Tokenizer, batch_ids::Vector{Vector{Int}})
    pyconvert(Vector{String}, tokenizer.py_tokenizer.decode_batch(batch_ids))
end

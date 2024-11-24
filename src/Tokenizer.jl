"""
    Tokenizer

A wrapper around a Python tokenizer.
"""
struct Tokenizer
    py_tokenizer::Py
end

"""
    from_file(::Type{Tokenizer}, path) -> Tokenizer

Create a tokenizer from a saved tokenizer file.

```julia
tokenizer = from_file(Tokenizer, "path/to/tokenizer.json")
```
"""
from_file(::Type{Tokenizer}, args...; kwargs...) = Tokenizer(tokenizers.Tokenizer.from_file(args...; kwargs...))

"""
    from_pretrained(::Type{Tokenizer}, name::String, revision="main", token=nothing) -> Tokenizer

Create a tokenizer from a pretrained tokenizer.

```julia
tokenizer = from_pretrained(Tokenizer, "bert-base-uncased")
```
"""
from_pretrained(::Type{Tokenizer}, args...; kwargs...) = Tokenizer(tokenizers.Tokenizer.from_pretrained(args...; kwargs...))

"""
    save(tokenizer::Tokenizer, path::String, pretty=true) -> Nothing

Save the tokenizer to a file.
"""
function save(tokenizer::Tokenizer, args...; kwargs...)
    pycall(tokenizer.py_tokenizer.save, args...; kwargs...)
    return nothing
end

"""
    encode(tokenizer::Tokenizer, text::String) -> (; tokens::Vector{String}, ids::Vector{Int}) -> @NamedTuple{tokens, ids}

Encode a single text string into tokens and their corresponding IDs.
"""
function encode(tokenizer::Tokenizer, args...; kwargs...)
    output = tokenizer.py_tokenizer.encode(args...; kwargs...)
    tokens = pyconvert(Vector{String}, output.tokens)
    ids = pyconvert(Vector{Int}, output.ids)
    return (; tokens, ids)
end

"""
    decode(tokenizer::Tokenizer, ids::Vector{Int}) -> String

Decode a sequence of token IDs back into text.
"""
function decode(tokenizer::Tokenizer, args...; kwargs...)
    return pyconvert(String, tokenizer.py_tokenizer.decode(args...; kwargs...))
end

"""
    encode_batch(tokenizer::Tokenizer, text_batch::Vector{String}) -> Vector{@NamedTuple{tokens, ids}}

Encode multiple texts in batch.
"""
function encode_batch(tokenizer::Tokenizer, args...; kwargs...)
    outputs = tokenizer.py_tokenizer.encode_batch(args...; kwargs...)
    return map(outputs) do output
        (; tokens = pyconvert(Vector{String}, output.tokens), ids = pyconvert(Vector{Int}, output.ids))
    end
end

"""
    decode_batch(tokenizer::Tokenizer, ids_batch::Vector{Vector{Int}}) -> Vector{String}

Decode multiple sequences of token IDs in batch.
"""
function decode_batch(tokenizer::Tokenizer, args...; kwargs...)
    return pyconvert(Vector{String}, tokenizer.py_tokenizer.decode_batch(args...; kwargs...))
end

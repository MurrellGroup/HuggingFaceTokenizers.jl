"""
    Tokenizer

A wrapper around a Python tokenizer.

It can be passed directly to Python functions called through PythonCall,
and `Py(tokenizer)` returns the underlying Python object.
"""
struct Tokenizer
    py_tokenizer::Py
end

PythonCall.ispy(::Tokenizer) = true
PythonCall.Py(tokenizer::Tokenizer) = tokenizer.py_tokenizer

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
    encode_batch_ids(tokenizer::Tokenizer, text_batch; T=UInt32, kwargs...) -> Vector{<:SubArray{T}}

Encode multiple texts in batch, returning only the token IDs of each text.

The IDs of all texts are stored contiguously in a single `Vector{T}`, available as
`parent(ids_batch[1])`, and the result holds a view into it for each text.
This is much faster and uses much less memory than `encode_batch`,
which makes it suited to tokenizing large corpora.
`T=UInt16` is enough for vocabularies of up to 65536 tokens, such as GPT-2's,
and an error is thrown if an ID does not fit in `T`.

Keyword arguments other than `T` are passed to the Python tokenizer's `encode_batch_fast`,
e.g. `add_special_tokens=false`.

```julia
ids_batch = encode_batch_ids(tokenizer, ["Hello, how are you?", "I'm doing great!"]; T=UInt16)
```
"""
function encode_batch_ids(tokenizer::Tokenizer, args...;
        T::Type{<:Union{Int8,Int16,Int32,Int64,UInt8,UInt16,UInt32,UInt64}}=UInt32, kwargs...)
    encodings = tokenizer.py_tokenizer.encode_batch_fast(args...; kwargs...)
    lengths = Vector{Int}(undef, pylen(encodings))
    write_lengths(encodings, lengths)
    ids = Vector{T}(undef, sum(lengths))
    write_ids(encodings, ids)
    PythonCall.pydel!(encodings)
    stops = cumsum(lengths)
    return [view(ids, stop-len+1:stop) for (len, stop) in zip(lengths, stops)]
end

"""
    decode_batch(tokenizer::Tokenizer, ids_batch::Vector{Vector{Int}}) -> Vector{String}

Decode multiple sequences of token IDs in batch.
"""
function decode_batch(tokenizer::Tokenizer, args...; kwargs...)
    return pyconvert(Vector{String}, tokenizer.py_tokenizer.decode_batch(args...; kwargs...))
end

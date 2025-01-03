"""
    HuggingFaceTokenizers

A Julia wrapper around HuggingFace's Tokenizers Python library.

See https://huggingface.co/docs/tokenizers/en/index for official documentation.
"""
module HuggingFaceTokenizers

using PythonCall

const tokenizers = PythonCall.pynew()

__init__() = PythonCall.pycopy!(tokenizers, pyimport("tokenizers"))

include("Tokenizer.jl")
export Tokenizer
export from_file
export from_pretrained
export save
export encode
export decode
export encode_batch
export decode_batch

end

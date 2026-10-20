"""
    HuggingFaceTokenizers

A Julia wrapper around HuggingFace's Tokenizers Python library.

See https://huggingface.co/docs/tokenizers/en/index for official documentation.
"""
module HuggingFaceTokenizers

using PythonCall

const tokenizers = PythonCall.pynew()
const write_lengths = PythonCall.pynew()
const write_ids = PythonCall.pynew()

# Writing into Julia arrays through the buffer protocol avoids creating
# a Julia wrapper for each `Encoding` and its `ids` list.
const HELPERS = """
from array import array

def write_lengths(encodings, out):
    out = memoryview(out)
    for i, e in enumerate(encodings):
        out[i] = len(e)

def write_ids(encodings, out):
    out = memoryview(out)
    i = 0
    for e in encodings:
        ids = array(out.format, e.ids)
        out[i:i + len(ids)] = ids
        i += len(ids)
"""

function __init__()
    PythonCall.pycopy!(tokenizers, pyimport("tokenizers"))
    helpers = pyexec(@NamedTuple{write_lengths::Py, write_ids::Py}, HELPERS, @__MODULE__)
    PythonCall.pycopy!(write_lengths, helpers.write_lengths)
    PythonCall.pycopy!(write_ids, helpers.write_ids)
end

include("Tokenizer.jl")
export Tokenizer
export from_file
export from_pretrained
export save
export encode
export decode
export encode_batch
export encode_batch_ids
export decode_batch

end

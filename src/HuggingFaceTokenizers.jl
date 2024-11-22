module HuggingFaceTokenizers

using PythonCall

const tokenizers = Ref{Py}()

function __init__()
    tokenizers[] = pyimport("tokenizers")
end

end

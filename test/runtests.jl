using HuggingFaceTokenizers
using Test

@testset "HuggingFaceTokenizers.jl" begin

    @testset "Basic Operations" begin
        # Load pretrained tokenizer
        tokenizer = from_pretrained(Tokenizer, "bert-base-uncased")
        
        # Test single text encoding/decoding
        text = "Hello, how are you?"
        result = encode(tokenizer, text)
        
        @test result.tokens isa Vector{String}
        @test result.ids isa Vector{Int}
        @test !isempty(result.tokens)
        @test !isempty(result.ids)
        @test length(result.tokens) == length(result.ids)
        
        decoded_text = decode(tokenizer, result.ids)
        @test decoded_text isa String
        @test !isempty(decoded_text)
        # Note: The decoded text might not match exactly due to tokenizer behavior
        @test lowercase(decoded_text) == lowercase(text)
        
        # Test batch processing
        texts = ["Hello, how are you?", "I'm doing great!"]
        batch_results = encode_batch(tokenizer, texts)
        
        @test batch_results isa Vector
        @test length(batch_results) == length(texts)
        
        for result in batch_results
            @test result.tokens isa Vector{String}
            @test result.ids isa Vector{Int}
            @test !isempty(result.tokens)
            @test !isempty(result.ids)
            @test length(result.tokens) == length(result.ids)
        end
        
        # Test batch decoding
        ids_batch = [result.ids for result in batch_results]
        decoded_texts = decode_batch(tokenizer, ids_batch)
        
        @test decoded_texts isa Vector{String}
        @test length(decoded_texts) == length(texts)
        @test all(!isempty, decoded_texts)
    end

    @testset "encode_batch_ids" begin
        tokenizer = from_pretrained(Tokenizer, "bert-base-uncased")
        texts = ["Hello, how are you?", "", "I'm doing great!"]

        ids_batch = encode_batch_ids(tokenizer, texts)
        @test ids_batch isa Vector{<:SubArray{UInt32}}
        @test ids_batch == [result.ids for result in encode_batch(tokenizer, texts)]
        @test parent(ids_batch[1]) == reduce(vcat, ids_batch)
        @test all(ids -> parent(ids) === parent(ids_batch[1]), ids_batch)

        @test encode_batch_ids(tokenizer, texts; T=UInt16) isa Vector{<:SubArray{UInt16}}
        @test encode_batch_ids(tokenizer, texts; T=Int) == ids_batch
        @test encode_batch_ids(tokenizer, texts; add_special_tokens=false) ==
            [result.ids for result in encode_batch(tokenizer, texts; add_special_tokens=false)]
        @test isempty(encode_batch_ids(tokenizer, String[]))
        @test_throws HuggingFaceTokenizers.PythonCall.PyException encode_batch_ids(tokenizer, texts; T=Int8)

        @test decode_batch(tokenizer, ids_batch) == decode_batch(tokenizer, [result.ids for result in encode_batch(tokenizer, texts)])
    end

    @testset "Python interop" begin
        PythonCall = HuggingFaceTokenizers.PythonCall
        tokenizer = from_pretrained(Tokenizer, "bert-base-uncased")
        @test PythonCall.ispy(tokenizer)
        @test PythonCall.Py(tokenizer) === tokenizer.py_tokenizer
        f = PythonCall.pyeval("lambda tokenizer, text: tokenizer.encode(text).ids", Main)
        @test PythonCall.pyconvert(Vector{Int}, f(tokenizer, "Hello")) == encode(tokenizer, "Hello").ids
    end

    @testset "File Operations" begin
        tokenizer = from_pretrained(Tokenizer, "bert-base-uncased")
        
        mktempdir() do temp_dir
            # Test save and load
            temp_path = joinpath(temp_dir, "tokenizer.json")
            
            # Test saving
            save(tokenizer, temp_path)
            @test isfile(temp_path)
            
            # Test loading from file
            loaded_tokenizer = from_file(Tokenizer, temp_path)
            @test loaded_tokenizer isa Tokenizer
            
            # Verify the loaded tokenizer works
            text = "Hello, world!"
            result1 = encode(tokenizer, text)
            result2 = encode(loaded_tokenizer, text)
            
            @test result1.tokens == result2.tokens
            @test result1.ids == result2.ids
        end
        # No need for explicit cleanup - mktempdir handles it
    end

end

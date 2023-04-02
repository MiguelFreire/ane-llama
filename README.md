# ane-llama
Llama port for Apple Neural Engine based on https://github.com/apple/ml-ane-transformers/tree/main


## Requirements
- macOS >= 13.x
- xCode with MLCore

## Roadmap
- [ ] Convert base Layers (SelfAttention, CrossAttention, LayerNorm and PW-FFN etc)
- [ ] Convert Decoding
- [ ] Convert full HF transformer model to be ANE-compatible (eg: https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/huggingface/distilbert.py)
- [ ] Optimize for TorchScript using JIT 
- [ ] Convert from TorchScript to CoreML using Python CoreML support tools.

## Summary Principles
# Principle 1: Picking the Right Data Format
To migrate to the desirable (B, C, 1, S) data format, we swap all nn.Linear layers with nn.Conv2d layers. 

#Principle 2: Chunking Large Intermediate Tensors
For the multihead attention function in the Transformer, we split the query, key, and value tensors to create an explicit list of single-head attention functions, each of which operates on smaller chunks of input data.

#Principle 3: Minimizing Memory Copies
We use the bchq,bkhc->bkhq einsum formula, which represents a batched matmul operation whose data format directly maps to hardware without intermediate transpose and reshape operations. 


## References 
- https://github.com/apple/ml-ane-transformers/
- https://machinelearning.apple.com/research/neural-engine-transformers

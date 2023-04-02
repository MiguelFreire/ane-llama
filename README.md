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


## References 
- https://github.com/apple/ml-ane-transformers/
- https://machinelearning.apple.com/research/neural-engine-transformers

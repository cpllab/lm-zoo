This image is adapted from github.com/cpllab/lm-zoo/tree/master/models/transformers-base.

DialoGPT-medium extends gpt-2-medium by fine-tuning on Reddit data in order to model dialogue.
For this, the eos token is used to mark a speaker change (represented by the `[SEP]` token in the input, which requires some modifications to `get_surprisals.py` and `tokenizer.py`).
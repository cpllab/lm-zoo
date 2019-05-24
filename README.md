# Pre-trained language model zoo

This folder contains scripts and source code for obtaining surprisals 
from the following pre-trained language models:

1. [GRNN](https://github.com/facebookresearch/colorlessgreenRNNs) (recurrent neural network trained on Wikipedia)
2. [JRNN](https://github.com/tensorflow/models/tree/master/research/lm_1b) (recurrent neural network trained on Billion Word Benchmark)
3. [RNNG](https://github.com/clab/rnng) (recurrent neural network grammar)
4. [Transformer-XL](https://github.com/kimiyoung/transformer-xl)
5. [Tiny LSTM](https://github.com/pytorch/examples/tree/master/word_language_model)
6. [KN-smoothed 5gram]

## Scripts

The scripts can be found in the `scripts` folder. Each script expects two arguments:
$1 is the input file containing the sentences, and $2 is the output file to save the surprisals.

The output file will have the following format:
```
token1 0.0
token2 ...
.      ...
<eos>  0.0
```
where the second column (separated by whitespace) gives the surprisal in bits of the token.

The input file should have each sentence on a new line. For every model except RNNG and Tiny LSTM,
the sentence should end with an `<eos>` token. 

Note that the sentences are also expected to be **tokenized**.
In the case of RNNG and Tiny LSTM, it also needs to be `UNK`-ified. 
An `UNK`ify function is provided in `rnng-incremental/get_raw.py`, which
can be used in the following way:

```bash
python2 get_raw.py train.02-21 \
    RAW.txt > UNKIFIED.txt
```

## Dependencies

The GRNN, JRNN, Transformer-XL, and Tiny LSTM models require `pytorch` and other dependencies that can be found
in their source folders. If you don't feel like creating your own environments, feel free to "steal" mine:
`/om2/user/jennhu/conda/envs/neural-nlp` (credit to Martin Schrimpf) works for GRNN, JRNN, and Tiny LSTM, and
`/om2/user/jennhu/conda/envs/transXL` was custom-built for Transformer-XL.

The dependencies for RNNG should already be set in the source code. If problems arise, I may make
a Singularity image available with the relevant C++ libraries.

## Other tips

When submitting jobs to SLURM, keep in mind that different models have different memory/time
requirements. The following settings have worked for me in the past:

| Model | Suggested memory | Speed  | GPU |
| :---: | :--------------: | :----: | :-: |
| GRNN  | `5G`             | Medium | Yes |
| JRNN  | `20G`            | Medium | No  |
| RNNG  | `12G`            | Slow   | No  |
| Tiny  | `5G`             | Fast   | No  |
| Trans | `5G`             | Fast   | Yes |
| ngram | `5G`             | Fast   | No  |

The speed is relative to the other models; for reference, Tiny LSTM takes under 1 minute to calculate
surprisal for 900 simple sentences (~7 words each), while RNNG takes several hours. 

If using GPU, remember to request the appropriate resources in your `sbatch` call.

## Todo

- [ ] add GPU functionality
- [ ] add BERT (currently have working pipeline, but pre-processing is a little more involved)
- [ ] compile stack-only ablated RNNG (Kuncoro et al. 2017) - raised [issue](https://github.com/clab/rnng/issues/17)
- [ ] add environments to shared folder

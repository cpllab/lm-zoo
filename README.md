# Language Model Zoo

![zoo-logo](https://cpllab.github.io/lm-zoo/_images/logo.png)

[![CircleCI](https://circleci.com/gh/cpllab/lm-zoo/tree/master.svg?style=svg&circle-token=d907824249db5ad63c03bfcc3b403c6d9ad845e2)](https://circleci.com/gh/cpllab/lm-zoo/tree/master)
[![Gitter chat](https://badges.gitter.im/lm-zoo/community.png)](https://gitter.im/lm-zoo/community.png)

The Language Model Zoo is an open-source repository of state-of-the-art
language models, designed to support black-box access to model predictions and
representations. It provides the command line tool `lm-zoo`, a standard
interface for interacting with language models.

You can use `lm-zoo` to

1. extract token-level surprisal data, and
2. preprocess corpora according to a language model's particular tokenization
   standards.

Quick links:

- [Quickstart](https://cpllab.github.io/lm-zoo/quickstart.html)
- [Supported models](https://cpllab.github.io/lm-zoo/models.html)
- [Contributing models](https://cpllab.github.io/lm-zoo/contributing.html)

## Getting started

Running language models from this repository requires [Docker][1].

You can install the `lm-zoo` via `pip`:

    $ pip install lm-zoo

List available language models:

    $ lm-zoo list
    gpt2
            Image URI:  docker.io/cpllab/language-models:gpt2
            Full name: None
            Reference URL: https://openai.com/blog/better-language-models/
            Maintainer: None
            Last updated: None
    RNNG
            Image URI:  docker.io/cpllab/language-models:rnng
            Full name: None
            Reference URL: TODO
            Maintainer: None
            Last updated: None
    ordered-neurons
            Image URI:  docker.io/cpllab/language-models:ordered-neurons
            Full name: None
            Reference URL: https://github.com/yikangshen/Ordered-Neurons
            Maintainer: None
            Last updated: None
    ...

Tokenize some text according to a language model's standard:

    $ wget https://cpllab.github.io/lm-zoo/metamorphosis.txt -O metamorphosis.txt
    $ lm-zoo tokenize gpt2 metamorphosis.txt
    Pulling latest Docker image for cpllab/language-models:gpt2.
    One Ġmorning , Ġwhen ĠGreg or ĠSam sa Ġwoke Ġfrom Ġtroubled Ġdreams , Ġhe Ġfound Ġhimself Ġtransformed Ġin Ġhis Ġbed Ġinto Ġa Ġhorrible Ġver min .
    He Ġlay Ġon Ġhis Ġarmour - like Ġback , Ġand Ġif Ġhe Ġlifted Ġhis Ġhead Ġa Ġlittle Ġhe Ġcould Ġsee Ġhis Ġbrown Ġbelly , Ġslightly Ġdom ed Ġand Ġdivided Ġby Ġar ches Ġinto Ġstiff Ġsections .
    The Ġbed ding Ġwas Ġhardly Ġable Ġto Ġcover Ġit Ġand Ġseemed Ġready Ġto Ġslide Ġoff Ġany Ġmoment .
    ...

Get token-level surprisals for text data:

    $ lm-zoo get-surprisals ngram metamorphosis.txt
    sentence_id     token_id        token   surprisal
    1       1       one     7.76847
    1       2       morning 9.40638
    1       3       ,       1.05009
    1       4       when    7.08489
    1       5       gregor  18.8963
    1       6       <unk>   4.27466
    1       7       woke    19.0607
    1       8       from    10.3404
    1       9       troubled        17.478
    1       10      dreams  10.671
    1       11      ,       3.39374
    1       12      he      5.99193
    1       13      found   8.07358
    1       14      himself 2.92718
    1       15      transformed     16.7328
    1       16      in      5.32057
    1       17      his     7.26454
    1       18      bed     9.78166
    1       19      into    8.90954
    1       20      a       3.72355
    1       21      horrible        14.2477
    1       22      <unk>   3.56907
    1       23      .       3.90242
    1       24      </s>    22.8395
    2       1       he      4.43708
    2       2       lay     14.1721
    ...

For more information, see our [Quickstart tutorial][2].

[1]: https://docs.docker.com/get-docker/
[2]: https://cpllab.github.io/lm-zoo/quickstart.html

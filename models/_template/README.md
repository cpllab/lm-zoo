This is a sample language model wrapper. It builds and passes the LM Zoo tests,
though it isn't very interesting. For context, see the [LM Zoo documentation on
contributing language models][1].

To build yourself and see, first run (from the LM-Zoo root directory):

```bash
$ ./scripts/build_and_test.sh _template mylm
Sending build context to Docker daemon  18.47MB
Step 1/18 : FROM alpine AS builder
...
Successfully built 07e5e7dc062e
Successfully tagged mylm
```

Now you can do things like fetch surprisals. Note that the language model only
has two words in its "vocabulary!"

```bash
$ echo "my vocabulary is poor" | docker run --rm -i mylm get_surprisals /dev/stdin
sentence_id     token_id        token   surprisal
1       1       my      1.7850445456419743
1       2       vocabulary      4.897686927776151
1       3       <unk>   0.8604369362313263
1       4       <unk>   4.077675053294858
```

[1]: https://cpllab.github.io/lm-zoo/contributing.html

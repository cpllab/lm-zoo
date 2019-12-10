.. _api:

Language model API
==================

Each language model in the LM Zoo is a Docker container which provides various
binaries for running and probing the internal model. This document specifies
the exact API for accessing and using these language models.

Each container makes available the following binaries, which communicate via
`stdout` and `stderr` with the host:

.. contents::
   :local:

``spec``
^^^^^^^^

:Arguments: None
:stdout: Description of language model and relevant metadata
:stdout format: JSON
:stderr: None

The ``spec`` binary outputs relevant metadata describing the language model image:

.. jsonschema:: schemas/language_model_spec.json


``get_surprisals``
^^^^^^^^^^^^^^^^^^

:Arguments:
   1. Path to a natural-language input text file, not pre-tokenized or unkified; one sentence per line
:stdout: Per-word surprisal values
:stdout format: TSV
:stderr: Not specified


``tokenize``
^^^^^^^^^^^^

:Arguments:
   1. Path to a natural-language input text file, not pre-tokenized or unkified; one sentence per line
:stdout: Tokenized text, one sentence per line
:stdout format: plain text
:stderr: Not specified


The following constraints must hold on the output of ``tokenize``:

1. All tokens (when splitting on whitespace) are elements of the language
   model's vocabulary, as given in ``vocabulary.items`` of the language model
   specification (see `spec`_).
2. If the model produces UNK tokens, all the UNK tokens must be part of the
   declared ``vocabulary.unk_types`` list of the language model specification
   (see `spec`_).
3. The model may prepend/insert/append sentence boundary tokens and other
   special tokens, so long as they are members of the relevant special type
   lists declared in the language model specification (see
   ``vocabulary.prefix_types``, ``vocabulary.suffix_types``, and
   ``vocabulary.special_types`` in `spec`_).


``unkify``
^^^^^^^^^^

:Arguments:
   1. Path to a natural-language input text file, not pre-tokenized or unkified; one sentence per line
:stdout: Sequence of mask values, one sentence per line (``0`` if the
         corresponding token is known by the model, and ``1`` otherwise)
:stdout format: plain text
:stderr: Not specified

The following constraints must hold on the output of ``unkify``:

1. Each line is composed only of ``0`` and ``1`` values separated by spaces.
2. Each line corresponds to a sentence line as output by `tokenize`_, with
   exactly the same number of tokens (when splitting on whitespace).


``get_predictions``
^^^^^^^^^^^^^^^^^^^

:Arguments:
   1. Path to a natural-language input text file, not pre-tokenized or unkified; one sentence per line
:stdout: Description of full next-word predictive distributions for each token in the input
:stdout format: JSON
:stderr: Not specified

TODO JSON spec


.. _model_api:

Language model Docker API
=========================

Each language model in the LM Zoo is a Docker container which provides various
binaries for running and probing the internal model. This document specifies
the exact API for accessing and using these language models.

.. contents::
   :local:

Language model binaries
-----------------------

Each container makes available the following binaries, which communicate via
`stdout` and `stderr` with the host:


``spec``
^^^^^^^^

:Arguments: None
:stdout: Description of language model and relevant metadata
:stdout format: JSON
:stderr: None

The ``spec`` binary outputs relevant metadata describing the language model image:

.. jsonschema:: schemas/language_model_spec.json


``get_predictions.hdf5``
^^^^^^^^^^^^^^^^^^^^^^^^

:Arguments:
  1. Path to a natural-language input text file, not pre-tokenized or unkified; one sentence per line
  2. Path to which HDF5 prediction data should be written
:stdout: Not specified
:stderr: Not specified

Extract word-level predictive distributions :math:`\log p(w_i \mid w_1, \dots,
w_{i-1})` for each word of each sentence. Writes results in HDF5 format as a
collection of matrices, along with prediction vocabulary metadata. The HDF5
file should have the following groups::

   /sentence/<i>/predictions: N_tokens_i * N_vocabulary array of
      log-probabilities (rows are log-probability distributions)
   /sentence/<i>/tokens: sequence of integer token IDs corresponding to
      indices in ``/vocabulary``
   /vocabulary: byte-encoded string array of vocabulary items (decode with
      ``numpy.char.decode(vocabulary, "utf-8")``)

where ``i`` is zero-indexed.



``get_surprisals``
^^^^^^^^^^^^^^^^^^

:Arguments:
   1. Path to a natural-language input text file, not pre-tokenized or unkified; one sentence per line
:stdout: Per-word surprisal values
:stdout format: TSV
:stderr: Not specified

The output of ``get_surprisals`` is a tab-separated file with the following
columns (including headings on the first line):

.. jsonschema:: schemas/get_surprisals_output.json

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


Building models with partial feature support
--------------------------------------------

Not all models in the LM Zoo need support all of the above interfaces. For
example, some models do not support the ``get_predictions.hdf5`` command at the
moment. Models explicitly mark which features they do and do not support. To do
this in your own model, do the following:

1. Set a ``false`` value for the relevant feature under the
   ``supported_features`` key of your model's spec (see `spec`_).
2. Define a dummy binary with the relevant feature's name which simply exits
   with a status code ``99``. This status code will be interpreted by the LM
   Zoo wrapper API and CLI tools to indicate that the requested feature is not
   supported.

   You can pull in the shared script ``shared/unsupported`` to do this for you.
   See an example in `the Dockerfile for the RNNG model
   <https://github.com/cpllab/lm-zoo/blob/5c72f5aa6a9b5e67f990d363c9ea4fc35c37679e/models/RNNG/Dockerfile#L58>`_,
   which does not support ``get_predictions.hdf5``.

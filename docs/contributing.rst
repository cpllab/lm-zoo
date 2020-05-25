.. _contributing:

Contributing to the LM Zoo
==========================

Anyone can contribute models to the LM Zoo. This page provides instructions on
creating a LM Zoo-friendly Docker image, and on submitting your image for Zoo
review.


Language models in the LM Zoo are Docker images which expose a set of binary
programs conforming to the :ref:`model_api`. At a high level, the steps you'll need
to take to prepare such an image are

1. Acquire open source code and model checkpoints for the language model you'd
   like to contribute.
2. Prepare a Docker image containing the code and model checkpoint, and verify
   that you can successfully run the model from within the container.  Write
   binary scripts conforming to the :ref:`model_api` which invoke your model.
3. Submit your language model as a pull request to the ``cpllab/lm-zoo`` Github
   repository.

Each of these steps are described in further detail below.

Acquiring open-source code
--------------------------

Because the LM Zoo system runs on Docker, you are welcome to contribute code
which uses any programming language and framework you favor, so long as you can
provide a supporting Docker image.

You are also welcome to contribute language model images for models which are
not your own, so long as this complies with the license of the model
developers. When in doubt, ask the developers for permission first!

Preparing a Docker image
------------------------

If you're not familiar with building your own Docker images, we recommend that
you first read `the official Docker documentation
<https://docs.docker.com/get-started/part2/>`_.

1. `Create your own fork <https://github.com/cpllab/lm-zoo/fork>`_ of the
   ``lm-zoo`` repository.
2. Create your own subdirectory in the ``models`` directory. We recommend
   copying from the `template directory
   <https://github.com/cpllab/lm-zoo/tree/master/models/_template>`_ for
   starters. Modify the ``Dockerfile`` and add wrapper scripts as necessary
   until your model conforms to the :ref:`model_api`.
3. You can build and test your wrapper as necessary using the file
   ``scripts/build_and_test.sh``::

     ./scripts/build_and_test.sh <my_directory_name> <docker_target>
     ./scripts/build_and_test.sh _template myname/mylm

A few tips on building an image:

Docker build context
""""""""""""""""""""
All LM Zoo Docker images are built with the root directory of the LM Zoo as
build context. This allows each Docker build routine to copy in shared scripts,
for example, from the ``shared`` directory. Structure your build files
accordingly -- see the template directory's ``Dockerfile`` and its use of a
build argument ``MODEL_ROOT`` as an example.

Shared scripts
""""""""""""""
The ``shared`` directory provides several useful shared binary scripts (e.g.
``unkify`` and ``spec``) which you may be able to reuse for your image. We
offer these scripts to avoid forcing contributors to reinvent mundane code for
every image they build.

Submitting your language model
------------------------------

Once you've confirmed that all tests pass for your image, create a pull request
at the ``lm-zoo`` Github repository. Your pull request should follow `this
template
<https://github.com/cpllab/lm-zoo/blob/master/docs/pull_request_template.md>`_
(this should be automatically inserted when you create a pull request).

The LM Zoo maintainers will evaluate your submission and respond on the pull
request thread. Thanks for contributing!

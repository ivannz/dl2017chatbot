# dl2017chatbot
Course project for Deep Learning 2017 course at Skoltech

## Introduction
This repository contains the code and custom modules for the Character-level
neiral chatbot project for the Spring 2017 course on deep learning at Skoltech.

## Requirements
The latest (bleeding edge) versions of theano, libgpuarray and lasagne are required
to replicate the experimnet. In order to run interactive dialogue session with
the trained models, it is necessary to install `python-telegram-bot` module.

## Running experiments
The experiments are contained in iPython notebooks, which were `run-all` tested
beforehand.

## Running standalone scripts
Instead of running and conversin with the trained chatbots in the notebooks,
it is much more convenient to register a telegram bot. In order to do so,
please make sure that you have a registered token for accessing the Telegram
API (see [here](https://core.telegram.org/bots#6-botfather) for a walkthrough
on how to obtain a token). In each standalone script there is a keyword
`<<TOKEN>>` which must be replace with your own.

## Future developments
The main issue with the repo in its current state (as of
[commit:fbdd500](https://github.com/ivannz/dl2017chatbot/tree/fbdd5007a213b79f5db8cec52dceaf340e8a2e5e))
is that the whole codebase need major  refactoring: all standalone scirps and
experiment notebooks use almost identical code snippets, e.g. `generator_step_sm()`
and `GRULayer_freeze()` functions. This was not done before due to time constraints
and the fact that the codebase was still in the prototype phase.

## Future experiments
In the future we would like to fuse the attention and stack RNN units to make
a Attentive Stack / Queue version of the chatbot. Another development vector is
to use Stack Augmented RNN in the DIALOGUE version of the bot. Finally, it would
be insightfull to abandon character level neural model in favor of a word level,
so that both stack and attention mechanisms can better demonstrate their benefits.

## Partial arxiv:1503.01007 replication
We also have an iPython notebook with a partial replication of the original paper
on [Atack Augmented Recurrent Nets](https://arxiv.org/abs/1503.01007). The experiment
in question of learning to generate sequences of the form `>a{n}b{m}c{n+m}<`.

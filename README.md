# Character Based Language Models Through Variational Sentence and Word Embeddings

Contributors:

1. Kevin Dsouza 
2. Zaccary Alperstein 

## VaeLM Model 
<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/VaeLM//blob/master/paper/Project/vaelm.png?raw=true">
</p>

## Overview 
We propose character level language model to overcome the limitations of the word level models. We use two hierarchical frameworks which differ in the way they incorporate the hierarchy, while both can jointly generate word and sentence level embeddings thus allowing us to form latent representations with sentence level context. We use a variational autoencoder to produce these latent representations and employ an attention mechanism over the latent word embeddings to account for long term dependencies in the sentence. This approach is not only novel from a language modelling perspective, but also from the perspective of VAEs as such a deep latent hierarchical structure has never been explored. If successful, this approach may serve to augment neural machine translation exchanging the encoder and decoder for ours, or for writing where one may need a different sentence with the same context.

### Framework 1: Hierarchy in the Sentence Based Latent representation

### Framework 2: Hierarchy in the RNN



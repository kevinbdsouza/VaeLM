# Character Based Language Models Through Variational Sentence and Word Embeddings

Contributors:

1. Kevin Dsouza
2. Zaccary Alperstein

## VaeLM Model
<p align="center">
<img align="center" src="https://github.com/kevinbdsouza/VaeLM//blob/master/paper/Project/vaelm.png?raw=true">
</p>

## Overview
We propose a character level language model to overcome the limitations of word level models. Two hierarchical frameworks are explored which differ in how the hierarchy is incorporated. Both frameworks jointly generate word and sentence level embeddings so that latent representations capture sentence level context. A variational autoencoder produces these latent representations while an attention mechanism over the latent word embeddings accounts for long term dependencies. This approach is novel from a language modelling perspective and from the perspective of variational autoencoders. When successful, this model can be used for tasks such as neural machine translation or for generating alternative sentences with the same context.

### Framework 1: Hierarchy in the Sentence Based Latent representation

### Framework 2: Hierarchy in the RNN

## Getting Started
The project requires Python 3 along with the following packages:

- TensorFlow 1.x
- NumPy
- NLTK
- h5py

Install the dependencies using pip:

```bash
pip install tensorflow numpy nltk h5py
```

## Data
Sample Penn Treebank character data is provided in `data/english`. The `README` file in that directory explains the format. You can replace these files with your own data in the same layout.

## Preprocessing
Before training, convert the text files into HDF5 files using `src/create_h5.py`:

```bash
python src/create_h5.py train.h5 valid.h5
```

This script invokes `src/preprocess.py` to tokenize the dataset and creates two HDF5 files containing the training and validation splits.

## Training
The main training loop is implemented in `src/train_vaeLM.py`. An example invocation is provided in `src/run_local.py`. After creating the HDF5 data files, start a new training run with something similar to:

```bash
python src/run_local.py
```

Modify the paths inside `run_local.py` to point to your own data if necessary.

## License
This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

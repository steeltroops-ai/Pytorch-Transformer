# Pytorch-Transformer

[![arXiv](https://img.shields.io/badge/arXiv-1706.03762-B31B1B.svg)](https://arxiv.org/abs/1706.03762)

This repository contains the implementation of the Transformer model from the paper **"Attention is All You Need"**.

## Overview

The Transformer model is a novel architecture for sequence-to-sequence modeling without recurrence or convolution. It leverages self-attention mechanisms and has become a foundational model in natural language processing tasks.

### Features
- Implementation of the Transformer model using PyTorch
- Support for both training and inference workflows
- Visualization of attention mechanisms

## Repository Structure

```
├── .gitignore              # Specifies intentionally untracked files to ignore
├── Beam_Search.ipynb       # Jupyter notebook for implementing beam search decoding
├── Colab_Train.ipynb       # Jupyter notebook for training the model in Google Colab
├── Inference.ipynb         # Jupyter notebook for performing inference with the trained model
├── Local_Train.ipynb       # Jupyter notebook for local training setup
├── README.md               # This file
├── attention_visual.ipynb  # Jupyter notebook for visualizing attention
├── conda.txt               # Conda environment specification
├── config.py               # Configuration settings for the model
├── dataset.py              # Data loading and preprocessing scripts
├── model.py                # Contains the implementation of the Transformer model architecture
├── requirements.txt        # List of Python dependencies needed to run this project
├── train.py                # Script for training the model
├── train_wb.py             # Training script adapted for Weights & Biases logging
└── translate.py            # Script for translation using the trained model
```

## Setup

### Dependencies

Ensure you have Python 3.6+ installed. Then, install the required packages:

```sh
pip install -r requirements.txt
```

Or if you use Conda:

```sh
conda env create -f conda.txt
conda activate pytorch-transformer
```

## Training

### Google Colab
Use `Colab_Train.ipynb` for training in Colab.

### Local Machine
Use `Local_Train.ipynb` or run:

```sh
python train.py --config config.yaml
```

For Weights & Biases integration:

```sh
python train_wb.py --config config.yaml
```

## Inference

To perform inference, you can use:

- `Inference.ipynb` for interactive notebook environment.
- `translate.py` for command line interface:

```sh
python translate.py --model_path path/to/model --source source.txt --target target.txt
```

## Visualization

To visualize attention mechanisms, refer to `attention_visual.ipynb`.

## Contribution

Contributions are welcome! Please fork this repository and submit pull requests with your changes.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Acknowledgments

- Thanks to the authors of the original paper for their groundbreaking work.
- Special thanks to the PyTorch community for providing an amazing framework for machine learning

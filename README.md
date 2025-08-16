
![MNISTique Logo](https://github.com/MosesTheRedSea/MNISTique/blob/main/Mystique.jpg)

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">MNISTique</h3>
  <p align="center">
    <a href="https://github.com/catiaspsilva/README-template/blob/main/images/docs.txt"><strong>Documentaton</strong></a>
    <a href="https://github.com/catiaspsilva/README-template/issues">Report Bug</a>
    <a href="https://github.com/catiaspsilva/README-template/issues">Add New Feature</a>
  </p>
</p>

<!-- ABOUT THE PROJECT -->
## Introduction

MNISTique V1 is a neural network project built for handwritten digit classification using the MNIST dataset. Designed as an educational and practical exploration of deep learning fundamentals, the project demonstrates how a feedforward neural network can effectively learn to identify digits (0–9) from grayscale 28x28 pixel images.

[MNISTique-V1](https://github.com/MosesTheRedSea/MNISTique)

<!-- GETTING STARTED -->
## Getting Started


### Project Structure
- [common/](./common) – Utility functions and plotting tools shared across the codebase.
- [data/SETUP.md](./data/SETUP.md) – Instructions for preparing and organizing datasets.
- [experiments/](./experiments) – Config files for training parameters and experiment settings.
- [models/hardcoded/](./models/hardcoded) – NumPy-based models for foundational or educational use.
- [models/torch/](./models/torch) – PyTorch models for scalable neural network training.
- [optimizer/](./optimizer) – Custom optimizers (e.g., SGD), modular and training-ready.
- [Mystique.jpg](./Mystique.jpg) – Sample image used for visualization or demos.
- [README.md](./README.md) – Main documentation and usage instructions.

### Dependencies

- torch==2.6.0 | Core library for building and training neural networks (PyTorch).
- matplotlib | Library for creating plots and visualizations of data and training results.
- numpy | Fundamental package for numerical computations and handling arrays efficiently.
- pyyaml | Library to read and write YAML configuration files.
- requests | Simple HTTP library for making web requests and fetching data.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/MosesTheRedSea/MNISTique.git
   ```
   
2. Setup Your Virtual Python Environment
   ```sh
   uv sync || uv run pyproject.toml
   ```
3. Activate The Virtual Python Environment
   ```sh
   source .venv/bin/activate
   ```

<!-- USAGE EXAMPLES -->
## Running Models

You can train different models on the MNISTique dataset using the <code>train.py</code> script and the <code>--model</code> argument. This allows you to choose which model to run without modifying the code.

### Hardcoded
- Train the Softmax Model
  ```sh
  python train.py --model softmax
  ```

- Train the Two-Layer Neural Network Model
  ```sh
  python train.py --model twolayer
  ```
  
### Torch

- Train MNISTique Model
  ```sh
  python train.py 
  ```
### Command-Line Arguments

`--model` Required. Choose which model to train. Options: `softmax` or `twolayer`.

`-h`, `--help` | Show the help message with usage information.

<!-- Authors -->
## Authors
Moses Adewolu - [@MosesTheRedSea](https://twitter.com/MosesTheRedSea) [mosesoluwatobiadewolu@gmail.com](mosesoluwatobiadewolu@gmail.com)

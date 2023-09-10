# cGOAI - Conway's Game of AI

**cGOAI** is a Python project that combines Conway's Game of Life with machine learning. It generates life forms in the Game of Life using a trained neural network model. This project provides a continuous simulation of evolving patterns in the Game of Life.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [License](#license)

## Introduction

Conway's Game of Life is a cellular automaton devised by mathematician John Conway. It consists of a grid of cells that evolve over generations based on simple rules. In this project, we use a neural network to generate initial configurations for the Game of Life, allowing for the creation of diverse life forms.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HeyItsSloth/cGOAI.git
   cd cGOAI
   ```
2. Install the required dependencies:
   ```bash
   CD into the requirements folder
   pip install -r requirements.txt
   ```

## Usage

Train the machine learning model:
  1. Inside the "train" folder
     ```bash
     py train-proper.py
     ```

Run the Game of Life simulation:
 1. Execute the script in the main folder
    ```bash
    py gol.py
    ```

## Customization

In the `train-proper.py` file, you can customize the sampling size, grid size and epochs.
  #### PLEASE NOTE
      Grid Size **MUST** match the Grid size in `gol.py`, else it will error out

In the `gol.py` you can customize the Grid Size (see above for disclaimer lol) 


## License

This project is licensed under the MIT License. Please see the [LICENSE](LICENSE) file located in the "etc" folder for details.



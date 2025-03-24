# SuperBPE: Space Travel for Language Models in Rust
Unofficial implementation of [SuperBPE: Space Travel for Language Models](https://arxiv.org/abs/2503.13423) in Rust.
Just for fun and there may be mistakes. Use with caution.

## System info
- OS: MacOS Sequoia 15.3.2
- Chip: Apple M2 Max

## Installation

1. To install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.79.0 -y`

2. Open a new terminal to set PATH for Rust installation.

3. After opening a new terminal, check the Rust installation by running `rustc --version`.

4. `conda create -n bpe python=3.10`

5. `conda activate bpe`

6. `git clone https://github.com/willxxy/superbpe.git`

6. `pip install -r requirements.txt`

7. `cd bpe` 

8. `maturin develop --release`

## Usage

1. `python main.py`

## Results

For now, here are some results on a smaller dataset with 500,000 characters.

![alt text](./pngs/bpe_comparison.png)

![alt text](./pngs/tokenization_visualization.png)

BPE Training time: **21.56 seconds**
SuperBPE Training time: **7274.00 seconds**

1. Token Length distribution plot looks pretty good. 
2. Training is quite slow...which is expected.
3. Compression rate is worse than BPE? This is just a simple inversion of their byte per token ratio.
4. Looking at the tokenization visualization, it seems like SuperBPE is not merging as many tokens as BPE.

Maybe need to do more analysis and check for bugs.
Feel free to contribute!
# TinyLLM Lab

TinyLLM Lab is an educational project for training a small GPT-style language model from scratch. The goal is to understand the complete language-modeling pipeline, including tokenizer training, dataset preprocessing, Transformer implementation, model training, evaluation, text generation, and performance benchmarking.

This project is intentionally small enough to run on consumer hardware while still demonstrating the core ideas behind modern decoder-only large language models.

## Project Goals

The main goals of this project are to:

* Build a small decoder-only Transformer language model from scratch.
* Train a custom tokenizer on a text dataset.
* Preprocess and batch tokenized text for causal language modeling.
* Implement a training loop with checkpointing and validation.
* Generate text using sampling strategies such as temperature, top-k, and top-p sampling.
* Evaluate model quality using validation loss and perplexity.
* Benchmark training and inference performance.
* Document design tradeoffs clearly for learning and portfolio demonstration.

This is not intended to compete with production LLMs. Instead, it is a hands-on systems and machine learning project for understanding how small language models work end to end.

## Features

* Custom GPT-style Transformer implementation
* Tokenizer training and dataset encoding
* Causal language modeling objective
* Configurable model sizes
* Training and validation loops
* Checkpoint save/load support
* Text generation script
* Perplexity evaluation
* Training loss logging
* Throughput and memory benchmarking
* Reproducible experiment configuration

## High-Level Architecture

```text
Input Text
   |
   v
Tokenizer Training
   |
   v
Tokenized Dataset
   |
   v
Mini-batch Loader
   |
   v
Decoder-only Transformer
   |
   v
Next-token Prediction
   |
   v
Training / Evaluation / Generation
```

The model follows a standard decoder-only Transformer architecture:

```text
Token IDs
   |
   v
Token Embedding + Positional Embedding
   |
   v
Transformer Block x N
   |   - Causal Self-Attention
   |   - Feed-Forward Network
   |   - LayerNorm
   |   - Residual Connections
   |
   v
Language Modeling Head
   |
   v
Next-token Logits
```

## Repository Structure

```text
tinyllm-lab/
├── README.md
├── requirements.txt
├── configs/
│   ├── tiny.yaml
│   ├── small.yaml
│   └── medium.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── tokenizer/
├── tinyllm/
│   ├── __init__.py
│   ├── model.py
│   ├── attention.py
│   ├── transformer.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── generation.py
│   └── utils.py
├── scripts/
│   ├── train_tokenizer.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py
│   └── benchmark.py
├── experiments/
│   ├── runs/
│   └── results/
└── tests/
    ├── test_model.py
    ├── test_attention.py
    └── test_tokenizer.py
```

## Model Variants

The project supports multiple model sizes for experimentation.

| Model  | Layers | Hidden Size | Attention Heads | Context Length | Approx. Parameters |
| ------ | -----: | ----------: | --------------: | -------------: | -----------------: |
| Tiny   |      4 |         256 |               4 |            256 |                ~5M |
| Small  |      6 |         384 |               6 |            512 |               ~15M |
| Medium |      8 |         512 |               8 |            512 |               ~30M |

These configurations are intentionally modest so that they can be trained and tested on a local machine.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/tinyllm-lab.git
cd tinyllm-lab
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
numpy
tqdm
pyyaml
tokenizers
matplotlib
```

## Dataset

This project can be used with any plain-text dataset. For early experiments, a small story, article, code, or educational text dataset is recommended.

Expected raw data format:

```text
data/raw/train.txt
data/raw/valid.txt
```

Each file should contain plain text. The preprocessing script tokenizes the text and creates training sequences for next-token prediction.

## Training a Tokenizer

Train a Byte Pair Encoding tokenizer:

```bash
python scripts/train_tokenizer.py \
  --input data/raw/train.txt \
  --vocab-size 8000 \
  --output data/tokenizer/tokenizer.json
```

Example output:

```text
Tokenizer saved to data/tokenizer/tokenizer.json
Vocabulary size: 8000
```

## Preparing the Dataset

Encode the raw text dataset:

```bash
python scripts/prepare_dataset.py \
  --tokenizer data/tokenizer/tokenizer.json \
  --train data/raw/train.txt \
  --valid data/raw/valid.txt \
  --output data/processed/
```

This creates tokenized binary or tensor files that can be efficiently loaded during training.

## Training

Train the tiny model:

```bash
python scripts/train.py \
  --config configs/tiny.yaml
```

Example configuration:

```yaml
model:
  vocab_size: 8000
  context_length: 256
  n_layers: 4
  n_heads: 4
  hidden_size: 256
  dropout: 0.1

training:
  batch_size: 32
  max_steps: 20000
  learning_rate: 0.0003
  weight_decay: 0.1
  warmup_steps: 1000
  eval_interval: 500
  checkpoint_interval: 1000

data:
  train_path: data/processed/train.bin
  valid_path: data/processed/valid.bin

output:
  checkpoint_dir: experiments/runs/tiny/checkpoints
  log_dir: experiments/runs/tiny/logs
```

## Text Generation

Generate text from a trained checkpoint:

```bash
python scripts/generate.py \
  --checkpoint experiments/runs/tiny/checkpoints/latest.pt \
  --tokenizer data/tokenizer/tokenizer.json \
  --prompt "Once upon a time" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 50
```

Example output:

```text
Once upon a time, there was a small robot who wanted to learn how to read...
```

## Evaluation

Evaluate validation loss and perplexity:

```bash
python scripts/evaluate.py \
  --checkpoint experiments/runs/tiny/checkpoints/latest.pt \
  --tokenizer data/tokenizer/tokenizer.json \
  --valid data/processed/valid.bin
```

Example output:

```text
Validation loss: 2.91
Perplexity: 18.36
```

## Benchmarking

Run a basic training and inference benchmark:

```bash
python scripts/benchmark.py \
  --checkpoint experiments/runs/tiny/checkpoints/latest.pt \
  --batch-size 1 \
  --context-length 256
```

Example benchmark table:

| Model | Device  | Context Length | Tokens/sec | Peak Memory |
| ----- | ------- | -------------: | ---------: | ----------: |
| Tiny  | CPU     |            256 |        TBD |         TBD |
| Tiny  | GPU/MPS |            256 |        TBD |         TBD |
| Small | GPU/MPS |            512 |        TBD |         TBD |

## Experiments

Planned experiments include:

| Experiment           | Description                                                     |
| -------------------- | --------------------------------------------------------------- |
| Tokenizer comparison | Compare character-level tokenization and BPE tokenization       |
| Model scaling        | Compare 5M, 15M, and 30M parameter models                       |
| Context length       | Compare 128, 256, and 512 token context windows                 |
| Sampling methods     | Compare greedy decoding, temperature sampling, top-k, and top-p |
| Training duration    | Analyze validation loss as training steps increase              |
| Inference speed      | Measure tokens/sec across different model sizes                 |

## Example Training Curve

Training and validation loss curves can be generated from the training logs:

```bash
python scripts/plot_loss.py \
  --log experiments/runs/tiny/logs/train_log.jsonl
```

Example report:

```text
Final train loss: TBD
Final validation loss: TBD
Final perplexity: TBD
Best checkpoint: TBD
```

## Implementation Notes

### Causal Self-Attention

The model uses causal self-attention so that each token can only attend to previous tokens and itself. This prevents the model from seeing future tokens during training.

### Positional Embeddings

Since the Transformer architecture does not have recurrence, positional embeddings are added to token embeddings so the model can learn token order.

### Language Modeling Head

The final hidden states are projected back to the vocabulary size to produce next-token logits.

### Loss Function

The model is trained using cross-entropy loss between predicted next-token logits and the actual next tokens.

## Skills Demonstrated

This project demonstrates practical experience with:

* Transformer architecture
* Tokenization
* Language model pretraining
* PyTorch model implementation
* Training loop design
* GPU/MPS/CPU device handling
* Checkpointing and reproducibility
* Evaluation and perplexity measurement
* Text generation algorithms
* Performance benchmarking
* Clean project structure and documentation

## Roadmap

* [ ] Implement tokenizer training
* [ ] Implement dataset preprocessing
* [ ] Implement GPT-style Transformer model
* [ ] Add training loop
* [ ] Add validation and checkpointing
* [ ] Add text generation
* [ ] Add perplexity evaluation
* [ ] Add benchmark script
* [ ] Add training loss plots
* [ ] Add model scaling experiments
* [ ] Add quantization experiment
* [ ] Add optional C++ inference prototype

## Future Work

Possible extensions:

* Add LoRA fine-tuning support
* Add instruction fine-tuning on a small custom dataset
* Add a C++ inference runtime
* Add int8 or int4 quantization
* Export model weights to a simple custom binary format
* Add a small web demo
* Compare PyTorch, MLX, and C++ inference performance
* Add attention visualization
* Add unit tests for each model component

## Limitations

This project trains small models for educational purposes. The generated text quality will be limited by model size, dataset size, training time, and available hardware. The project is intended to demonstrate understanding of LLM training mechanics rather than to produce a production-quality assistant.

## License

This project is released under the MIT License.

## Acknowledgments

This project is inspired by modern decoder-only Transformer language models and educational small-scale LLM training projects. The goal is to make the core ideas behind language model training understandable, reproducible, and practical on local hardware.
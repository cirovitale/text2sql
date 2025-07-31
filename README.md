# Fine-Tuned LLMs for Conversational Text-to-SQL

A system developed as part of the **Natural Language Processing (NLP)** course at the **University of Salerno**, focused on translating natural language into SQL queries in multi-turn conversational settings.
The solution is based on the **CoSQL** dataset and employs **fine-tuning** of the **open-source LLM** ``deepseek-coder-1.3B-instruct`` using **Low-Rank Adaptation (LoRA)**.
The model is trained with **Parameter-Efficient Fine-Tuning (PEFT)** and evaluated through standard metrics such as **Question Match** and **Interaction Match**, measuring the **Exact Match** between predicted and gold queries.


## Prerequisites

### System Requirements

- Python ≥ 3.13
- Conda (recommended for environment management)
- RAM ≥ 16GB+
- VRAM ≥ 6GB+

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/cirovitale/text2sql
cd text2sql
```

### 2. Environment Setup with Conda

#### Create and Activate Conda Environment

```bash
# Create the environment from the provided YAML file
conda env create -f environment.yml

# Activate the environment
conda activate unisa-nlp
```

## Usage

### Training
```bash
# To fine-tune the base model on the CoSQL dataset:
python training.py
```
### Inference


```bash
# To generate SQL queries from natural language prompts:
python inference.py
```
### Evaluation


```bash
# To compute evaluation metrics on the test set:
python testing.py
```
The evaluation includes:

- **Question Match**: Accuracy per individual question (exact SQL match)
- **Interaction Match**: Accuracy on the entire multi-turn interaction

## Project Structure

```
text2sql/
├── training.py              # Training pipeline
├── inference.py             # Inference pipeline
├── testing.py               # Testing pipeline
├── environment.yml          # Conda environment specification
├── dataset/                 # CoSQL dataset
│   └── cosql_dataset/
├── model-007/               # Selected fine-tuned model
│   └── checkpoint-1000/     # Selected fine-tuned checkpoint
├── Documentazione.pdf       # Full academic report
```

## Documentation _(coming soon)_

The full technical documentation of the project, including literature review, methodology, datasets, training pipeline, and experimental results, is available in _italian language_ in: **Documentazione.pdf (coming soon)**

# IOA: Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of the **IOA (Identifier-Organizer-Adapter)** framework for knowledge distillation from Large Language Models (LLMs) to smaller models, as described in the paper *"Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation"*.

## 📋 Overview

IOA is a three-stage pedagogically-inspired framework that systematically transfers knowledge from teacher LLMs to student models through:

1. **Identifier** - Diagnoses knowledge deficiencies and builds dependency graphs
2. **Organizer** - Constructs progressive curricula with mastery-based learning
3. **Adapter** - Adapts knowledge representations to student's cognitive capacity

The framework draws from educational principles including **Bloom's Mastery Learning** and **Vygotsky's Zone of Proximal Development (ZPD)**.

## 🌟 Key Features

- **Knowledge-aware targeting**: Identifies specific knowledge gaps rather than treating capabilities as monolithic
- **Progressive curriculum**: Organizes learning with controlled difficulty increments
- **Cognitive alignment**: Adapts representations through five strategies:
  - Abstract Concept Concretization
  - Complex Reasoning Decomposition
  - Cognitive Load Management
  - Representation Format Optimization
  - Linguistic Complexity Reduction
- **Mastery-based progression**: Students must achieve τ_mastery (90%) before advancing

## 📊 Results

IOA achieves significant improvements over baseline distillation methods:

| Benchmark | Metric | Improvement |
|-----------|--------|-------------|
| DollyEval | ROUGE-L | 94.7% teacher retention |
| MATH | Pass@1 | +19.2% vs baselines |
| HumanEval | Pass@1 | +22.3% vs baselines |

## 🚀 Quick Start

### Installation

```bash
cd ioa-distillation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with default settings
python main.py --domain math_problem_solving --student Qwen/Qwen2.5-3B

# With custom configuration
python main.py --config config.yaml

# Specify teacher model
python main.py --domain code_generation \
    --student Qwen/Qwen2.5-3B \
    --teacher deepseek-ai/DeepSeek-R1
```

### Python API

```python
from config.config import get_default_config
from main import IOAPipeline

# Initialize configuration
config = get_default_config()
config.model.student_model_name = "Qwen/Qwen2.5-3B"
config.model.teacher_model_name = "deepseek-ai/DeepSeek-R1"

# Create and run pipeline
pipeline = IOAPipeline(config, output_dir="./outputs")
results = pipeline.run(domain="math_problem_solving")

print(f"Final evaluation: {results['evaluation_results']}")
```

## 📁 Project Structure

```
ioa_distillation/
├── config/
│   ├── __init__.py
│   └── config.py              # All hyperparameters (τ_gap, τ_ZPD, τ_mastery, etc.)
├── data/
│   ├── __init__.py
│   ├── data_utils.py          # Data processing utilities
│   └── seed_data.py           # Seed dataset management
├── modules/
│   ├── __init__.py
│   ├── identifier.py          # Knowledge Identifier (Eq. 2-6)
│   ├── organizer.py           # Knowledge Organizer (Eq. 7-9)
│   └── adapter.py             # Knowledge Adapter (5 strategies)
├── synthesis/
│   ├── __init__.py
│   ├── prompts.py             # System/User prompt templates
│   └── synthesizer.py         # Data synthesis coordinator
├── training/
│   ├── __init__.py
│   └── trainer.py             # Stage-wise trainer with mastery loops
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py           # ROUGE-L, Pass@k evaluation
├── utils/
│   ├── __init__.py
│   ├── llm_client.py          # LLM API client
│   └── graph_utils.py         # Dependency graph utilities
├── main.py                    # Main entry point (Algorithm 1)
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## ⚙️ Configuration

### Key Hyperparameters

| Parameter | Default | Description | Reference |
|-----------|---------|-------------|-----------|
| `τ_gap` | 0.3 | Deficiency threshold | Eq. 2 |
| `τ_high` | 0.9 | High mastery threshold | Eq. 3 |
| `τ_low` | 0.7 | Low mastery threshold | Eq. 3 |
| `τ_dep` | 0.3 | Dependency inclusion threshold | Eq. 3 |
| `α` | 0.7 | Severity score weight | Eq. 4 |
| `τ_ZPD` | 0.15 | Zone of Proximal Development | Eq. 8 |
| `τ_mastery` | 0.9 | Mastery requirement | Eq. 9 |
| `J_i` | 10 | Samples per seed | Eq. 1 |

### Configuration File Example

```yaml
identifier:
  tau_gap: 0.3
  tau_high: 0.9
  tau_low: 0.7
  alpha: 0.7

organizer:
  tau_zpd: 0.15
  tau_mastery: 0.9

adapter:
  num_samples_per_seed: 10
  enable_verification: true

model:
  student_model_name: "Qwen/Qwen2.5-3B"
  teacher_model_name: "deepseek-ai/DeepSeek-R1"

training:
  learning_rate_full: 2e-5
  global_batch_size: 128
  max_epochs: 3
```

## 📚 Supported Models

### Teacher Models
- OpenAI o1
- DeepSeek-R1
- GPT-4
- Any API-accessible LLM

### Student Models
- Qwen2.5 family (3B, 7B, 14B)
- LLaMA 3.1/3.2 family (3B, 8B)
- Any HuggingFace causal LM

## 📈 Evaluation Benchmarks

| Category | Benchmarks | Metric |
|----------|------------|--------|
| Instruction Following | DollyEval, VicunaEval | ROUGE-L |
| Math Reasoning | GSM8K, MATH, AIME2024 | Pass@1 |
| Code Generation | HumanEval, MBPP, LiveCodeBench | Pass@1 |
| Academic QA | GPQA-Diamond | Accuracy |

## 📦 Seed Data Preparation

Prepare seed data in `./data/seed/` directory with the following structure:

```
data/seed/
├── instruction_following/
│   └── data.jsonl
├── math_problem_solving/
│   └── data.jsonl
├── code_generation/
│   └── data.jsonl
└── academic_knowledge_reasoning/
    └── data.jsonl
```

Each JSONL file should contain entries like:

```json
{"input": "Solve for x: 2x + 5 = 13", "output": "x = 4", "domain": "math_problem_solving", "module": "algebra/linear"}
```

Recommended seed data sizes (per Appendix B):
- Instruction Following: ~800 items
- Math Problem Solving: ~900 items
- Code Generation: ~700 items
- Academic Knowledge Reasoning: ~600 items

## 🔧 Advanced Usage

### Using LoRA for Larger Models

```python
config = get_default_config()
config.model.use_lora = True
config.training.lora_r = 16
config.training.lora_alpha = 32
```

### Custom Teacher API

```python
from utils.llm_client import LLMClient

client = LLMClient(
    api_base="https://your-api-endpoint.com",
    api_key="your-api-key",
    model_name="your-model"
)
```

### Running Specific Phases

```python
pipeline = IOAPipeline(config)
pipeline.setup()

# Run only identification
pipeline.run_identification(domain="math_problem_solving")

# Run only organization
pipeline.run_organization(domain="math_problem_solving")

# Run only synthesis
pipeline.run_adaptation_and_synthesis()
```

## 🙏 Acknowledgments

- This work draws inspiration from Bloom's Mastery Learning and Vygotsky's Zone of Proximal Development
- Thanks to the teams behind LLaMA, Qwen, DeepSeek, and other open-source LLMs
- Built with PyTorch, HuggingFace Transformers, and PEFT
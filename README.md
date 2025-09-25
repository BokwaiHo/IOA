<h1>
  <img src="logo.png" alt="IOA" width="32" height="32">
  Pedagogically-Inspired Data Synthesis for LLM Knowledge Distillation
</h1>


This is a streamlined implementation of the IOA (Identifier-Organizer-Adapter) framework from the paper "Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation".

## Overview

The IOA framework implements a three-stage pedagogical approach to knowledge distillation:

1. **Identifier**: Diagnoses knowledge deficiencies in student models and identifies critical gaps
2. **Organizer**: Creates progressive curricula with mastery-based learning sequences
3. **Adapter**: Adapts knowledge representation to match student model cognitive capacity

## Key Features

- **Knowledge-aware synthesis**: Targets specific deficiencies rather than general data augmentation
- **Curriculum organization**: Structures learning with prerequisite dependencies and mastery gates
- **Cognitive adaptation**: Five adaptation strategies (concretization, decomposition, cognitive load management, format optimization, linguistic simplification)
- **Modular design**: Easy to customize and extend for different domains

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Method 1: Using Main Script

1. **Set up your OpenAI API key**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. **Run different modes**:
```bash
# Run tests
python main.py test

# Run demo with mock data (no API required)
python main.py demo

# Run full experiment with real API calls
python main.py full
```

### Method 2: Direct Usage

```python
from ioa_framework import IOAFramework
from ioa_config import DatasetBuilder, IOAConfig

# Initialize framework
framework = IOAFramework(openai_api_key="your-key")

# Create seed data
builder = DatasetBuilder()
seed_data = builder.create_math_seed_data()
probe_tasks = builder.create_probe_tasks("mathematics")

# Run distillation
results = framework.distill_knowledge(
    teacher_model_name="o1-preview",
    student_model=student_model,
    target_domain="mathematics", 
    seed_data=seed_data,
    probe_tasks=probe_tasks
)
```

## File Structure

- `ioa_framework.py`: Core IOA implementation (Identifier, Organizer, Adapter)
- `ioa_config.py`: Configuration, data utilities, and prompt templates
- `ioa_training.py`: Training utilities and complete usage example
- `ioa_utils.py`: Utility functions, logging, experiment tracking, and validation tools
- `ioa_tests.py`: Comprehensive test suite with unit and integration tests
- `main.py`: Main execution script with multiple modes (test/demo/full)
- `requirements.txt`: Python dependencies
- `config.yaml`: Configuration file (auto-generated)
- `data/`: Directory for seed data and probe tasks

## Configuration

Edit `config.yaml` to customize framework parameters:

```yaml
# Identifier parameters
tau_gap: 0.3        # Deficiency threshold
tau_mastery: 0.9    # Mastery requirement

# Organizer parameters  
tau_zpd: 0.15       # Zone of proximal development

# Training parameters
num_examples_per_stage: 10
learning_rate: 2e-5
batch_size: 8
```

## Supported Domains

Currently implemented:
- **Mathematics**: Arithmetic, algebra, calculus
- **Programming**: Python basics, algorithms, string processing

Easy to extend for new domains by:
1. Adding knowledge modules in `IOAIdentifier.decompose_knowledge()`
2. Creating seed data with `DatasetBuilder`
3. Defining probe tasks for evaluation

## Adaptation Strategies

The framework implements five pedagogical adaptation techniques:

1. **Abstract Concept Concretization**: Use concrete analogies before formal concepts
2. **Complex Reasoning Decomposition**: Break down into explicit step-by-step reasoning
3. **Cognitive Load Management**: Control difficulty increments and problem complexity
4. **Representation Format Optimization**: Consistent templates and structured formats
5. **Linguistic Complexity Reduction**: Simplified language and clear connectors

## Evaluation

Built-in evaluation metrics:
- ROUGE-L for instruction following tasks
- Pass@K for reasoning tasks
- Custom domain-specific correctness checks

## Limitations and Notes

- **API Dependency**: Requires OpenAI API access for teacher model queries
- **Simplified Implementation**: This is a research prototype focused on core concepts
- **Mock Components**: Some evaluation functions use simplified heuristics
- **Resource Requirements**: Fine-tuning requires GPU resources for larger models

## Customization

### Adding New Domains

1. Extend `IOAIdentifier.decompose_knowledge()`:
```python
elif domain == "physics":
    return [
        KnowledgeModule("mechanics", "Classical Mechanics", "foundation", [], 1),
        KnowledgeModule("thermodynamics", "Thermodynamics", "advanced", ["mechanics"], 2),
    ]
```

2. Create domain-specific seed data and probe tasks using `DatasetBuilder`

### Custom Adaptation Strategies

Modify `IOAAdapter.adapt_knowledge_representation()` to implement domain-specific adaptations or extend the five core strategies.

### Integration with Different Models

The framework is designed to work with any HuggingFace compatible model. Simply change the model name in the configuration or training scripts.


## License

This implementation is for research purposes. Please ensure compliance with the original paper's licensing terms and the APIs/models you use.

## Support

This is a research implementation. For questions about the methodology, please refer to the original paper. For implementation issues, check that all dependencies are installed and API keys are configured correctly.
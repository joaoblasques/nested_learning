# Nested Learning Codebase: Comprehensive Explanation

## Table of Contents
1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Architecture Components](#architecture-components)
4. [Implementation Structure](#implementation-structure)
5. [Training and Evaluation](#training-and-evaluation)
6. [Data Pipeline](#data-pipeline)
7. [Usage Examples](#usage-examples)

---

## Overview

### What is This Repository?

This repository is a **high-fidelity, open-source reproduction of Google's Nested Learning (HOPE) architecture**, inspired by two groundbreaking research papers:

1. **Nested Learning** - A new learning paradigm that represents neural networks as nested, multi-level optimization problems
2. **TITANs (Test-Time Memorization)** - A memory architecture that learns to memorize information dynamically during inference

The implementation aims to match the quality of reference implementations while remaining fully open-source and managed with modern Python tooling (`uv` package manager).

### Key Characteristics

- **Language**: Python 3.12+ with PyTorch 2.9.0
- **Purpose**: Research implementation of novel neural architecture for sequence modeling
- **Applications**: Language modeling, continual learning, long-context reasoning
- **Scale**: Supports CPU smoke tests to multi-GPU distributed training
- **Status**: Smoke-test ready with full reproducibility documentation

---

## Core Concepts

### 1. Nested Learning Paradigm

**Traditional Deep Learning** views neural networks as a stack of layers optimized with a single gradient descent process.

**Nested Learning** reconceptualizes neural networks as **hierarchical systems of nested optimization problems**, where different components update at different frequencies (multi-timescale learning).

#### Key Insights:

**Associative Memory Foundation**: All components (optimizers, attention, MLPs) are viewed as associative memory systems that compress their context into parameters.

**Multi-Level Optimization**: Instead of one optimization problem, the model consists of nested optimization processes:
- **Level 0 (Fastest)**: Working memory (attention) - updates every token
- **Level 1**: Short-term memory (TITAN) - updates every few tokens
- **Level 2+**: Long-term memory (CMS blocks) - updates every chunk of tokens

**Update Frequency Hierarchy**: Components are ordered by their update frequency:
```
Fast updates (every token)     → Attention / TITAN memory
Medium updates (every chunk)   → CMS Level 1 
Slow updates (every epoch)     → CMS Level 2+
Slowest (pre-training)         → Base MLP parameters
```

#### Why This Matters:

1. **Continual Learning**: Different timescales allow the model to consolidate memories like the human brain
2. **Efficiency**: Not everything needs to update on every token
3. **Better Optimization**: Each level can use specialized optimizers suited to its timescale
4. **Biological Plausibility**: Mimics how the brain processes information at different frequencies

### 2. Memory Perspective

Nested Learning introduces a **neuroscience-inspired memory framework**:

#### Short-Term Memory (Working Memory)
- **Implementation**: Standard attention mechanism
- **Function**: Processes immediate context within a fixed window
- **Update**: Every forward pass
- **Analogy**: Human working memory (7±2 items)

#### Long-Term Memory (Persistent Storage)
- **Implementation**: TITAN memory module + CMS
- **Function**: Consolidates and stores abstractions from past contexts
- **Update**: Periodic, based on surprise signals
- **Analogy**: Human long-term declarative memory

#### Meta-Memory (Learning to Learn)
- **Implementation**: Deep optimizers with momentum and memory
- **Function**: Learns how to update parameters effectively
- **Update**: Adapts based on gradient history
- **Analogy**: Human meta-cognition and learning strategies

### 3. The "Anterograde Amnesia" Problem

Traditional language models suffer from a condition analogous to anterograde amnesia:
- They can access **immediate context** (short-term memory)
- They can access **pre-trained knowledge** (distant past)
- They **cannot form new long-term memories** during inference

**HOPE architecture solves this** by enabling online consolidation of context into persistent memory during test time.

---

## Architecture Components

### HOPE (Hierarchical Optimization with Persistent Evolution)

HOPE is the complete architecture combining all nested learning components:

```
Input Tokens
    ↓
Embedding
    ↓
[HOPE Block] × N layers
    │
    ├─→ Self-Attention (working memory)
    │       ↓
    ├─→ TITAN Memory (test-time learning)
    │       ↓
    ├─→ Self-Modifier (dynamic weight adjustment)
    │       ↓
    └─→ CMS (Continuum Memory System)
            ↓
Layer Norm
    ↓
LM Head (output logits)
```

### 1. TITAN Memory Module

**Purpose**: Learn to memorize at test time using gradient-based updates

**Key Innovation**: Computes a "surprise signal" (gradient of loss w.r.t. activations) and uses it to update memory parameters during inference.

**Implementation Highlights**:
- Deep MLP structure for memory storage
- Surprise-based memory consolidation
- Forgetting mechanism to manage capacity
- Parallelizable training algorithm

**How It Works**:
```python
# Conceptual flow:
1. Compute prediction from current memory
2. Calculate surprise: how wrong was the prediction?
3. Update memory parameters to reduce surprise
4. Apply forgetting to prevent overflow
```

**Code Location**: `src/nested_learning/titan/memory.py`

### 2. Continuum Memory System (CMS)

**Purpose**: Multi-frequency parameter updates for different abstraction levels

**Key Innovation**: Instead of single MLP, chains multiple MLPs that update at different frequencies.

**Structure**:
```
Input → [MLP_fast] → [MLP_medium] → [MLP_slow] → Output
         Update      Update           Update
         every 10    every 100        every 1000
         tokens      tokens           tokens
```

**Update Rule**:
```python
# Each CMS level updates based on its chunk size
if step % chunk_size == 0:
    θ_level = θ_level - lr * ∇L(θ_level; accumulated_gradients)
```

**Benefits**:
- Captures information at multiple timescales
- Reduces catastrophic forgetting
- Mimics human memory consolidation
- Better parameter efficiency

**Code Location**: `src/nested_learning/cms.py`

### 3. Self-Modifier

**Purpose**: Generate dynamic adjustments to memory updates based on context

**Key Innovation**: Small network that learns to modulate how TITAN memory responds to teach signals.

**Function**:
```python
modifier = SelfModifier(key=context, value=memory_output, error_signal=teach_signal)
adjusted_target = teach_signal + modifier
```

**Code Location**: `src/nested_learning/hope/self_mod.py`

### 4. Deep Optimizers

**Purpose**: Enhanced gradient descent variants that treat momentum as learnable memory

**Key Innovations**:

#### Deep Momentum Gradient Descent (DMGD)
- Replaces scalar momentum with deep MLP
- Better capture of gradient dynamics
- More expressive than traditional momentum

#### Enhanced Update Rules
```python
# Traditional momentum:
m_t = α * m_{t-1} - η * ∇L

# Deep momentum:
m_t = MLP(∇L) where MLP learns optimal accumulation
```

#### Modified Gradient Descent
- Uses L2 regression instead of dot product
- Considers token dependencies
- Delta-rule based updates

**Code Location**: `src/nested_learning/optim/`

---

## Implementation Structure

### Source Code Organization

```
src/nested_learning/
├── __init__.py                 # Package initialization
├── model.py                    # HOPEModel (main model class)
├── hope/
│   ├── block.py               # HOPEBlock (single layer)
│   ├── self_mod.py            # Self-modifier network
│   └── __init__.py
├── titan/
│   ├── memory.py              # TITAN memory implementation
│   ├── model.py               # TITAN-specific model variants
│   └── __init__.py
├── cms.py                      # Continuum Memory System
├── levels.py                   # Level specification & management
├── optim/
│   ├── factory.py             # Optimizer factory
│   ├── manager.py             # Level-wise optimizer management
│   ├── deep.py                # Deep optimizer variants
│   └── __init__.py
├── backbones.py                # Attention implementations
├── data.py                     # Data loading & processing
├── tokenizer.py                # Tokenizer utilities
├── training.py                 # Training loop & utilities
├── assoc_memory.py            # Associative memory primitives
├── memorize.py                # Memorization utilities
├── logging_utils.py           # Logging helpers
└── instrumentation.py         # Metrics & monitoring
```

### Configuration System

The repository uses **Hydra** for configuration management:

```
configs/
├── hope/
│   ├── pilot.yaml              # 760M params, 30B tokens
│   ├── mid.yaml                # 1.3B params, 100B tokens
│   └── target.yaml             # Future large-scale config
├── pilot_smoke.yaml            # Quick CPU test
├── mid_stage2.yaml            # Intermediate training config
└── data/
    ├── refinedweb_mixture.yaml # Data mixture specification
    └── continual_segments_sample.yaml
```

**Configuration Structure**:
```yaml
model:
  vocab_size: 32000
  dim: 1024
  num_layers: 24
  heads: 16
  titan_level:
    name: "titan"
    freq: 1                     # Update every token
    chunk_size: 1
  cms_levels:
    - name: "cms_fast"
      freq: 0.1                 # Update every 10 tokens
      chunk_size: 10
    - name: "cms_slow"
      freq: 0.01                # Update every 100 tokens
      chunk_size: 100
  teach_scale: 1.0
  teach_clip: 0.5
```

### Training Entry Points

```
train.py              # Single GPU/CPU training
train_dist.py         # Distributed Data Parallel (DDP)
train_fsdp.py         # Fully Sharded Data Parallel
train_deepspeed.py    # DeepSpeed integration
```

Each supports full Hydra config overrides via CLI.

---

## Training and Evaluation

### Training Workflow

#### 1. Environment Setup
```bash
# Install Python 3.12
uv python install 3.12

# Install dependencies
uv sync --all-extras

# Verify installation
uv run python -c "import torch; print(torch.__version__)"
```

#### 2. Data Preparation
```bash
# Quick sample for testing
uv run bash scripts/data/run_sample.sh

# Full pipeline (requires significant disk space)
uv run bash scripts/data/run_full.sh
```

#### 3. Smoke Test
```bash
# CPU-only pilot smoke test
uv run bash scripts/run_smoke.sh pilot

# Full end-to-end smoke test
uv run bash scripts/run_e2e_smoke.sh
```

#### 4. Full Training
```bash
# Single GPU
uv run python train.py --config-name pilot

# Multi-GPU DDP
torchrun --nproc_per_node=2 train_dist.py --config-name mid

# With W&B logging
uv run python train.py --config-name pilot \
  logging.enabled=true \
  logging.backend=wandb \
  logging.project=nested-learning
```

### Key Training Features

#### Multi-Timescale Updates
The `LevelOptimizerManager` orchestrates updates:
```python
# Each HOPE block maintains level clocks
for level in levels:
    if step % level.chunk_size == 0:
        # Accumulate gradients over chunk
        loss = compute_loss(accumulated_context)
        optimizer.step(level.parameters(), loss)
        # Reset accumulation
```

#### Teach Signal Mechanism
During training, a teach signal (gradient of loss w.r.t. output) guides memory updates:
```python
# Compute teach signal
teach_signal = grad(loss, model_output)

# Scale and clip for stability
teach_signal = teach_signal * teach_scale
if teach_clip > 0:
    teach_signal = clip_by_norm(teach_signal, teach_clip)

# Forward pass with memory updates
output = model(tokens, teach_signal=teach_signal)
```

### Evaluation Suite

#### 1. Zero-Shot Evaluation
Tests the model on common-sense reasoning without fine-tuning:

```bash
uv run python scripts/eval/zeroshot.py \
  --config configs/hope/mid.yaml \
  --checkpoint checkpoints/mid/step_100000.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks all \
  --max-samples 200 \
  --device cuda:0
```

**Supported Tasks**:
- PIQA (Physical commonsense reasoning)
- HellaSwag (Sentence completion)
- WinoGrande (Pronoun resolution)
- ARC-Easy/Challenge (Science questions)
- BoolQ (Yes/no questions)
- SIQA (Social reasoning)
- CommonsenseQA
- OpenBookQA

#### 2. Needle-in-a-Haystack (NIAH)
Tests long-context retrieval ability:

```bash
uv run python scripts/eval/niah.py \
  --config configs/hope/mid.yaml \
  --checkpoint checkpoints/mid/step_100000.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --context-lengths 2048 4096 8192 16384 \
  --samples-per-length 20
```

**What it measures**: Can the model find a specific fact embedded in a long document?

#### 3. Continual Learning
Measures catastrophic forgetting:

```bash
uv run python scripts/eval/continual.py \
  --config configs/hope/mid.yaml \
  --checkpoints checkpoints/mid/step_50000.pt checkpoints/mid/step_100000.pt \
  --segments-yaml configs/data/continual_segments_sample.yaml \
  --batch-size 4 \
  --max-batches 10
```

**What it measures**: Does performance on early data degrade as model learns new data?

#### 4. Test-Time Memorization
All evaluators support TITAN-style test-time adaptation:

```bash
uv run python scripts/eval/zeroshot.py \
  ... \
  --memorize \
  --memorize-steps 2 \
  --memorize-use-correct-answer \
  --memorize-no-reset
```

**Flags**:
- `--memorize`: Enable test-time memory updates
- `--memorize-steps N`: Number of adaptation passes per example
- `--memorize-use-correct-answer`: Use ground truth during memorization (ablation)
- `--memorize-no-reset`: Retain memories across samples

---

## Data Pipeline

### Overview

The data pipeline processes multiple corpora into tokenized shards:

```
Raw Text Corpora
    ↓
1. Tokenizer Training (SentencePiece)
    ↓
2. Corpus Filtering (language, length, dedup)
    ↓
3. Tokenization & Sharding
    ↓
4. Shard Statistics & Manifests
    ↓
Training Data Loader
```

### Supported Corpora

- **RefinedWeb**: High-quality web text
- **Wikipedia**: Encyclopedia articles
- **C4**: Colossal Clean Crawled Corpus
- **SlimPajama**: Deduplicated web data
- **The Stack**: Source code (multiple languages)

### Pipeline Stages

#### 1. Tokenizer Training
```bash
uv run python scripts/data/train_tokenizer.py \
  --manifest configs/data/refinedweb_mixture.yaml \
  --vocab-size 32000 \
  --output-dir artifacts/tokenizer/refinedweb_mix \
  --log-file data/mixtures/refinedweb_mix_tokenizer.json
```

Creates a SentencePiece unigram model from the data mixture.

#### 2. Filtering & Quality Control
```python
# Filters applied:
- Language detection (keep English)
- Length constraints (min/max tokens)
- Deduplication (approximate near-duplicate removal)
- Quality heuristics (punctuation ratio, etc.)
```

#### 3. Sharding
```bash
uv run python scripts/data/process_mixture.py \
  configs/data/refinedweb_mixture_filtered.yaml \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --log-file data/mixtures/refinedweb_mix_filtered_shards.json
```

Outputs:
```
data/shards/
├── refinedweb_filtered/
│   ├── shard_0000.pt
│   ├── shard_0001.pt
│   └── ...
├── wikipedia_filtered/
│   └── ...
└── ...
```

Each shard is a PyTorch tensor file containing token IDs.

#### 4. Data Loader
The `ShardedDataset` class handles:
- Multi-corpus mixing with specified proportions
- Random shard sampling
- Sequence packing to fixed length
- Efficient memory-mapped loading

**Configuration**:
```yaml
data:
  mixture:
    sources:
      - name: refinedweb
        shards_dir: data/shards/refinedweb_filtered
        weight: 0.6
      - name: wikipedia
        shards_dir: data/shards/wikipedia_filtered
        weight: 0.2
      - name: code
        shards_dir: data/shards/stack_filtered
        weight: 0.2
  sequence_length: 2048
  batch_size: 8
```

---

## Usage Examples

### Example 1: Quick Smoke Test

```bash
# Install and prepare
uv python install 3.12
uv sync --all-extras

# Get sample data
uv run bash scripts/data/run_sample.sh

# Train for a few steps on CPU
uv run python train.py --config-name pilot_smoke

# Check the output
ls artifacts/checkpoints/pilot_smoke/
```

### Example 2: Full Pilot Training (3B tokens)

```bash
# In a tmux session (long running)
tmux new -s pilot_train

# Set up environment
set -a && source git.env && set +a
export UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy

# Launch training with W&B logging
uv run python train.py --config-name pilot \
  logging.enabled=true \
  logging.backend=wandb \
  logging.project=nested-learning \
  logging.run_name=pilot-main-$(date +%Y%m%d%H%M%S) \
  train.device=cuda:1

# Detach: Ctrl+B, then D
```

Expected runtime: ~52 hours on RTX 6000 Ada

### Example 3: Multi-GPU Training

```bash
# Distributed Data Parallel (2 GPUs)
torchrun --nproc_per_node=2 train_dist.py --config-name mid

# Fully Sharded Data Parallel (memory efficient)
torchrun --nproc_per_node=2 train_fsdp.py --config-name mid

# DeepSpeed Zero-3 (very large models)
deepspeed --num_gpus=2 train_deepspeed.py \
  --config-name target \
  deepspeed.config=configs/deepspeed/zero3.json
```

### Example 4: Evaluation After Training

```bash
# Zero-shot on all tasks
uv run python scripts/eval/zeroshot.py \
  --config configs/hope/pilot.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_final.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --tasks all \
  --max-samples 200 \
  --device cuda:0

# Needle-in-haystack at multiple scales
uv run python scripts/eval/niah.py \
  --config configs/hope/pilot.yaml \
  --checkpoint artifacts/checkpoints/pilot/step_final.pt \
  --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model \
  --context-lengths 2048 4096 8192 \
  --samples-per-length 20

# Results saved to eval/ directory
```

### Example 5: Custom Model Configuration

```python
# Create a custom HOPE model
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.levels import LevelSpec

config = ModelConfig(
    vocab_size=32000,
    dim=768,
    num_layers=12,
    heads=12,
    titan_level=LevelSpec(name="titan", freq=1.0, chunk_size=1),
    cms_levels=[
        LevelSpec(name="cms_fast", freq=0.1, chunk_size=10),
        LevelSpec(name="cms_medium", freq=0.01, chunk_size=100),
        LevelSpec(name="cms_slow", freq=0.001, chunk_size=1000),
    ],
    teach_scale=1.0,
    teach_clip=0.5,
)

model = HOPEModel(config)

# Use in training
import torch
tokens = torch.randint(0, 32000, (4, 512))  # batch=4, seq=512
logits = model(tokens)
print(logits.shape)  # [4, 512, 32000]
```

### Example 6: Enabling Test-Time Memorization

```python
# During evaluation
from nested_learning.memorize import enable_test_time_learning

# Enable memorization with specific settings
model = enable_test_time_learning(
    model,
    num_steps=2,              # Gradient steps per example
    learning_rate=1e-3,
    use_correct_answer=False, # Don't cheat during test
    reset_per_sample=True,    # Fresh memory each sample
)

# Now inference will update memory
outputs = model(test_tokens)
```

---

## Key Algorithms in Detail

### Algorithm 1: TITAN Memory Update

```python
def titan_update(memory, query, target, learning_rate):
    """
    Update TITAN memory using surprise-based learning
    
    Args:
        memory: Current memory state (deep MLP)
        query: Input context
        target: Desired output (from teach signal)
        learning_rate: Step size
    """
    # 1. Compute prediction
    prediction = memory(query)
    
    # 2. Compute surprise (error)
    surprise = target - prediction
    
    # 3. Compute gradient of memory parameters
    loss = (surprise ** 2).mean()
    grads = autograd.grad(loss, memory.parameters())
    
    # 4. Update memory (gradient descent)
    for param, grad in zip(memory.parameters(), grads):
        param.data -= learning_rate * grad
    
    # 5. Apply forgetting (weight decay)
    for param in memory.parameters():
        param.data *= (1 - forget_rate)
    
    return prediction
```

### Algorithm 2: CMS Multi-Frequency Update

```python
def cms_forward_and_update(cms, x, step, teach_signal=None):
    """
    Forward pass through CMS with conditional updates
    
    Args:
        cms: Continuum Memory System
        x: Input activations
        step: Global training step
        teach_signal: Optional gradient signal for updates
    """
    # Forward through all levels
    intermediates = []
    current = x
    for level in cms.levels:
        current = level.block(current)
        intermediates.append(current)
    
    # Update levels that should fire this step
    if teach_signal is not None:
        for i, level in enumerate(cms.levels):
            if step % level.chunk_size == 0:
                # Accumulate context over chunk
                context = intermediates[i]
                
                # Compute loss with teach signal
                loss = ((context - teach_signal) ** 2).mean()
                
                # Update this level's parameters
                level.optimizer.zero_grad()
                loss.backward()
                level.optimizer.step()
    
    return current  # Final output
```

### Algorithm 3: Self-Modifier

```python
def self_modifier(key, value, error_signal, hidden_dim=4):
    """
    Generate dynamic modification to memory update
    
    Args:
        key: Context representation
        value: Memory output
        error_signal: Teach signal
    
    Returns:
        modifier: Adjustment to add to teach signal
    """
    # Combine inputs
    combined = torch.cat([key, value, error_signal], dim=-1)
    
    # Small MLP to generate modifier
    h1 = gelu(linear1(combined))
    h2 = gelu(linear2(h1))
    modifier = linear3(h2)
    
    # Scale modifier (usually small)
    modifier = modifier * 0.1
    
    return modifier
```

---

## Performance and Scaling

### Optimization Techniques

#### 1. Mixed Precision Training
```bash
uv run python train.py --config-name pilot \
  train.mixed_precision.enabled=true \
  train.mixed_precision.dtype=bf16
```
Reduces memory usage by ~50% and speeds up training on modern GPUs.

#### 2. Torch Compile
```bash
uv run python train.py --config-name pilot \
  train.compile.enable=true \
  train.compile.mode=max-autotune
```
JIT compiles model for faster execution.

#### 3. Fused Optimizers
```bash
uv run python train.py --config-name pilot \
  optim.type=adamw \
  optim.fused=auto
```
Uses CUDA-fused kernels for Adam optimizer.

#### 4. Muon Optimizer (Experimental)
```bash
uv run python train.py --config-name pilot \
  optim.type=muon \
  optim.lr=2.5e-4 \
  optim.weight_decay=0.01
```
Routes matrix weights through specialized optimizer.

### Scaling Guidelines

| Scale | Params | Tokens | GPUs | Memory | Time |
|-------|--------|--------|------|--------|------|
| Pilot | 760M | 30B | 1-2 | 32 GB | ~52h |
| Mid | 1.3B | 100B | 2-4 | 48 GB | ~5 days |
| Target | 3B+ | 300B+ | 8+ | 80 GB+ | weeks |

**Hardware Recommendations**:
- **Pilot**: Single RTX 6000 Ada (48 GB) or A100 (40/80 GB)
- **Mid**: 2× RTX 6000 Ada or 4× A100
- **Target**: 8× H100 or equivalent

---

## Research Background

### Theoretical Foundations

1. **Associative Memory Theory**: All components viewed as key-value memory systems
2. **Fast Weight Programs**: Dynamic weight updates during inference
3. **Synaptic Consolidation**: Multi-timescale memory consolidation from neuroscience
4. **Delta Rule Learning**: Error-correcting updates to memory
5. **Meta-Learning**: Learning how to learn through nested optimization

### Paper Results

From the Nested Learning paper (HOPE architecture):

**Language Modeling (1.3B params, 100B tokens)**:
- HOPE: 15.11 perplexity (Wikipedia)
- Titan (LMM): 15.60 perplexity
- Transformer++: 18.53 perplexity

**Common-Sense Reasoning**:
- HOPE average: 57.23%
- Titan average: 56.82%
- Best baseline (Samba): 54.00%

**Long Context**:
- Successfully scales to 2M+ token context windows
- Superior performance on needle-in-haystack vs baselines

### Key Innovations

1. **Test-Time Learning**: Memory updates during inference without traditional fine-tuning
2. **Multi-Timescale Updates**: Different components learn at different rates
3. **Biological Inspiration**: Mimics human memory consolidation processes
4. **Continual Learning**: Reduces catastrophic forgetting through multi-frequency parameters
5. **Unified Framework**: Optimizers, attention, and MLPs all viewed as associative memories

---

## Development and Testing

### Running Tests

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/test_model.py

# With coverage
uv run pytest --cov=src/nested_learning

# Verbose output
uv run pytest -v
```

### Code Quality

```bash
# Linting
uv run ruff check .

# Type checking
uv run mypy src

# Format code
uv run ruff format .
```

### Debugging

```bash
# Enable debug logging
export NESTED_LEARNING_DEBUG=1

# Run with Python debugger
uv run python -m pdb train.py --config-name pilot_smoke

# Profile training
uv run python -m cProfile -o profile.stats train.py --config-name pilot_smoke
```

---

## Future Directions

From `docs/future_directions.md`:

1. **Scaling**: Extend to 7B+ parameter models
2. **Architecture Search**: Automated tuning of level frequencies
3. **More Modalities**: Extend to vision, audio, multimodal
4. **Efficiency**: Further optimization for edge deployment
5. **Theory**: Formal analysis of convergence properties
6. **Applications**: Domain-specific variants (code, science, etc.)

---

## References

### Papers
- Behrouz et al. (2024). "Nested Learning: The Illusion of Deep Learning Architectures" [arXiv]
- Behrouz et al. (2024). "Titans: Learning to Memorize at Test Time" [arXiv]

### Related Work
- Vaswani et al. (2017). "Attention is All You Need"
- Gu & Dao (2024). "Mamba: Linear-Time Sequence Modeling"
- Schmidhuber (1992). "Fast Weight Programs"
- Sun et al. (2024). "Learning to Learn at Test Time"

### Code References
- lucidrains/TITAN-pytorch (reference implementation)
- Google Research papers (theoretical foundation)

---

## Contributing

See `docs/guide.md` and `CHANGELOG.md` for contribution guidelines.

**Typical workflow**:
1. Fork the repository
2. Create a feature branch
3. Run tests: `uv run pytest`
4. Format code: `uv run ruff format .`
5. Open a pull request with clear description

---

## License

Apache 2.0 - See LICENSE file

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{behrouz2024nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:TBD},
  year={2024}
}

@article{behrouz2024titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2024}
}
```

---

## Contact and Support

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: See `docs/` directory for detailed guides

---

*Last Updated: 2025-01-12*
*Repository: https://github.com/joaoblasques/nested_learning*

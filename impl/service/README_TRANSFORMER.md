# AlphaZero with Transformer Architecture

This directory contains transformer-based neural network models for the AlphaZero Jass agent.

## Files

- `alpha_zero_model_transformer.py` - Transformer model architectures
- `alpha_zero_train_transformer.py` - Training script for transformer models
- `alpha_zero_model_cnn.py` - CNN/ResNet model architecture (baseline)
- `alpha_zero_train_cnn.py` - Training script for CNN models
- `alpha_zero_utils.py` - Shared utilities for observation encoding

## Transformer Models

Two transformer architectures are available:

### 1. JassTransformer (Default)
A standard transformer that treats the entire game state as a single embedded vector.

**Architecture:**
- Input: 332 features → Embedding dimension
- Transformer Encoder (multi-head attention + feedforward)
- Separate policy and value heads

### 2. JassTransformerSequence
A sequence-based transformer that treats game state components as a sequence.

**Architecture:**
- Splits 332 features into 10 chunks (hand, player histories, current trick, trump)
- Each chunk gets its own embedding + positional encoding
- Transformer processes the sequence
- Global average pooling → policy and value heads

## Training

### Train Transformer Model

```bash
export PYTHONPATH=$PWD
jass-bot/bin/python impl/service/alpha_zero_train_transformer.py
```

### Train CNN Model (Baseline)

```bash
export PYTHONPATH=$PWD
jass-bot/bin/python impl/service/alpha_zero_train_cnn.py
```

## Configuration

Edit the `TrainConfig` class in the training scripts to adjust:

**Training Parameters:**
- `iterations`: Number of training iterations (default: 50)
- `games_per_iteration`: Self-play games per iteration (default: 20)
- `mcts_simulations`: MCTS simulations per move (default: 200)
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Learning rate (default: 0.0001 for transformer, 0.001 for CNN)

**Transformer Architecture:**
- `use_sequence_model`: Use JassTransformerSequence vs JassTransformer (default: False)
- `embed_dim`: Embedding dimension (default: 128)
- `num_heads`: Number of attention heads (default: 4)
- `ff_dim`: Feedforward dimension (default: 512)
- `num_layers`: Number of transformer layers (default: 3)
- `dropout`: Dropout rate (default: 0.1)

## Evaluation

Run the evaluation script to test the trained model:

```bash
export PYTHONPATH=$PWD
jass-bot/bin/python impl/agents/alpha_zero.py
```

The script will automatically detect and load the transformer model if available, otherwise it falls back to the CNN model.

## Model Files

Trained models are saved in `impl/models/`:

- `alpha_zero_transformer_model.pth` - Final transformer model
- `best_alpha_zero_transformer_model.pth` - Best transformer model (lowest loss)
- `checkpoint_iter_N.pth` - Checkpoint every 10 iterations
- `alpha_zero_model_cnn.pth` - CNN model (if trained)

## Key Improvements Over CNN

1. **Attention Mechanism**: Transformers can learn to focus on relevant parts of the game state
2. **Better Generalization**: Sequence model treats game components more naturally
3. **Scalability**: Easier to add new features or modify architecture
4. **State-of-the-art**: Transformers are the current best practice in many domains

## Training Tips

1. **Start Small**: Begin with 3 layers, 128 embedding dim, 4 heads
2. **Monitor Loss**: Both policy and value loss should decrease over time
3. **Replay Buffer**: Larger buffer (50k+) provides better training stability
4. **Learning Rate**: Transformer benefits from lower LR (1e-4) with warmup
5. **Checkpoints**: Save checkpoints regularly to recover from crashes

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size` (try 16 or 8)
- Reduce `embed_dim` (try 64)
- Reduce `num_layers` (try 2)

**Poor Performance:**
- Increase `mcts_simulations` (try 400+)
- Train for more iterations (try 100+)
- Increase replay buffer size

**Slow Training:**
- Ensure GPU is enabled (`torch.cuda.is_available()`)
- Reduce `train_steps` for faster iterations
- Use mixed precision training (add to config)

## Perspective-Invariant Encoding

The observation encoding has been fixed to be **perspective-invariant**:

- Player positions are encoded relative to current player (me, next, partner, previous)
- Score difference is always from current player's team perspective
- This allows the network to generalize across all 4 player positions

This is a critical fix that was missing in earlier versions and significantly improves learning.

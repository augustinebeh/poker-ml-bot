# Poker ML Bot

A machine learning-based poker playing bot that combines neural networks for hand evaluation and imitation learning for strategic decision making.

## Features

- **Hand Evaluator**: Neural network that learns to evaluate poker hand strength and expected value
- **Imitation Learner**: Learns expert poker strategies through imitation learning
- **Integrated Bot**: Combines both models for autonomous poker playing
- **Flexible Architecture**: Modular design allowing easy customization and extension
- **Training Pipeline**: Comprehensive training infrastructure with validation and checkpointing

## Architecture

### Components

#### 1. Hand Evaluator (`poker_bot/hand_evaluator.py`)

A neural network that evaluates poker hands:

- **Input**: Encoded card states and game information
- **Output**: Win probability and expected value estimates
- **Architecture**: 3-layer fully connected network with batch normalization and dropout

Key functions:
- `encode_card()`: Converts card strings to one-hot encodings
- `PokerEnvironment`: Simulates poker game states
- `HandEvaluator`: Neural network for hand strength evaluation

#### 2. Imitation Learner (`poker_bot/imitation_learner.py`)

Learns expert poker decision-making:

- **Input**: Game state encodings
- **Output**: Action probabilities across 6 actions (fold, check, call, raise_small, raise_medium, raise_large)
- **Architecture**: 3-layer fully connected network with batch normalization

Components:
- `PokerDecisionDataset`: Manages training data from expert demonstrations
- `ImitationLearner`: Neural network for action prediction
- `TrainingPipeline`: Full training loop with validation and early stopping

#### 3. Poker Bot (`poker_bot/bot.py`)

Integrates both models for decision making:

- Combines hand evaluator and imitation learner predictions
- Supports deterministic and stochastic action selection
- Maintains game history and decision confidence scores
- Configurable weighting between models

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.1.1
- NumPy 1.24.3
- Pandas 2.0.3
- Scikit-learn 1.3.0
- Matplotlib 3.7.2
- Jupyter 1.0.0

### Setup

1. Clone the repository:
```bash
git clone https://github.com/augustinebeh/poker-ml-bot.git
cd poker-ml-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

#### Train all models:
```bash
python train.py --model all --epochs 20 --samples 5000 --device cpu
```

#### Train specific model:
```bash
# Hand evaluator only
python train.py --model hand_eval --epochs 50

# Imitation learner only
python train.py --model imitation --epochs 20
```

#### Available arguments:
- `--model`: Which model to train ('hand_eval', 'imitation', 'all')
- `--device`: Training device ('cpu' or 'cuda')
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--samples`: Number of training samples to generate
- `--output-dir`: Output directory for model checkpoints

### Using the Bot

#### Basic usage:
```python
from poker_bot.bot import PokerBot
from poker_bot.hand_evaluator import HandEvaluator
from poker_bot.imitation_learner import ImitationLearner
import torch

# Initialize models
hand_eval = HandEvaluator()
imit_learner = ImitationLearner()

# Create bot
bot = PokerBot(
    hand_evaluator=hand_eval,
    imitation_learner=imit_learner,
    hand_eval_weight=0.3  # 30% hand eval, 70% imitation learning
)

# Load trained weights
bot.load_models(
    hand_eval_path='models/hand_evaluator.pth',
    imitation_learner_path='models/imitation_learner.pth'
)

# Make decision
game_state = {
    'community_cards': ['AH', 'KD', 'QC'],
    'player_cards': [['AS', 'KS']],
    'stacks': [1000, 1000],
    'bets': [50, 100],
    'pot': 150
}

action, confidence = bot.decide_action(game_state)
print(f"Action: {action}, Confidence: {confidence:.2f}")
```

#### Advanced usage:
```python
# Create environment
from poker_bot.hand_evaluator import PokerEnvironment

env = PokerEnvironment(num_players=2)
state = env.reset()
env.deal_cards(2)

# Encode state for model
encoded = env.encode_state()

# Get action with history
history = bot.get_game_history()
print(f"Bot made {len(history)} decisions")

# Save trained bot
bot.save_models(
    hand_eval_path='models/hand_evaluator_v2.pth',
    imitation_learner_path='models/imitation_learner_v2.pth'
)
```

### Jupyter Notebooks

Work with the bot in interactive notebooks:

```bash
jupyter notebook
```

## Project Structure

```
poker-ml-bot/
├── poker_bot/
│   ├── __init__.py              # Package initialization
│   ├── hand_evaluator.py        # Hand evaluation and environment
│   ├── imitation_learner.py     # Imitation learning pipeline
│   └── bot.py                   # Main poker bot class
├── train.py                     # Training scripts
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── models/                      # Trained model checkpoints (created during training)
```

## Model Details

### Hand Evaluator Architecture

```
Input (208) 
    → Linear(256) → ReLU → Dropout(0.2)
    → Linear(256) → ReLU → Dropout(0.2)
    → Linear(256) → ReLU → Dropout(0.2)
    → Linear(2)
    → Output: [Win Probability (sigmoid), Expected Value]
```

### Imitation Learner Architecture

```
Input (364)
    → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(6)
    → Output: Action logits (softmax for probabilities)
```

## Training Details

### Hand Evaluator Training

- **Loss**: MSE loss for both win probability and expected value
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Data**: Synthetically generated poker hands with ground truth hand strength
- **Epochs**: 50 (default)
- **Batch Size**: 32

### Imitation Learner Training

- **Loss**: Cross-entropy loss for action classification
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau for learning rate scheduling
- **Early Stopping**: Patience of 10 epochs on validation loss
- **Epochs**: 20 (default)
- **Batch Size**: 32
- **Train/Val Split**: 80/20

## Future Enhancements

- [ ] Self-play reinforcement learning
- [ ] Multi-table tournament support
- [ ] Advanced poker strategies (position awareness, pot odds calculation)
- [ ] Real poker engine integration
- [ ] Opponent modeling
- [ ] Transfer learning from expert databases
- [ ] GPU optimization and mixed precision training

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Author

Augustine Beh

## References

- "Superhuman AI for heads-up no-limit poker" - Noam Brown & Tuomas Sandholm
- "A Brief Survey of Deep Reinforcement Learning" - Kai Arulkumaran et al.
- PyTorch Documentation: https://pytorch.org/

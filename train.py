"""Training script for poker ML bot"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from poker_bot.hand_evaluator import HandEvaluator
from poker_bot.hand_evaluator import encode_game_state

# Configuration
INPUT_SIZE = 364  # Matches encoded state dimensions
HIDDEN_SIZE = 128
OUTPUT_SIZE = 4  # Fold, Call, Raise, All-in
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32


class PokerAgent(nn.Module):
    """Neural network model for poker decision making"""
    
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super(PokerAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


def train_model():
    """Train the poker agent model"""
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = PokerAgent(input_size=INPUT_SIZE).to(device)
    
    # Initialize hand evaluator
    hand_evaluator = HandEvaluator(input_size=INPUT_SIZE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    
    # Training loop (placeholder - integrate with actual game data)
    print(f"Model initialized on device: {device}")
    print(f"Input size: {INPUT_SIZE}")
    print(f"Model parameters: {sum(p.numel() for p in agent.parameters())}")
    
    return agent, hand_evaluator


if __name__ == "__main__":
    model, evaluator = train_model()
    print("Training setup complete!")

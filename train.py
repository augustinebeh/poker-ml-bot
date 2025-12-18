#!/usr/bin/env python3
"""
Training script for poker ML bot with improved architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

from models.hand_evaluator import HandEvaluator
from data.game_state_encoder import GameStateEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train():
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model with corrected input size
    model = HandEvaluator(input_size=364, hidden_size=256, output_size=1)
    model.to(device)
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    logger.info("Model initialized with input_size=364 to match encoded game state dimensions")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop (placeholder for actual training data)
    logger.info("Training setup complete. Ready for training data.")


if __name__ == "__main__":
    train()

"""
Training script for poker imitation learning model.

This script trains a neural network to imitate expert poker decisions
using the imitation learning module from poker_bot package.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from poker_bot.hand_evaluator import HandEvaluator
from poker_bot.imitation_learner import ImitationLearner, PokerDataset


def create_sample_dataset(num_samples: int = 1000):
    """
    Create a sample dataset for training.
    
    Args:
        num_samples: Number of training samples to generate
        
    Returns:
        tuple: (game_states, actions) where game_states has shape (num_samples, 364)
               and actions has shape (num_samples,)
    """
    # Generate random game states with input_size=364
    game_states = np.random.randn(num_samples, 364).astype(np.float32)
    
    # Normalize game states to reasonable ranges [0, 1]
    game_states = (game_states - game_states.min()) / (game_states.max() - game_states.min())
    
    # Generate random actions (e.g., 0=fold, 1=call, 2=raise)
    actions = np.random.randint(0, 3, num_samples).astype(np.int64)
    
    return game_states, actions


def train_model(
    model: ImitationLearner,
    train_loader: DataLoader,
    learning_rate: float = 1e-3,
    num_epochs: int = 10,
    device: str = "cpu"
):
    """
    Train the imitation learning model.
    
    Args:
        model: ImitationLearner model to train
        train_loader: DataLoader with training data
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (game_states, actions) in enumerate(train_loader):
            # Move data to device
            game_states = game_states.to(device)
            actions = actions.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(game_states)
            
            # Compute loss
            loss = criterion(logits, actions)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")


def evaluate_model(model: ImitationLearner, test_loader: DataLoader, device: str = "cpu"):
    """
    Evaluate the model on test data.
    
    Args:
        model: ImitationLearner model to evaluate
        test_loader: DataLoader with test data
        device: Device to evaluate on ('cpu' or 'cuda')
        
    Returns:
        float: Accuracy on test set
    """
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for game_states, actions in test_loader:
            game_states = game_states.to(device)
            actions = actions.to(device)
            
            logits = model(game_states)
            predictions = torch.argmax(logits, dim=1)
            
            correct += (predictions == actions).sum().item()
            total += actions.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def main():
    """Main training script."""
    # Configuration
    input_size = 364  # Poker game state dimension
    hidden_size = 256
    num_actions = 3  # fold, call, raise
    learning_rate = 1e-3
    batch_size = 32
    num_epochs = 10
    num_train_samples = 1000
    num_test_samples = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Initialize hand evaluator (if needed for feature extraction)
    evaluator = HandEvaluator()
    print(f"Hand evaluator initialized")
    
    # Create sample dataset
    print("Creating sample dataset...")
    game_states, actions = create_sample_dataset(num_train_samples + num_test_samples)
    
    # Split into train and test sets
    train_states = game_states[:num_train_samples]
    train_actions = actions[:num_train_samples]
    test_states = game_states[num_train_samples:]
    test_actions = actions[num_train_samples:]
    
    # Create datasets
    train_dataset = PokerDataset(train_states, train_actions)
    test_dataset = PokerDataset(test_states, test_actions)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print(f"Initializing ImitationLearner with input_size={input_size}, "
          f"hidden_size={hidden_size}, num_actions={num_actions}")
    model = ImitationLearner(
        input_size=input_size,
        hidden_size=hidden_size,
        num_actions=num_actions
    )
    
    # Train model
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device
    )
    
    # Evaluate model
    print("Evaluating model on test set...")
    test_accuracy = evaluate_model(model, test_loader, device=device)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "poker_imitation_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

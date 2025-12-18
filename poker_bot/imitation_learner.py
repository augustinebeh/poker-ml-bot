"""Imitation Learning Module

Implements imitation learning for the poker bot, including dataset management,
neural network for decision making, and training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


class PokerDecisionDataset(Dataset):
    """Dataset for imitation learning from expert poker decisions.
    
    Stores game states and corresponding expert actions for training.
    """
    
    def __init__(self, max_size: int = 10000):
        """Initialize the dataset.
        
        Args:
            max_size: Maximum number of samples to store
        """
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_mapping = {
            'fold': 0,
            'check': 1,
            'call': 2,
            'raise_small': 3,
            'raise_medium': 4,
            'raise_large': 5,
        }
        self.reverse_mapping = {v: k for k, v in self.action_mapping.items()}
    
    def add_sample(self, state: np.ndarray, action: str, reward: float):
        """Add a new sample to the dataset.
        
        Args:
            state: Game state encoding (numpy array)
            action: Action taken (string)
            reward: Reward received
        """
        if len(self.states) >= self.max_size:
            # Remove oldest sample
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
        
        self.states.append(state)
        self.actions.append(self.action_mapping[action])
        self.rewards.append(reward)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (state_tensor, action_tensor, reward)
        """
        state = torch.FloatTensor(self.states[idx])
        action = torch.LongTensor([self.actions[idx]])
        reward = float(self.rewards[idx])
        
        return state, action, reward
    
    def get_action_name(self, action_idx: int) -> str:
        """Convert action index to action name.
        
        Args:
            action_idx: Action index
            
        Returns:
            Action name string
        """
        return self.reverse_mapping.get(action_idx, 'unknown')
    
    def clear(self):
        """Clear all samples from the dataset."""
        self.states = []
        self.actions = []
        self.rewards = []


class ImitationLearner(nn.Module):
    """Neural network for imitation learning of poker decisions.
    
    Learns to predict expert poker actions from game states.
    """
    
    def __init__(self, input_size: int = 364, hidden_size: int = 256,
                 num_actions: int = 6, num_layers: int = 3,
                 dropout_rate: float = 0.2):
        """Initialize the ImitationLearner network.
        
        Args:
            input_size: Size of input features (game state encoding)
            hidden_size: Number of hidden units in each layer
            num_actions: Number of possible actions
            num_layers: Number of hidden layers
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # Build network
        layers = []
        prev_size = input_size
        
        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer: action logits
        layers.append(nn.Linear(prev_size, num_actions))
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Action logits of shape (batch_size, num_actions)
        """
        return self.network(x)
    
    def get_action_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities from input state.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Action probabilities of shape (batch_size, num_actions)
        """
        logits = self.forward(x)
        return self.softmax(logits)
    
    def predict_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict best action for input state.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) or (input_size,)
            deterministic: If True, return argmax; if False, sample from distribution
            
        Returns:
            Action indices
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            probs = self.get_action_probabilities(x)
            
            if deterministic:
                actions = torch.argmax(probs, dim=1)
            else:
                actions = torch.multinomial(probs, 1).squeeze(1)
            
            return actions


class TrainingPipeline:
    """Training pipeline for the imitation learner.
    
    Handles data loading, training, validation, and model checkpointing.
    """
    
    def __init__(self, model: ImitationLearner, device: str = 'cpu',
                 learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """Initialize the training pipeline.
        
        Args:
            model: ImitationLearner model to train
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for states, actions, rewards in tqdm(train_loader, desc='Training'):
            states = states.to(self.device)
            actions = actions.squeeze(1).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(states)
            loss = self.criterion(logits, actions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == actions).sum().item()
            total += actions.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for states, actions, rewards in tqdm(val_loader, desc='Validation'):
                states = states.to(self.device)
                actions = actions.squeeze(1).to(self.device)
                
                logits = self.model(states)
                loss = self.criterion(logits, actions)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == actions).sum().item()
                total += actions.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 20, patience: int = 10):
        """Full training loop with early stopping.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.training_history,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['history']

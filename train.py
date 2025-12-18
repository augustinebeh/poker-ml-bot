import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import HandEvaluator
from game_state_encoder import GameStateEncoder
import json

def load_training_data(filepath):
    """Load training data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def prepare_training_tensors(training_data, encoder):
    """Convert training data to PyTorch tensors."""
    states = []
    labels = []
    
    for sample in training_data:
        encoded_state = encoder.encode(sample['game_state'])
        states.append(encoded_state)
        labels.append(sample['hand_strength'])
    
    return torch.FloatTensor(np.array(states)), torch.FloatTensor(np.array(labels))

def train_model(model, train_loader, optimizer, criterion, num_epochs=100, device='cpu'):
    """Train the HandEvaluator model."""
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_states, batch_labels in train_loader:
            batch_states = batch_states.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(batch_states)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return losses

def evaluate_model(model, test_loader, criterion, device='cpu'):
    """Evaluate the model on test data."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_states, batch_labels in test_loader:
            batch_states = batch_states.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            
            outputs = model(batch_states)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

def save_model(model, filepath):
    """Save trained model to file."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath, device='cpu'):
    """Load trained model from file."""
    model.load_state_dict(torch.load(filepath, map_location=device))
    print(f"Model loaded from {filepath}")
    return model

def main():
    """Main training pipeline."""
    # Configuration
    TRAINING_DATA_PATH = 'training_data.json'
    MODEL_PATH = 'hand_evaluator_model.pth'
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Initialize encoder
    encoder = GameStateEncoder()
    
    # Load and prepare data
    print("Loading training data...")
    training_data = load_training_data(TRAINING_DATA_PATH)
    
    print("Preparing training tensors...")
    states, labels = prepare_training_tensors(training_data, encoder)
    
    # Split into train and test sets (80-20 split)
    split_idx = int(0.8 * len(states))
    train_states = states[:split_idx]
    train_labels = labels[:split_idx]
    test_states = states[split_idx:]
    test_labels = labels[split_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_states, train_labels)
    test_dataset = TensorDataset(test_states, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("Initializing HandEvaluator model...")
    model = HandEvaluator(input_size=364)
    model = model.to(DEVICE)
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Train model
    print("Starting training...")
    losses = train_model(model, train_loader, optimizer, criterion, 
                        num_epochs=NUM_EPOCHS, device=DEVICE)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    evaluate_model(model, test_loader, criterion, device=DEVICE)
    
    # Save model
    save_model(model, MODEL_PATH)
    
    print("Training complete!")

if __name__ == '__main__':
    main()

"""Training scripts for poker bot models.

Provides training loops for both hand evaluator and imitation learner.
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from poker_bot.hand_evaluator import HandEvaluator, PokerEnvironment
from poker_bot.imitation_learner import (
    ImitationLearner, PokerDecisionDataset, TrainingPipeline
)


def train_hand_evaluator(
    output_path: str = 'models/hand_evaluator.pth',
    num_samples: int = 10000,
    batch_size: int = 32,
    epochs: int = 50,
    device: str = 'cpu'
):
    """Train the hand evaluator network.
    
    Args:
        output_path: Path to save trained model
        num_samples: Number of training samples to generate
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
    """
    print("=" * 50)
    print("Training Hand Evaluator")
    print("=" * 50)
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = HandEvaluator(input_size=208, hidden_size=256, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on device: {device}")
    
    # Generate synthetic training data
    print(f"\nGenerating {num_samples} training samples...")
    train_states = []
    train_targets = []
    
    env = PokerEnvironment(num_players=2)
    
    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
        
        env.reset()
        env.deal_cards(2)
        env.community_cards = np.random.choice(env.deck, size=np.random.randint(0, 6),
                                              replace=False).tolist()
        
        # Encode state
        state = env.encode_state()
        train_states.append(state)
        
        # Generate target (hand strength and random EV)
        hand_strength = env.evaluate_hand(0)
        ev = np.random.uniform(-1, 1)
        train_targets.append([hand_strength, ev])
    
    train_states = torch.FloatTensor(np.array(train_states)).to(device)
    train_targets = torch.FloatTensor(np.array(train_targets)).to(device)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(train_states))
        states = train_states[perm]
        targets = train_targets[perm]
        
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            optimizer.zero_grad()
            win_prob, ev = model(batch_states)
            
            # Combine losses
            win_loss = criterion(win_prob, batch_targets[:, 0:1])
            ev_loss = criterion(ev, batch_targets[:, 1:2])
            loss = win_loss + ev_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.6f}")
    
    # Save model
    torch.save(model.state_dict(), output_path)
    print(f"\nHand Evaluator saved to {output_path}")
    
    return model


def train_imitation_learner(
    output_path: str = 'models/imitation_learner.pth',
    num_samples: int = 5000,
    batch_size: int = 32,
    epochs: int = 20,
    device: str = 'cpu'
):
    """Train the imitation learner network.
    
    Args:
        output_path: Path to save trained model
        num_samples: Number of training samples to generate
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
    """
    print("\n" + "=" * 50)
    print("Training Imitation Learner")
    print("=" * 50)
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset
    dataset = PokerDecisionDataset(max_size=num_samples)
    
    print(f"Generating {num_samples} training samples...")
    
    # Generate synthetic expert demonstrations
    env = PokerEnvironment(num_players=2)
    action_names = ['fold', 'check', 'call', 'raise_small', 'raise_medium', 'raise_large']
    
    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
        
        env.reset()
        env.deal_cards(2)
        env.community_cards = np.random.choice(env.deck, size=np.random.randint(0, 6),
                                              replace=False).tolist()
        
        state = env.encode_state()
        
        # Simulate expert action based on hand strength
        hand_strength = env.evaluate_hand(0)
        
        if hand_strength < 0.3:
            action = np.random.choice(['fold', 'check'])
        elif hand_strength < 0.6:
            action = np.random.choice(['check', 'call'])
        else:
            action = np.random.choice(['call', 'raise_small', 'raise_medium'])
        
        reward = hand_strength * 100  # Dummy reward
        dataset.add_sample(state, action, reward)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and training pipeline
    model = ImitationLearner(input_size=364, hidden_size=256, num_actions=6)
    pipeline = TrainingPipeline(model, device=device, learning_rate=0.001)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on device: {device}")
    print(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")
    
    # Train
    pipeline.train(train_loader, val_loader, epochs=epochs, patience=5)
    
    # Save model
    pipeline.save_checkpoint(output_path)
    print(f"\nImitation Learner saved to {output_path}")
    
    return model


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train poker bot models')
    parser.add_argument('--model', type=str, choices=['hand_eval', 'imitation', 'all'],
                       default='all', help='Which model to train')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       default='cpu', help='Device to train on')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of training samples to generate')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print(f"Poker Bot Training Script")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Train models
    if args.model in ['hand_eval', 'all']:
        train_hand_evaluator(
            output_path=f'{args.output_dir}/hand_evaluator.pth',
            num_samples=args.samples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device
        )
    
    if args.model in ['imitation', 'all']:
        train_imitation_learner(
            output_path=f'{args.output_dir}/imitation_learner.pth',
            num_samples=args.samples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device
        )
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()

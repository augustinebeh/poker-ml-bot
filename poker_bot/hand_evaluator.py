"""Hand Evaluator Module

Provides neural network-based hand evaluation, poker environment simulation,
and card encoding utilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
import itertools


def encode_card(card: str) -> np.ndarray:
    """Encode a card (e.g., 'AS', 'KH', '2D') into a numerical vector.
    
    Args:
        card: Card representation (rank + suit), e.g., 'AS' for Ace of Spades
        
    Returns:
        A one-hot encoded vector of length 52 representing the card
    """
    ranks = {'A': 0, 'K': 1, 'Q': 2, 'J': 3, 'T': 4, '9': 5, '8': 6,
             '7': 7, '6': 8, '5': 9, '4': 10, '3': 11, '2': 12}
    suits = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
    
    if len(card) != 2:
        raise ValueError(f"Invalid card format: {card}")
    
    rank_idx = ranks[card[0]]
    suit_idx = suits[card[1]]
    card_idx = rank_idx * 4 + suit_idx
    
    encoding = np.zeros(52)
    encoding[card_idx] = 1
    return encoding


def decode_card(encoding: np.ndarray) -> str:
    """Decode a card encoding back to string representation.
    
    Args:
        encoding: One-hot encoded vector of length 52
        
    Returns:
        Card string representation (e.g., 'AS')
    """
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    suits = ['S', 'H', 'D', 'C']
    
    card_idx = np.argmax(encoding)
    rank_idx = card_idx // 4
    suit_idx = card_idx % 4
    
    return ranks[rank_idx] + suits[suit_idx]


class HandEvaluator(nn.Module):
    """Neural network for evaluating poker hands.
    
    Takes as input encoded cards and game state, outputs win probability
    and expected value estimates.
    """
    
    def __init__(self, input_size: int = 208, hidden_size: int = 256,
                 num_layers: int = 3, dropout_rate: float = 0.2):
        """Initialize the HandEvaluator network.
        
        Args:
            input_size: Size of input features (default: 52*4 for all card combinations)
            hidden_size: Number of hidden units in each layer
            num_layers: Number of hidden layers
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Build network
        layers = []
        prev_size = input_size
        
        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer: win probability and expected value
        layers.append(nn.Linear(prev_size, 2))
        
        self.network = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (win_probability, expected_value)
        """
        output = self.network(x)
        win_prob = self.sigmoid(output[:, 0:1])
        ev = output[:, 1:2]  # Expected value can be negative
        return win_prob, ev


class PokerEnvironment:
    """Simulates a poker game environment for agent interaction.
    
    Manages game state, hand evaluation, and reward calculation.
    """
    
    def __init__(self, num_players: int = 2, starting_stack: float = 1000.0):
        """Initialize the poker environment.
        
        Args:
            num_players: Number of players in the game
            starting_stack: Starting chip count for each player
        """
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.reset()
        
    def reset(self) -> Dict:
        """Reset the environment to initial state.
        
        Returns:
            Initial game state dictionary
        """
        self.stacks = [self.starting_stack] * self.num_players
        self.pot = 0.0
        self.bets = [0.0] * self.num_players
        self.community_cards = []
        self.player_cards = [[] for _ in range(self.num_players)]
        self.current_player = 0
        self.done = False
        self.deck = self._create_deck()
        
        return self.get_state()
    
    def _create_deck(self) -> List[str]:
        """Create a standard 52-card poker deck.
        
        Returns:
            List of card strings
        """
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        suits = ['S', 'H', 'D', 'C']
        return [rank + suit for rank in ranks for suit in suits]
    
    def deal_cards(self, num_cards: int = 2):
        """Deal cards to players.
        
        Args:
            num_cards: Number of cards to deal to each player
        """
        for i in range(self.num_players):
            for _ in range(num_cards):
                if self.deck:
                    card = self.deck.pop(0)
                    self.player_cards[i].append(card)
    
    def get_state(self) -> Dict:
        """Get current game state.
        
        Returns:
            Dictionary containing current game state
        """
        return {
            'community_cards': self.community_cards.copy(),
            'player_cards': [cards.copy() for cards in self.player_cards],
            'stacks': self.stacks.copy(),
            'bets': self.bets.copy(),
            'pot': self.pot,
            'current_player': self.current_player,
            'done': self.done,
        }
    
    def encode_state(self) -> np.ndarray:
        """Encode game state into neural network input.
        
        Returns:
            Encoded state as numpy array
        """
        encoding = []
        
        # Encode player's cards (52 * 2)
        player_idx = self.current_player
        for card in self.player_cards[player_idx]:
            encoding.extend(encode_card(card))
        
        # Encode community cards (52 * 5)
        for card in self.community_cards:
            encoding.extend(encode_card(card))
        
        # Pad if needed
        expected_size = 52 * 7  # 2 player cards + 5 community cards
        while len(encoding) < expected_size:
            encoding.extend(np.zeros(52))
        
        return np.array(encoding[:expected_size])
    
    def take_action(self, action: str, amount: float = 0.0) -> Tuple[float, Dict]:
        """Execute an action in the game.
        
        Args:
            action: Action type ('fold', 'check', 'call', 'raise')
            amount: Amount for bet or raise
            
        Returns:
            Tuple of (reward, new_state)
        """
        reward = 0.0
        
        if action == 'fold':
            self.done = True
            reward = -self.bets[self.current_player]
        elif action == 'check':
            pass
        elif action == 'call':
            call_amount = max(self.bets) - self.bets[self.current_player]
            if call_amount <= self.stacks[self.current_player]:
                self.stacks[self.current_player] -= call_amount
                self.bets[self.current_player] += call_amount
                self.pot += call_amount
        elif action == 'raise':
            if amount > 0 and amount <= self.stacks[self.current_player]:
                self.stacks[self.current_player] -= amount
                self.bets[self.current_player] += amount
                self.pot += amount
        
        # Move to next player
        self.current_player = (self.current_player + 1) % self.num_players
        
        return reward, self.get_state()
    
    def evaluate_hand(self, player_idx: int) -> float:
        """Evaluate the strength of a player's hand.
        
        Args:
            player_idx: Index of player
            
        Returns:
            Hand strength score between 0 and 1
        """
        cards = self.player_cards[player_idx] + self.community_cards
        
        if len(cards) < 5:
            return 0.0
        
        # Simple hand strength evaluation
        # In a real implementation, this would use a more sophisticated algorithm
        hand_strength = self._calculate_hand_strength(cards[:7])
        return hand_strength
    
    def _calculate_hand_strength(self, cards: List[str]) -> float:
        """Calculate hand strength from cards.
        
        Args:
            cards: List of card strings
            
        Returns:
            Hand strength score
        """
        ranks = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
                 '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
        
        rank_values = [ranks[card[0]] for card in cards]
        high_card_strength = sum(rank_values) / 98.0  # Normalize
        
        return min(high_card_strength, 1.0)

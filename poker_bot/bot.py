"""Main Poker Bot Module

Combines hand evaluation and imitation learning for autonomous poker playing.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from poker_bot.hand_evaluator import HandEvaluator, PokerEnvironment
from poker_bot.imitation_learner import ImitationLearner


class PokerBot:
    """Autonomous poker playing bot combining neural networks.
    
    Integrates hand evaluation and imitation learning for decision making.
    """
    
    def __init__(self, hand_evaluator: Optional[HandEvaluator] = None,
                 imitation_learner: Optional[ImitationLearner] = None,
                 device: str = 'cpu', hand_eval_weight: float = 0.3):
        """Initialize the PokerBot.
        
        Args:
            hand_evaluator: HandEvaluator network (optional)
            imitation_learner: ImitationLearner network (optional)
            device: Device to use ('cpu' or 'cuda')
            hand_eval_weight: Weight for hand evaluator in decision making (0-1)
        """
        self.device = device
        self.hand_evaluator = hand_evaluator
        self.imitation_learner = imitation_learner
        self.hand_eval_weight = hand_eval_weight
        
        if self.hand_evaluator:
            self.hand_evaluator.to(device).eval()
        if self.imitation_learner:
            self.imitation_learner.to(device).eval()
        
        self.action_space = [
            'fold', 'check', 'call', 'raise_small', 'raise_medium', 'raise_large'
        ]
        self.game_history = []
    
    def decide_action(self, game_state: Dict, deterministic: bool = True) -> Tuple[str, float]:
        """Decide next action based on game state.
        
        Uses both hand evaluation and imitation learning to make decisions.
        
        Args:
            game_state: Current game state dictionary
            deterministic: If True, use argmax; if False, sample
            
        Returns:
            Tuple of (action, confidence_score)
        """
        # Get state encoding
        if 'encoded_state' not in game_state:
            # Simple state encoding: community + player cards
            encoding = self._encode_game_state(game_state)
        else:
            encoding = game_state['encoded_state']
        
        state_tensor = torch.FloatTensor(encoding).unsqueeze(0).to(self.device)
        
        action_probs = np.zeros(len(self.action_space))
        confidence = 0.0
        
        # Get imitation learner prediction
        if self.imitation_learner is not None:
            with torch.no_grad():
                il_probs = self.imitation_learner.get_action_probabilities(state_tensor)
                il_probs = il_probs.cpu().numpy()[0]
                action_probs += (1 - self.hand_eval_weight) * il_probs
        
        # Get hand evaluator prediction
        if self.hand_evaluator is not None:
            with torch.no_grad():
                win_prob, ev = self.hand_evaluator(state_tensor)
                win_prob = win_prob.cpu().item()
                
                # Map hand strength to action probabilities
                he_probs = self._hand_strength_to_action_probs(win_prob, ev.cpu().item())
                action_probs += self.hand_eval_weight * he_probs
        
        # Normalize probabilities
        if np.sum(action_probs) > 0:
            action_probs /= np.sum(action_probs)
        else:
            action_probs = np.ones(len(self.action_space)) / len(self.action_space)
        
        # Select action
        if deterministic:
            action_idx = np.argmax(action_probs)
            confidence = action_probs[action_idx]
        else:
            action_idx = np.random.choice(len(self.action_space), p=action_probs)
            confidence = action_probs[action_idx]
        
        action = self.action_space[action_idx]
        
        # Record decision
        self.game_history.append({
            'state': encoding,
            'action': action,
            'confidence': confidence,
            'action_probs': action_probs.copy(),
        })
        
        return action, confidence
    
    def _encode_game_state(self, game_state: Dict) -> np.ndarray:
        """Encode game state for neural network input.
        
        Args:
            game_state: Game state dictionary
            
        Returns:
            Encoded state as numpy array
        """
        from poker_bot.hand_evaluator import encode_card
        
        encoding = []
        
        # Encode player cards
        player_cards = game_state.get('player_cards', [])
        if len(player_cards) > 0:
            for cards in player_cards:
                for card in cards:
                    encoding.extend(encode_card(card))
        
        # Encode community cards
        community_cards = game_state.get('community_cards', [])
        for card in community_cards:
            encoding.extend(encode_card(card))
        
        # Add game state features
        stacks = game_state.get('stacks', [])
        bets = game_state.get('bets', [])
        pot = game_state.get('pot', 0.0)
        
        if len(stacks) > 0:
            max_stack = max(stacks) if stacks else 1.0
            encoding.extend([s / max_stack for s in stacks])
        
        if len(bets) > 0:
            max_bet = max(bets) if bets else 1.0
            encoding.extend([b / max_bet for b in bets])
        
        encoding.append(pot / 10000.0)  # Normalize pot
        
        # Pad to fixed size
        expected_size = 364  # Standard input size
        while len(encoding) < expected_size:
            encoding.append(0.0)
        
        return np.array(encoding[:expected_size], dtype=np.float32)
    
    def _hand_strength_to_action_probs(self, win_prob: float,
                                      ev: float) -> np.ndarray:
        """Convert hand strength to action probabilities.
        
        Args:
            win_prob: Estimated win probability (0-1)
            ev: Estimated expected value
            
        Returns:
            Action probabilities array
        """
        probs = np.zeros(len(self.action_space))
        
        # fold, check, call, raise_small, raise_medium, raise_large
        if win_prob < 0.3:
            # Weak hand: prefer fold and check
            probs[0] = 0.5  # fold
            probs[1] = 0.5  # check
        elif win_prob < 0.5:
            # Medium hand: prefer check and call
            probs[1] = 0.4  # check
            probs[2] = 0.6  # call
        else:
            # Strong hand: prefer raising
            probs[2] = 0.2  # call
            probs[3] = 0.3  # raise_small
            probs[4] = 0.3  # raise_medium
            probs[5] = 0.2  # raise_large
        
        # Adjust based on expected value
        if ev > 0:
            probs[3:] *= 1.5  # Increase raise probabilities
        elif ev < -0.5:
            probs[0] *= 1.5  # Increase fold probability
        
        # Normalize
        if np.sum(probs) > 0:
            probs /= np.sum(probs)
        else:
            probs = np.ones(len(self.action_space)) / len(self.action_space)
        
        return probs
    
    def update_from_experience(self, outcome: Dict):
        """Update bot's experience from game outcome.
        
        Args:
            outcome: Dictionary containing game outcome and rewards
        """
        # This would integrate feedback to improve models
        # In a production system, would update models based on outcomes
        pass
    
    def get_game_history(self) -> List[Dict]:
        """Get the bot's game history.
        
        Returns:
            List of game decisions and outcomes
        """
        return self.game_history.copy()
    
    def clear_history(self):
        """Clear game history."""
        self.game_history = []
    
    def save_models(self, hand_eval_path: Optional[str] = None,
                   imitation_learner_path: Optional[str] = None):
        """Save model weights.
        
        Args:
            hand_eval_path: Path to save hand evaluator
            imitation_learner_path: Path to save imitation learner
        """
        if self.hand_evaluator and hand_eval_path:
            torch.save(self.hand_evaluator.state_dict(), hand_eval_path)
        
        if self.imitation_learner and imitation_learner_path:
            torch.save(self.imitation_learner.state_dict(), imitation_learner_path)
    
    def load_models(self, hand_eval_path: Optional[str] = None,
                   imitation_learner_path: Optional[str] = None):
        """Load model weights.
        
        Args:
            hand_eval_path: Path to load hand evaluator
            imitation_learner_path: Path to load imitation learner
        """
        if self.hand_evaluator and hand_eval_path:
            self.hand_evaluator.load_state_dict(
                torch.load(hand_eval_path, map_location=self.device)
            )
        
        if self.imitation_learner and imitation_learner_path:
            self.imitation_learner.load_state_dict(
                torch.load(imitation_learner_path, map_location=self.device)
            )

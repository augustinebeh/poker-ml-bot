"""Poker ML Bot - Machine Learning Poker Playing Bot

A reinforcement learning and imitation learning-based poker bot that combines
neural networks for hand evaluation and decision making.
"""

__version__ = "0.1.0"
__author__ = "Augustine Beh"

from poker_bot.hand_evaluator import HandEvaluator, PokerEnvironment, encode_card
from poker_bot.imitation_learner import ImitationLearner, PokerDecisionDataset, TrainingPipeline
from poker_bot.bot import PokerBot

__all__ = [
    "HandEvaluator",
    "PokerEnvironment",
    "encode_card",
    "ImitationLearner",
    "PokerDecisionDataset",
    "TrainingPipeline",
    "PokerBot",
]

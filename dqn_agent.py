import logging
import chess
import numpy as np
import random
import json
import math

class DQNAgent:
    """Deep Q-Learning Neural Network agent for chess"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Hyperparameters
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.95   # Discount factor
        self.alpha = 0.01   # Learning rate
        
        # Simplified network for demonstration
        # In a real implementation, this would use TensorFlow/PyTorch
        self.board_values = {}  # State-value mapping
        
        # Network visualization data (simplified for demonstration)
        self.network_layers = [
            {"name": "input", "neurons": 64 * 12},  # 8x8 board x 12 piece types (6 per color)
            {"name": "hidden1", "neurons": 256},
            {"name": "hidden2", "neurons": 128},
            {"name": "output", "neurons": 1}  # Value output
        ]
    
    def board_to_features(self, fen):
        """Convert board FEN to input features for neural network"""
        board = chess.Board(fen)
        
        # Simplified feature extraction - just count material for demonstration
        # In a real implementation, this would extract more sophisticated features
        piece_values = {
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
            'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
        }
        
        material_balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                material_balance += piece_values[piece.symbol()]
        
        # Return a fingerprint for this position
        return f"{fen}:{material_balance}"
    
    def evaluate_position(self, fen):
        """Evaluate a position using the DQN"""
        try:
            board = chess.Board(fen)
            features = self.board_to_features(fen)
            
            # Check if terminal state
            if board.is_checkmate():
                return -100 if board.turn == chess.WHITE else 100, self.generate_network_visual()
            if board.is_stalemate() or board.is_insufficient_material():
                return 0, self.generate_network_visual()
            
            # Simplified: Use material balance as evaluation if we haven't seen this position
            if features not in self.board_values:
                material_balance = 0
                piece_values = {
                    'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
                    'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
                }
                
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece:
                        material_balance += piece_values[piece.symbol()]
                
                # Add some randomness to evaluation for demonstration
                evaluation = material_balance + (random.random() - 0.5) * 0.5
                self.board_values[features] = evaluation
            
            return self.board_values[features], self.generate_network_visual()
        except Exception as e:
            self.logger.error(f"Error evaluating position: {e}")
            return 0, self.generate_network_visual()
    
    def get_move(self, fen):
        """Get the best move according to the DQN agent with epsilon-greedy exploration"""
        try:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            
            if not legal_moves:
                return None, 0, self.generate_network_visual()
            
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                # Exploration: choose a random move
                chosen_move = random.choice(legal_moves)
                next_fen = self.make_move_and_get_fen(board, chosen_move)
                evaluation, _ = self.evaluate_position(next_fen)
                return chosen_move.uci(), evaluation, self.generate_network_visual()
            
            # Exploitation: choose the best move according to the value function
            best_move = None
            best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
            move_values = []
            
            for move in legal_moves:
                next_fen = self.make_move_and_get_fen(board, move)
                value, _ = self.evaluate_position(next_fen)
                
                move_values.append({
                    "move": move.uci(),
                    "value": value
                })
                
                if board.turn == chess.WHITE:
                    if value > best_value:
                        best_value = value
                        best_move = move
                else:
                    if value < best_value:
                        best_value = value
                        best_move = move
            
            # Scale the confidence based on the relative advantage of the best move
            if len(move_values) > 1:
                values = [m["value"] for m in move_values]
                if board.turn == chess.WHITE:
                    confidence = (best_value - min(values)) / max(1, max(values) - min(values))
                else:
                    confidence = (max(values) - best_value) / max(1, max(values) - min(values))
            else:
                confidence = 1.0
            
            return best_move.uci(), confidence, self.generate_network_visual()
        except Exception as e:
            self.logger.error(f"Error getting move: {e}")
            # Return a random move if there's an error
            if legal_moves:
                return random.choice(legal_moves).uci(), 0, self.generate_network_visual()
            return None, 0, self.generate_network_visual()
    
    def make_move_and_get_fen(self, board, move):
        """Make a move on a copy of the board and return the new FEN"""
        board_copy = board.copy()
        board_copy.push(move)
        return board_copy.fen()
    
    def generate_network_visual(self):
        """Generate a visualization of the neural network state"""
        # In a real implementation, this would extract actual weights and activations
        # For this demo, we'll generate random activations
        visual_data = []
        
        # Generate random activations for visualization
        for layer in self.network_layers:
            neurons = []
            for i in range(min(10, layer["neurons"])):  # Limit to first 10 neurons for visualization
                activation = random.random()
                neurons.append({
                    "id": i,
                    "activation": activation,
                    "weight": random.random() * 2 - 1  # Random weight between -1 and 1
                })
            
            visual_data.append({
                "layer": layer["name"],
                "neurons": neurons
            })
        
        return visual_data
    
    def update_network(self, fen, move, reward, next_fen):
        """Update the DQN based on the observed transition"""
        # In a real implementation, this would update the neural network weights
        # For this demo, we'll do a simple value function update
        features = self.board_to_features(fen)
        next_features = self.board_to_features(next_fen)
        
        if features not in self.board_values:
            self.board_values[features] = 0
        
        if next_features not in self.board_values:
            next_value, _ = self.evaluate_position(next_fen)
            self.board_values[next_features] = next_value
        
        # Q-learning update
        self.board_values[features] += self.alpha * (
            reward + self.gamma * self.board_values[next_features] - self.board_values[features]
        )
        
        return self.generate_network_visual()

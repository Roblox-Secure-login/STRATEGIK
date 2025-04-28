import logging
import chess
import numpy as np
import random
import json
import math
import time
import uuid
from datetime import datetime
from collections import deque

class DQNAgent:
    """Deep Q-Learning Neural Network agent for chess"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Hyperparameters
        self.epsilon = 0.1           # Exploration rate
        self.epsilon_decay = 0.999   # Epsilon decay rate for reducing exploration over time
        self.epsilon_min = 0.01      # Minimum exploration rate
        self.gamma = 0.95            # Discount factor
        self.alpha = 0.01            # Learning rate
        
        # Enhanced network with experience replay
        self.board_values = {}       # State-value mapping
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.batch_size = 32         # Batch size for experience replay
        self.min_replay_size = 100   # Minimum experiences before learning
        self.train_count = 0         # Counter for training iterations
        
        # Game history for learning from past games
        self.game_history = []
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_games = 0
        self.last_game_moves = []
        self.training_stats = []
        
        # Network visualization data (enhanced for demonstration)
        self.network_layers = [
            {"name": "input", "neurons": 64 * 12 + 16},  # 8x8 board x 12 piece types + additional features
            {"name": "hidden1", "neurons": 512},         # Increased capacity
            {"name": "hidden2", "neurons": 256},         # Increased capacity
            {"name": "hidden3", "neurons": 128},         # Added another layer
            {"name": "output", "neurons": 1}             # Value output
        ]
        
        # Initialize piece-square tables
        self.piece_square_tables = self.init_piece_square_tables()
        
        # Weights and biases for the neural network layers (simplified)
        self.weights = self.initialize_weights()
        self.position_count = 0
    
    def initialize_weights(self):
        """Initialize random weights for the neural network (simplified)"""
        weights = []
        for i in range(len(self.network_layers) - 1):
            layer_weights = np.random.randn(
                min(10, self.network_layers[i]["neurons"]),  # Input size (limited for demo)
                min(10, self.network_layers[i+1]["neurons"])  # Output size (limited for demo)
            ) * 0.1  # Scale weights to be small initially
            weights.append(layer_weights)
        return weights
    
    def init_piece_square_tables(self):
        """Initialize piece-square tables for positional evaluation"""
        tables = {}
        
        # Pawn position values (white perspective)
        tables['P'] = np.array([
            [0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5,  5, 10, 25, 25, 10,  5,  5],
            [0,  0,  0, 20, 20,  0,  0,  0],
            [5, -5,-10,  0,  0,-10, -5,  5],
            [5, 10, 10,-20,-20, 10, 10,  5],
            [0,  0,  0,  0,  0,  0,  0,  0]
        ])
        
        # Knight position values
        tables['N'] = np.array([
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50]
        ])
        
        # Bishop position values
        tables['B'] = np.array([
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10, 10, 10, 10, 10, 10, 10,-10],
            [-10,  5,  0,  0,  0,  0,  5,-10],
            [-20,-10,-10,-10,-10,-10,-10,-20]
        ])
        
        # Rook position values
        tables['R'] = np.array([
            [0,  0,  0,  0,  0,  0,  0,  0],
            [5, 10, 10, 10, 10, 10, 10,  5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [0,  0,  0,  5,  5,  0,  0,  0]
        ])
        
        # Queen position values
        tables['Q'] = np.array([
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [-5,  0,  5,  5,  5,  5,  0, -5],
            [0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20]
        ])
        
        # King position values (middle game)
        tables['K'] = np.array([
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [20, 20,  0,  0,  0,  0, 20, 20],
            [20, 30, 10,  0,  0, 10, 30, 20]
        ])
        
        # Mirror tables for black pieces (negate values and flip board)
        for piece in ['P', 'N', 'B', 'R', 'Q', 'K']:
            tables[piece.lower()] = -np.flipud(tables[piece])
        
        return tables
    
    def board_to_features(self, fen):
        """Convert board FEN to input features for neural network"""
        board = chess.Board(fen)
        self.position_count += 1
        
        # Material balance with enhanced values
        piece_values = {
            'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
            'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
        }
        
        # Calculate material balance and positional score
        material_balance = 0
        positional_score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                material_balance += piece_values[piece.symbol()]
                
                # Add positional value based on piece-square tables
                piece_symbol = piece.symbol()
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                positional_score += self.piece_square_tables[piece_symbol][rank][file] * 0.01
        
        # Additional features
        mobility = len(list(board.legal_moves)) * (1 if board.turn == chess.WHITE else -1)
        in_check = 10 if board.is_check() else 0
        castling_rights = sum([board.has_queenside_castling_rights(chess.WHITE),
                              board.has_kingside_castling_rights(chess.WHITE),
                              -board.has_queenside_castling_rights(chess.BLACK),
                              -board.has_kingside_castling_rights(chess.BLACK)])
        
        # Combine features
        combined_score = material_balance + positional_score + mobility * 0.1 + in_check + castling_rights * 5
        
        # Create a more sophisticated feature fingerprint
        feature_key = f"{fen}:{combined_score:.2f}"
        return feature_key
    
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
                
                # Add a small bonus for white to counter the observed black advantage
                white_advantage_bonus = 0.25
                
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece:
                        material_balance += piece_values[piece.symbol()]
                
                # Give white a small advantage to balance performance
                material_balance += white_advantage_bonus
                
                # Add some randomness to evaluation for demonstration
                evaluation = material_balance + (random.random() - 0.5) * 0.3
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
    
    def generate_network_visual(self, skip_for_training=False):
        """Generate a visualization of the neural network state"""
        # Skip visualization during training to improve performance
        if skip_for_training:
            return []
            
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
        
        # Decay epsilon (reduce exploration over time as the agent learns)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Store experience in replay memory
        self.memory.append((features, move, reward, next_features))
        
        # Update network with mini-batch of experiences if we have enough samples
        if len(self.memory) >= self.min_replay_size:
            self.train_with_replay()
            
        return self.generate_network_visual()
        
    def train_with_replay(self):
        """Train the DQN using experience replay"""
        self.train_count += 1
        
        # Sample a mini-batch from the replay memory
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        
        # Update network based on batch of experiences
        for state, action, reward, next_state in mini_batch:
            if state not in self.board_values:
                self.board_values[state] = 0
                
            if next_state not in self.board_values:
                try:
                    # Handle the case where next_state might be a string or a board object
                    if isinstance(next_state, str) and ":" in next_state:
                        fen_str = next_state.split(":")[0]
                        next_value, _ = self.evaluate_position(fen_str)
                    else:
                        next_value, _ = self.evaluate_position(next_state)
                    self.board_values[next_state] = next_value
                except Exception as e:
                    self.logger.error(f"Error in replay training: {e}")
                    self.board_values[next_state] = 0  # Use default value on error
            
            # Update with higher learning rate for replay experiences to prioritize them
            self.board_values[state] += self.alpha * 0.5 * (
                reward + self.gamma * self.board_values[next_state] - self.board_values[state]
            )
    
    def load_games_from_database(self, aggressive_training=True):
        """Load past games from the database for learning
        
        Args:
            aggressive_training: If True, trains on each game after loading to 
                                immediately incorporate knowledge
        """
        try:
            # Import here to avoid circular imports
            import models
            from app import db
            
            # Count available games
            game_count = models.GameHistory.query.count()
            if game_count == 0:
                self.logger.info("No past games found in the database.")
                return False
                
            self.logger.info(f"Loading knowledge from {game_count} past games...")
            
            # Get past games from database - sort by timestamp to prioritize newer games
            past_games = models.GameHistory.query.order_by(models.GameHistory.timestamp.desc()).all()
            games_processed = 0
            
            # Dynamically adjust batch size based on total game count for better efficiency with large datasets
            batch_size = 50 if game_count > 500 else (25 if game_count > 100 else 10)
            
            # Process each game
            for game_idx, game in enumerate(past_games):
                try:
                    # Get moves list from JSON string
                    moves = game.get_moves_list()
                    if not moves:
                        continue
                        
                    # Extract game result to assign appropriate rewards
                    result = game.result
                    white_win = result == "1-0"
                    black_win = result == "0-1"
                    is_draw = result == "1/2-1/2"
                    
                    # Play through the game to learn from it
                    board = chess.Board()
                    for i in range(len(moves) - 1):
                        move_uci = moves[i]
                        move = chess.Move.from_uci(move_uci)
                        
                        # Current state before the move
                        current_fen = board.fen()
                        current_features = self.board_to_features(current_fen)
                        
                        # Make the move
                        board.push(move)
                        next_fen = board.fen()
                        next_features = self.board_to_features(next_fen)
                        
                        # Calculate reward based on game outcome
                        # Higher rewards for moves that led to victory
                        reward = 0
                        if board.is_checkmate():
                            reward = 100 if not board.turn == chess.WHITE else -100
                        elif board.is_stalemate() or board.is_insufficient_material():
                            reward = 0
                        elif board.is_check():
                            reward = 1 if board.turn == chess.WHITE else -1
                        elif i == len(moves) - 2:  # Last move
                            if white_win and board.turn == chess.BLACK:  # White's move led to win
                                reward = 10
                            elif black_win and board.turn == chess.WHITE:  # Black's move led to win
                                reward = -10
                            elif is_draw:
                                reward = 0
                                               
                        # Store transitions in both raw form and feature form for flexibility
                        self.memory.append((current_fen, move_uci, reward, next_fen))
                        
                        # Also store processed features for faster training
                        if current_features not in self.board_values:
                            self.board_values[current_features] = 0
                            
                        # Update position values directly from stored games
                        if white_win and board.turn == chess.WHITE:
                            self.board_values[current_features] += 0.1
                        elif black_win and board.turn == chess.BLACK:
                            self.board_values[current_features] -= 0.1
                    
                    games_processed += 1
                    
                    # Train in batches to immediately incorporate knowledge
                    if aggressive_training and games_processed % batch_size == 0:
                        self.logger.info(f"Processing batch of {batch_size} games, running training...")
                        if len(self.memory) >= self.min_replay_size:
                            # More efficient training for different batch sizes
                            training_iterations = 2 if batch_size > 25 else 3
                            for _ in range(training_iterations):  # Adjust training iterations based on batch size
                                self.train_with_replay()
                except Exception as e:
                    self.logger.error(f"Error processing game: {e}")
                    continue
            
            self.logger.info(f"Successfully processed {games_processed} games from database.")
            
            # Final training pass after loading all games, with optimized iterations based on memory size
            if len(self.memory) >= self.min_replay_size:
                self.logger.info("Running final training pass on all loaded games...")
                # Use fewer iterations for very large datasets to avoid diminishing returns
                iterations = 3 if len(self.memory) > 1000 else (4 if len(self.memory) > 500 else 5)
                for _ in range(iterations):
                    self.train_with_replay()
                
            return games_processed > 0
        except Exception as e:
            self.logger.error(f"Error loading past games: {e}")
            return False
    
    def self_play_training(self, num_games=10):
        """Perform self-play training to improve the agent"""
        # First load knowledge from past games with aggressive training
        loaded_games = self.load_games_from_database(aggressive_training=True)
        self.logger.info(f"Loaded and trained on {loaded_games} games from database")
        
        # Update database stats total for UI
        self.get_training_stats()
        
        training_data = []
        
        # Run self-play games
        self.logger.info(f"Starting self-play training with {num_games} games")
        for game_num in range(num_games):
            board = chess.Board()
            game_moves = []
            move_count = 0
            game_reward = 0
            
            # Play a complete game against itself
            while not board.is_game_over():
                move_count += 1
                current_fen = board.fen()
                
                # Get AI move for current board state
                move_uci, confidence, _ = self.get_move(current_fen)
                if move_uci is None:
                    break
                    
                move = chess.Move.from_uci(move_uci)
                game_moves.append(move_uci)
                
                # Make the move
                board.push(move)
                next_fen = board.fen()
                
                # Calculate reward
                reward = 0
                if board.is_checkmate():
                    reward = 100 if not board.turn == chess.WHITE else -100
                    game_reward = reward
                elif board.is_stalemate() or board.is_insufficient_material():
                    reward = 0
                    game_reward = 0
                elif board.is_check():
                    reward = 1 if board.turn == chess.WHITE else -1
                    
                # Store the transition
                self.memory.append((current_fen, move_uci, reward, next_fen))
                
                # Update the network without generating visualization during training
                features = self.board_to_features(current_fen)
                next_features = self.board_to_features(next_fen)
                
                if features not in self.board_values:
                    self.board_values[features] = 0
                
                if next_features not in self.board_values:
                    # Evaluate position without visualization during training
                    board = chess.Board(next_fen)
                    material_balance = 0
                    piece_values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0,
                                  "p": -1, "n": -3, "b": -3, "r": -5, "q": -9, "k": 0}
                    
                    # Add a small bonus for white to counter the observed black advantage
                    white_advantage_bonus = 0.25
                    
                    for square in chess.SQUARES:
                        piece = board.piece_at(square)
                        if piece:
                            material_balance += piece_values[piece.symbol()]
                    
                    # Give white a small advantage to balance performance
                    material_balance += white_advantage_bonus
                    
                    # Add some randomness to evaluation but reduce the range
                    evaluation = material_balance + (random.random() - 0.5) * 0.3
                    self.board_values[next_features] = evaluation
                
                # Q-learning update
                self.board_values[features] += self.alpha * (
                    reward + self.gamma * self.board_values[next_features] - self.board_values[features]
                )
                
                # Decay epsilon (reduce exploration over time as the agent learns)
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                    
                # Store experience in replay memory
                self.memory.append((features, move_uci, reward, next_features))
                
                # Perform mini-batch training without visualization
                if len(self.memory) >= self.min_replay_size:
                    self.train_count += 1
                    
                    # Sample a mini-batch 
                    batch_size = min(self.batch_size, len(self.memory))
                    mini_batch = random.sample(self.memory, batch_size)
                    
                    # Update network based on batch of experiences
                    for state, action, r, next_state in mini_batch:
                        if state not in self.board_values:
                            self.board_values[state] = 0
                            
                        if next_state not in self.board_values:
                            self.board_values[next_state] = 0
                        
                        # Update with higher learning rate for experiences
                        self.board_values[state] += self.alpha * 0.5 * (
                            r + self.gamma * self.board_values[next_state] - self.board_values[state]
                        )
                
            # Record game result
            result = board.result()
            self.total_games += 1
            self.last_game_moves = game_moves
            
            if result == "1-0":
                self.wins += 1
            elif result == "0-1":
                self.losses += 1
            else:
                self.draws += 1
                
            # Record training statistics
            training_data.append({
                "game": self.total_games,
                "moves": move_count,
                "moves_list": game_moves,  # Store the actual list of moves in UCI format
                "result": result,
                "reward": game_reward,
                "epsilon": self.epsilon
            })
            
        self.training_stats.extend(training_data)
        
        # Save games to database for future training
        try:
            # Import here to avoid circular imports
            import models
            import json
            import uuid
            from app import db
            from datetime import datetime
            
            # For each completed game, save to database
            for game_data in training_data:
                try:
                    # Create a new game history entry
                    game_history = models.GameHistory(
                        game_id=str(uuid.uuid4()),
                        result=game_data["result"],
                        white_player="AI",
                        black_player="AI",
                        timestamp=datetime.utcnow(),
                        game_type="self-play",
                        evaluation=game_data["reward"]
                    )
                    
                    # Set moves list
                    game_history.set_moves_list(game_data["moves_list"])
                    
                    # Add to database
                    db.session.add(game_history)
                    
                except Exception as e:
                    self.logger.error(f"Error saving game to database: {e}")
                    continue
                    
            # Commit all games at once
            db.session.commit()
            self.logger.info(f"Successfully saved {len(training_data)} games to database")
            
            # Update training statistics in database
            try:
                # Calculate stats
                session_id = str(uuid.uuid4())[:8]  # Use part of UUID as session ID
                white_wins = sum(1 for game in training_data if game["result"] == "1-0")
                black_wins = sum(1 for game in training_data if game["result"] == "0-1")
                draws = sum(1 for game in training_data if game["result"] == "1/2-1/2")
                avg_game_length = sum(game["moves"] for game in training_data) / max(1, len(training_data))
                avg_reward = sum(game["reward"] for game in training_data) / max(1, len(training_data))
                
                # Create training stats record
                training_stats = models.TrainingStats(
                    training_session=session_id,
                    total_games=len(training_data),
                    white_wins=white_wins,
                    black_wins=black_wins,
                    draws=draws,
                    avg_game_length=avg_game_length,
                    avg_reward=avg_reward,
                    epsilon=self.epsilon,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    positions_evaluated=self.position_count
                )
                
                db.session.add(training_stats)
                db.session.commit()
                self.logger.info(f"Successfully saved training stats to database")
                
            except Exception as e:
                self.logger.error(f"Error saving training stats to database: {e}")
                db.session.rollback()
                
        except Exception as e:
            self.logger.error(f"Database operation failed: {e}")
            
        return training_data
        
    def get_training_stats(self):
        """Return statistics about the training progress"""
        # Calculate local stats from current session
        win_rate = (self.wins / max(1, self.total_games)) * 100
        avg_game_length = sum(stat["moves"] for stat in self.training_stats) / max(1, len(self.training_stats))
        avg_reward = sum(stat["reward"] for stat in self.training_stats) / max(1, len(self.training_stats))
        
        # Initialize database stats
        db_stats = {
            "db_total_games": 0,
            "db_white_wins": 0,
            "db_black_wins": 0,
            "db_draws": 0,
            "white_win_percentage": 0,
            "black_win_percentage": 0,
            "draw_percentage": 0
        }
        
        # Try to get global stats from database
        try:
            # Import here to avoid circular imports
            import models
            from app import db
            from sqlalchemy import func
            
            # Get game counts by result
            game_counts = db.session.query(
                models.GameHistory.result,
                func.count(models.GameHistory.id).label('count')
            ).group_by(models.GameHistory.result).all()
            
            # Calculate stats
            total_db_games = 0
            white_wins = 0
            black_wins = 0
            draws = 0
            
            for result, count in game_counts:
                total_db_games += count
                if result == "1-0":
                    white_wins = count
                elif result == "0-1":
                    black_wins = count
                else:  # Draw results like "1/2-1/2"
                    draws = count
            
            # Calculate percentages
            if total_db_games > 0:
                white_win_pct = (white_wins / total_db_games) * 100
                black_win_pct = (black_wins / total_db_games) * 100
                draw_pct = (draws / total_db_games) * 100
            else:
                white_win_pct = black_win_pct = draw_pct = 0
                
            # Get average game length without using JSON functions 
            # which might not be available in all PostgreSQL versions
            games = models.GameHistory.query.all()
            move_counts = []
            for game in games:
                # Use the get_moves_list method which loads JSON
                moves = game.get_moves_list()
                move_counts.append(len(moves))
            
            # Calculate average
            avg_db_game_length = sum(move_counts) / max(1, len(move_counts)) if move_counts else 0
            
            # Get average reward
            avg_db_reward = db.session.query(
                func.avg(models.GameHistory.evaluation)
            ).scalar() or 0
            
            # Update db_stats
            db_stats = {
                "db_total_games": total_db_games,
                "db_white_wins": white_wins,
                "db_black_wins": black_wins,
                "db_draws": draws,
                "white_win_percentage": white_win_pct,
                "black_win_percentage": black_win_pct,
                "draw_percentage": draw_pct,
                "db_avg_game_length": float(avg_db_game_length),
                "db_avg_reward": float(avg_db_reward)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            # Continue with default values if database query fails
        
        # Combine local and database stats
        combined_stats = {
            "total_games": self.total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": win_rate,
            "avg_game_length": avg_game_length,
            "avg_reward": avg_reward,
            "epsilon": self.epsilon,
            "positions_evaluated": self.position_count,
            "last_game": self.last_game_moves,
            "training_history": self.training_stats[-10:] if self.training_stats else []
        }
        
        # Merge with database stats
        combined_stats.update(db_stats)
        
        return combined_stats

from datetime import datetime
import json
import os
from database import db

class GameHistory(db.Model):
    """Model to represent a chess game played by the AI or a user"""
    __tablename__ = 'game_history'
    
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String(36), unique=True, nullable=False)  # UUID
    moves = db.Column(db.Text, nullable=False)  # Stores moves in UCI format as JSON string
    result = db.Column(db.String(10), nullable=False)  # "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (draw)
    white_player = db.Column(db.String(50), nullable=False, default="AI")  # Player name or "AI"
    black_player = db.Column(db.String(50), nullable=False, default="AI")  # Player name or "AI"
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    fen_position = db.Column(db.String(100), nullable=True)  # Final position in FEN format
    game_type = db.Column(db.String(20), nullable=False, default="self-play")  # "self-play", "user-vs-ai", etc.
    evaluation = db.Column(db.Float, nullable=True)  # Final position evaluation
    
    def __repr__(self):
        return f"<GameHistory {self.game_id}: {self.result}>"
        
    def get_moves_list(self):
        """Returns the moves as a Python list"""
        try:
            return json.loads(self.moves)
        except:
            return []
            
    def set_moves_list(self, moves_list):
        """Sets the moves from a Python list"""
        self.moves = json.dumps(moves_list)
        
class TrainingStats(db.Model):
    """Model to track AI training statistics"""
    __tablename__ = 'training_stats'
    
    id = db.Column(db.Integer, primary_key=True)
    training_session = db.Column(db.String(36), nullable=False)  # Session identifier
    total_games = db.Column(db.Integer, nullable=False, default=0)
    white_wins = db.Column(db.Integer, nullable=False, default=0)
    black_wins = db.Column(db.Integer, nullable=False, default=0)
    draws = db.Column(db.Integer, nullable=False, default=0)
    avg_game_length = db.Column(db.Float, nullable=False, default=0.0)
    avg_reward = db.Column(db.Float, nullable=False, default=0.0)
    epsilon = db.Column(db.Float, nullable=False, default=0.1)
    alpha = db.Column(db.Float, nullable=False, default=0.01)
    gamma = db.Column(db.Float, nullable=False, default=0.95)
    positions_evaluated = db.Column(db.Integer, nullable=False, default=0)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<TrainingStats session={self.training_session}, games={self.total_games}>"
        
    def white_win_percentage(self):
        """Calculate white win percentage"""
        if self.total_games == 0:
            return 0
        return (self.white_wins / self.total_games) * 100
        
    def black_win_percentage(self):
        """Calculate black win percentage"""
        if self.total_games == 0:
            return 0
        return (self.black_wins / self.total_games) * 100
        
    def draw_percentage(self):
        """Calculate draw percentage"""
        if self.total_games == 0:
            return 0
        return (self.draws / self.total_games) * 100

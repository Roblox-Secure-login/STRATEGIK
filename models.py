# This file is kept as a placeholder for potential user data models
# Currently our application doesn't need database models as it focuses on the chess AI
# Could be extended in the future to store game history, user profiles, etc.

class GameHistory:
    """A simple model to represent a game history (not using a database currently)"""
    def __init__(self, game_id, moves, result, timestamp):
        self.game_id = game_id
        self.moves = moves  # list of moves in algebraic notation
        self.result = result  # 1-0 (white wins), 0-1 (black wins), 1/2-1/2 (draw)
        self.timestamp = timestamp

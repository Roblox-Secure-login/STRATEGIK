import chess
import logging

class ChessEngine:
    """Chess engine to handle game mechanics and rules"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_legal_moves(self, fen, square):
        """Get all legal moves for a piece at a specific square"""
        try:
            board = chess.Board(fen)
            square_idx = chess.parse_square(square)
            
            legal_moves = []
            for move in board.legal_moves:
                if move.from_square == square_idx:
                    legal_moves.append(move.uci())
            
            return legal_moves
        except Exception as e:
            self.logger.error(f"Error getting legal moves: {e}")
            return []
    
    def make_move(self, fen, move_uci):
        """Make a move on the board and return the new position"""
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            
            if move in board.legal_moves:
                board.push(move)
                return board.fen()
            else:
                self.logger.warning(f"Illegal move attempted: {move_uci}")
                return fen
        except Exception as e:
            self.logger.error(f"Error making move: {e}")
            return fen
    
    def check_game_state(self, fen):
        """Check if the game is in a terminal state"""
        try:
            board = chess.Board(fen)
            
            # Check for checkmate
            if board.is_checkmate():
                return {
                    "state": "checkmate",
                    "message": "Checkmate! " + ("Black" if board.turn == chess.WHITE else "White") + " wins."
                }
            
            # Check for stalemate
            if board.is_stalemate():
                return {
                    "state": "stalemate",
                    "message": "Draw by stalemate."
                }
            
            # Check for insufficient material
            if board.is_insufficient_material():
                return {
                    "state": "draw",
                    "message": "Draw by insufficient material."
                }
            
            # Check for threefold repetition
            if board.can_claim_threefold_repetition():
                return {
                    "state": "repetition",
                    "message": "Draw by threefold repetition can be claimed."
                }
            
            # Check for fifty-move rule
            if board.can_claim_fifty_moves():
                return {
                    "state": "fifty_moves",
                    "message": "Draw by fifty-move rule can be claimed."
                }
            
            # Game is ongoing
            return {
                "state": "ongoing",
                "message": "Game in progress."
            }
        except Exception as e:
            self.logger.error(f"Error checking game state: {e}")
            return {
                "state": "error",
                "message": "Error checking game state."
            }
    
    def get_piece_at(self, fen, square):
        """Get the piece at a specific square"""
        try:
            board = chess.Board(fen)
            square_idx = chess.parse_square(square)
            piece = board.piece_at(square_idx)
            
            if piece:
                return {
                    "piece": piece.symbol(),
                    "color": "white" if piece.color == chess.WHITE else "black"
                }
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error getting piece: {e}")
            return None

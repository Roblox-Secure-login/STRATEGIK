import os
import logging
from flask import Flask, render_template, jsonify, request
from chess_engine import ChessEngine
from dqn_agent import DQNAgent

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "strategik-chess-ai-secret")

# Initialize the chess engine and DQN agent
chess_engine = ChessEngine()
dqn_agent = DQNAgent()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/ai-description')
def ai_description():
    return render_template('ai_description.html')

@app.route('/watch-training')
def watch_training():
    return render_template('watch_training.html')

@app.route('/api/get-ai-move', methods=['POST'])
def get_ai_move():
    """Get the AI's next move based on the current board state"""
    data = request.get_json()
    fen = data.get('fen')
    
    # Use the DQN agent to calculate the next move
    move, confidence, network_states = dqn_agent.get_move(fen)
    
    return jsonify({
        'move': move,
        'confidence': confidence,
        'network_states': network_states
    })

@app.route('/api/evaluate-position', methods=['POST'])
def evaluate_position():
    """Evaluate the current board position"""
    data = request.get_json()
    fen = data.get('fen')
    
    # Get an evaluation of the current position
    evaluation, network_states = dqn_agent.evaluate_position(fen)
    
    return jsonify({
        'evaluation': evaluation,
        'network_states': network_states
    })

@app.route('/api/check-game-state', methods=['POST'])
def check_game_state():
    """Check if the game is in a terminal state (checkmate, stalemate, etc.)"""
    data = request.get_json()
    fen = data.get('fen')
    
    # Check the game state
    game_state = chess_engine.check_game_state(fen)
    
    return jsonify({
        'state': game_state['state'],
        'message': game_state['message']
    })

@app.route('/api/get-legal-moves', methods=['POST'])
def get_legal_moves():
    """Get all legal moves for a specific piece"""
    data = request.get_json()
    fen = data.get('fen')
    square = data.get('square')
    
    # Get legal moves for the piece
    legal_moves = chess_engine.get_legal_moves(fen, square)
    
    return jsonify({
        'moves': legal_moves
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

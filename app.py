import os
import logging
import uuid
import json
from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from chess_engine import ChessEngine
from dqn_agent import DQNAgent

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "strategik-chess-ai-secret")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the database
db = SQLAlchemy(app)

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

@app.route('/api/start-training', methods=['POST'])
def start_training():
    """Start self-play training for the AI"""
    data = request.get_json()
    num_games = data.get('num_games', 10)
    
    # Start training in a separate thread to not block the response
    try:
        training_results = dqn_agent.self_play_training(num_games)
        
        return jsonify({
            'status': 'success',
            'games_completed': len(training_results),
            'training_data': training_results
        })
    except Exception as e:
        logging.error(f"Error in training: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-training-stats', methods=['GET'])
def get_training_stats():
    """Get statistics about the AI's training progress"""
    try:
        stats = dqn_agent.get_training_stats()
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
    except Exception as e:
        logging.error(f"Error getting training stats: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/update-training-parameters', methods=['POST'])
def update_training_parameters():
    """Update the DQN agent's hyperparameters"""
    data = request.get_json()
    
    try:
        if 'epsilon' in data:
            dqn_agent.epsilon = float(data['epsilon'])
        if 'alpha' in data:
            dqn_agent.alpha = float(data['alpha'])
        if 'gamma' in data:
            dqn_agent.gamma = float(data['gamma'])
            
        return jsonify({
            'status': 'success',
            'parameters': {
                'epsilon': dqn_agent.epsilon,
                'alpha': dqn_agent.alpha,
                'gamma': dqn_agent.gamma
            }
        })
    except Exception as e:
        logging.error(f"Error updating parameters: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

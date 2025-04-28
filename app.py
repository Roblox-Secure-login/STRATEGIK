import os
import logging
import uuid
import json
from flask import Flask, render_template, jsonify, request
from database import db
from chess_engine import ChessEngine
from dqn_agent import DQNAgent

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "strategyk-chess-ai-secret")

# Set up database path - use SQLite for portability and store in workspace
# Using workspace root directory makes the db file more accessible for download
db_path = os.path.join(os.path.expanduser("~"), "workspace", "strategyk_chess.db")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,  # Verify database connection before query
    "pool_recycle": 300     # Recycle connections every 5 minutes
}

# Log the database location
logging.info(f"Using SQLite database at: {db_path}")

# Initialize the chess engine and DQN agent
chess_engine = ChessEngine()
dqn_agent = DQNAgent()

# Pre-load database games into the DQN agent when the application starts
# This ensures the AI retains knowledge across application restarts
def load_games_at_startup():
    """Load games from database into AI memory on first request"""
    # We'll load games only once
    if not hasattr(load_games_at_startup, 'loaded'):
        logging.info("Loading past games from database into AI memory...")
        try:
            dqn_agent.load_games_from_database(aggressive_training=True)
            load_games_at_startup.loaded = True
            logging.info("Successfully loaded games from database")
        except Exception as e:
            logging.error(f"Error loading games at startup: {e}")
            
# Register route to trigger database loading on first access
@app.before_request
def before_request():
    load_games_at_startup()

@app.route('/')
def home():
    return render_template('about.html')  # Using about.html as the homepage

@app.route('/play')
def play():
    return render_template('index.html')  # The chess game page

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
    
    # Limit max games for web requests to prevent timeout, allowing up to 1000 games
    if num_games > 1000:
        num_games = min(num_games, 1000)
        logging.info(f"Limited training games to {num_games} to prevent timeout")
    
    # Start training
    try:
        training_results = dqn_agent.self_play_training(num_games)
        
        # Save training session to database
        training_id = str(uuid.uuid4())
        
        # Create training stats summary
        total_games = len(training_results)
        white_wins = sum(1 for game in training_results if game['result'] == '1-0')
        black_wins = sum(1 for game in training_results if game['result'] == '0-1')
        draws = sum(1 for game in training_results if game['result'] == '1/2-1/2')
        avg_game_length = sum(game['moves'] for game in training_results) / max(1, total_games)
        avg_reward = sum(game['reward'] for game in training_results) / max(1, total_games)
        
        # Import models here to avoid circular imports
        import models
        
        # Save to database
        stats = models.TrainingStats(
            training_session=training_id,
            total_games=total_games,
            white_wins=white_wins,
            black_wins=black_wins,
            draws=draws,
            avg_game_length=avg_game_length,
            avg_reward=avg_reward,
            epsilon=dqn_agent.epsilon,
            alpha=dqn_agent.alpha,
            gamma=dqn_agent.gamma,
            positions_evaluated=dqn_agent.position_count
        )
        db.session.add(stats)
        
        # Save each game to the database
        for i, game_data in enumerate(training_results):
            game_id = str(uuid.uuid4())
            # Convert the move list to JSON string
            moves_json = json.dumps(game_data.get('moves_list', []))
            
            game = models.GameHistory(
                game_id=game_id,
                moves=moves_json,
                result=game_data['result'],
                white_player="AI",
                black_player="AI",
                game_type="self-play",
                evaluation=game_data.get('reward', 0)
            )
            db.session.add(game)
        
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'games_completed': total_games,
            'training_data': training_results,
            'summary': {
                'white_wins_percent': (white_wins / total_games) * 100 if total_games > 0 else 0,
                'black_wins_percent': (black_wins / total_games) * 100 if total_games > 0 else 0,
                'draws_percent': (draws / total_games) * 100 if total_games > 0 else 0
            }
        })
    except Exception as e:
        logging.error(f"Error in training: {e}")
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get-training-stats', methods=['GET'])
def get_training_stats():
    """Get statistics about the AI's training progress"""
    try:
        # Create a new session to isolate database operations
        with db.session.begin():
            # Get in-memory stats from the agent
            agent_stats = dqn_agent.get_training_stats()
            
            # Import models here to avoid circular imports
            import models
            
            # Get database stats
            db_stats = models.TrainingStats.query.order_by(models.TrainingStats.timestamp.desc()).first()
            
            if db_stats:
                # Add database stats if available
                combined_stats = {
                    'total_games': db_stats.total_games,
                    'white_wins': db_stats.white_wins,
                    'black_wins': db_stats.black_wins,
                    'draws': db_stats.draws,
                    'white_win_percentage': db_stats.white_win_percentage(),
                    'black_win_percentage': db_stats.black_win_percentage(),
                    'draw_percentage': db_stats.draw_percentage(),
                    'avg_game_length': db_stats.avg_game_length,
                    'avg_reward': db_stats.avg_reward,
                    'epsilon': db_stats.epsilon,
                    'alpha': db_stats.alpha,
                    'gamma': db_stats.gamma,
                    'positions_evaluated': db_stats.positions_evaluated,
                    'last_updated': db_stats.timestamp.isoformat(),
                    # Include agent's in-memory data
                    'last_game': agent_stats.get('last_game', []),
                    'training_history': agent_stats.get('training_history', [])
                }
            else:
                # Fall back to agent stats if no database records
                combined_stats = agent_stats
            
        # Get count of games in database
        game_count = models.GameHistory.query.count()
        if game_count:
            combined_stats['total_stored_games'] = game_count
        
        return jsonify({
            'status': 'success',
            'stats': combined_stats
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

@app.route('/api/save-game', methods=['POST'])
def save_game():
    """Save a completed game to the database for training"""
    data = request.get_json()
    
    try:
        # Import models here to avoid circular imports
        import models
        
        # Extract game data
        moves_list = data.get('moves', [])
        result = data.get('result', '1/2-1/2')  # Default to draw if not specified
        white_player = data.get('white_player', 'User')
        black_player = data.get('black_player', 'AI')
        final_position = data.get('final_position', '')
        evaluation = data.get('evaluation', 0)
        game_type = data.get('game_type', 'user-vs-ai')
        
        # Create a new game history record
        game = models.GameHistory(
            game_id=str(uuid.uuid4()),
            result=result,
            white_player=white_player,
            black_player=black_player,
            fen_position=final_position,
            game_type=game_type,
            evaluation=evaluation
        )
        
        # Set the moves list
        game.set_moves_list(moves_list)
        
        # Add to database and commit
        db.session.add(game)
        db.session.commit()
        
        # After saving, refresh the DQN agent's knowledge
        # Aggressively train on this new game immediately with True flag
        dqn_agent.load_games_from_database(aggressive_training=True)
        
        # Get updated stats for the UI
        agent_stats = dqn_agent.get_training_stats()
        
        return jsonify({
            'status': 'success',
            'message': 'Game saved and learned from successfully',
            'game_id': game.game_id,
            'stats': agent_stats 
        })
    except Exception as e:
        logging.error(f"Error saving game: {e}")
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

/**
 * DQN (Deep Q-Network) Agent Interface
 * Handles communication with the backend DQN agent
 */

class DQNInterface {
    constructor() {
        this.isThinking = false;
        this.latestEvaluation = 0;
        this.confidence = 0;
        this.networkState = null;
        this.moveHistory = [];
        this.trainingInProgress = false;
        this.trainingStats = null;
    }
    
    /**
     * Get the next move from the AI
     * @param {string} fen - The current board state in FEN notation
     * @returns {Promise<Object>} - The AI's chosen move and related data
     */
    async getMove(fen) {
        this.isThinking = true;
        
        try {
            const response = await fetch('/api/get-ai-move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ fen })
            });
            
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            this.isThinking = false;
            this.latestEvaluation = data.confidence;
            this.confidence = data.confidence;
            this.networkState = data.network_states;
            
            // Record the move
            this.moveHistory.push({
                fen: fen,
                move: data.move,
                evaluation: data.confidence
            });
            
            // Trigger visualization update
            this.updateVisualization();
            
            return data;
        } catch (error) {
            console.error('Error getting AI move:', error);
            this.isThinking = false;
            return null;
        }
    }
    
    /**
     * Evaluate the current board position
     * @param {string} fen - The current board state in FEN notation
     * @returns {Promise<Object>} - The evaluation data
     */
    async evaluatePosition(fen) {
        try {
            const response = await fetch('/api/evaluate-position', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ fen })
            });
            
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            this.latestEvaluation = data.evaluation;
            this.networkState = data.network_states;
            
            // Update visualization
            this.updateVisualization();
            
            return data;
        } catch (error) {
            console.error('Error evaluating position:', error);
            return null;
        }
    }
    
    /**
     * Check the current game state
     * @param {string} fen - The current board state in FEN notation
     * @returns {Promise<Object>} - The game state data
     */
    async checkGameState(fen) {
        try {
            const response = await fetch('/api/check-game-state', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ fen })
            });
            
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error checking game state:', error);
            return null;
        }
    }
    
    /**
     * Get legal moves for a piece
     * @param {string} fen - The current board state in FEN notation
     * @param {string} square - The square to check (e.g., "e4")
     * @returns {Promise<Array>} - Array of legal moves
     */
    async getLegalMoves(fen, square) {
        try {
            const response = await fetch('/api/get-legal-moves', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ fen, square })
            });
            
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            return data.moves;
        } catch (error) {
            console.error('Error getting legal moves:', error);
            return [];
        }
    }
    
    /**
     * Update the neural network visualization
     */
    updateVisualization() {
        // Only update if we have network state data
        if (this.networkState) {
            // Create an event to communicate with the visualization module
            const event = new CustomEvent('network-update', {
                detail: {
                    networkState: this.networkState,
                    evaluation: this.latestEvaluation,
                    confidence: this.confidence,
                    isThinking: this.isThinking
                }
            });
            
            document.dispatchEvent(event);
            
            // Update evaluation meter
            this.updateEvaluationMeter(this.latestEvaluation);
        }
    }
    
    /**
     * Update the evaluation meter display
     * @param {number} evaluation - The current position evaluation
     */
    updateEvaluationMeter(evaluation) {
        const evaluationMeter = document.getElementById('evaluation-meter');
        if (!evaluationMeter) return;
        
        const progressBar = evaluationMeter.querySelector('.progress-bar');
        if (!progressBar) return;
        
        // Map the evaluation to a 0-100 scale for the progress bar
        // Typical evaluation values might range from -5 (black winning) to +5 (white winning)
        // Map this to 0-100% with 50% being equal position
        const normalizedEval = Math.min(100, Math.max(0, (evaluation + 5) * 10));
        progressBar.style.width = `${normalizedEval}%`;
        
        // Change color based on evaluation
        if (normalizedEval < 40) {
            progressBar.className = 'progress-bar bg-danger'; // Black advantage
        } else if (normalizedEval > 60) {
            progressBar.className = 'progress-bar bg-success'; // White advantage
        } else {
            progressBar.className = 'progress-bar bg-primary'; // Roughly equal
        }
        
        // Update the displayed value text
        const evalText = document.querySelector('#evaluation-meter + div small:nth-child(2)');
        if (evalText) {
            evalText.textContent = `Even (${evaluation.toFixed(1)})`;
        }
    }
    
    /**
     * Start AI self-play training with specified parameters
     * @param {number} numGames - Number of games to play
     * @param {Object} parameters - Training parameters
     * @returns {Promise<Object>} - Training results
     */
    async startTraining(numGames, parameters = {}) {
        try {
            this.trainingInProgress = true;
            
            // First update parameters
            if (Object.keys(parameters).length > 0) {
                const paramsResponse = await fetch('/api/update-training-parameters', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(parameters)
                });
                
                if (!paramsResponse.ok) {
                    throw new Error(`Failed to update parameters: ${paramsResponse.status}`);
                }
            }
            
            // Start the training
            const trainingResponse = await fetch('/api/start-training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ num_games: numGames })
            });
            
            if (!trainingResponse.ok) {
                throw new Error(`Failed to start training: ${trainingResponse.status}`);
            }
            
            const data = await trainingResponse.json();
            this.trainingInProgress = false;
            return data;
        } catch (error) {
            console.error('Error starting training:', error);
            this.trainingInProgress = false;
            return { status: 'error', message: error.message };
        }
    }
    
    /**
     * Get current training statistics
     * @returns {Promise<Object>} - Training statistics
     */
    async getTrainingStats() {
        try {
            const response = await fetch('/api/get-training-stats');
            
            if (!response.ok) {
                throw new Error(`Failed to get training stats: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.trainingStats = data.stats;
                return data.stats;
            } else {
                throw new Error(data.message || 'Unknown error');
            }
        } catch (error) {
            console.error('Error getting training stats:', error);
            return null;
        }
    }
    
    /**
     * Reset the agent's state
     */
    reset() {
        this.isThinking = false;
        this.latestEvaluation = 0;
        this.confidence = 0;
        this.networkState = null;
        this.moveHistory = [];
        this.trainingInProgress = false;
        
        // Reset visualization
        this.updateVisualization();
    }
}

// Create a global instance of the DQN interface
const dqnAgent = new DQNInterface();

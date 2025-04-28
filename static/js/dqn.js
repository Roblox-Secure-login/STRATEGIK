/**
 * DQN (Deep Q-Network) Agent Interface
 * Handles communication with the backend DQN agent
 */
class DQNInterface {
    constructor() {
        this.networkVisualization = null;
        this.evaluationMeter = null;
    }
    
    /**
     * Get the next move from the AI
     * @param {string} fen - The current board state in FEN notation
     * @returns {Promise<Object>} - The AI's chosen move and related data
     */
    async getMove(fen) {
        try {
            const response = await fetch('/api/get-ai-move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ fen })
            });
            
            const data = await response.json();
            
            // Update visualization if available
            if (this.networkVisualization) {
                this.networkVisualization.updateVisualization(data.network_states);
            }
            
            // Update evaluation meter if available
            if (this.evaluationMeter) {
                this.updateEvaluationMeter(data.confidence);
            }
            
            return {
                move: data.move,
                confidence: data.confidence,
                networkStates: data.network_states
            };
        } catch (error) {
            console.error('Error getting AI move:', error);
            throw error;
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
            
            const data = await response.json();
            
            // Update visualization if available
            if (this.networkVisualization) {
                this.networkVisualization.updateVisualization(data.network_states);
            }
            
            // Update evaluation meter if available
            if (this.evaluationMeter) {
                this.updateEvaluationMeter(data.evaluation);
            }
            
            return {
                evaluation: data.evaluation,
                networkStates: data.network_states
            };
        } catch (error) {
            console.error('Error evaluating position:', error);
            throw error;
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
            
            return await response.json();
        } catch (error) {
            console.error('Error checking game state:', error);
            throw error;
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
            
            const data = await response.json();
            return data.moves;
        } catch (error) {
            console.error('Error getting legal moves:', error);
            throw error;
        }
    }
    
    /**
     * Start a training session for the DQN
     * @param {number} numGames - Number of games to play
     * @returns {Promise<Object>} - Training results
     */
    async startTraining(numGames = 10) {
        try {
            const response = await fetch('/api/start-training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ num_games: numGames })
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error starting training:', error);
            throw error;
        }
    }
    
    /**
     * Get current training statistics
     * @returns {Promise<Object>} - Training statistics
     */
    async getTrainingStats() {
        try {
            const response = await fetch('/api/get-training-stats');
            return await response.json();
        } catch (error) {
            console.error('Error getting training stats:', error);
            throw error;
        }
    }
    
    /**
     * Update training parameters
     * @param {Object} params - Training parameters to update
     * @returns {Promise<Object>} - Updated parameters
     */
    async updateTrainingParameters(params) {
        try {
            const response = await fetch('/api/update-training-parameters', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error updating training parameters:', error);
            throw error;
        }
    }
    
    /**
     * Set the network visualization component
     * @param {NetworkVisualization} visualization - The visualization component
     */
    setNetworkVisualization(visualization) {
        this.networkVisualization = visualization;
    }
    
    /**
     * Set the evaluation meter element
     * @param {HTMLElement} meterElement - The meter DOM element
     */
    setEvaluationMeter(meterElement) {
        this.evaluationMeter = meterElement;
    }
    
    /**
     * Update the evaluation meter display
     * @param {number} evaluation - The current position evaluation
     */
    updateEvaluationMeter(evaluation) {
        if (!this.evaluationMeter) return;
        
        // Normalize evaluation to a percentage (-100 to 100 -> 0 to 100%)
        const normalizedValue = Math.max(0, Math.min(100, (evaluation + 1) * 50));
        
        // Update the meter
        this.evaluationMeter.style.width = `${normalizedValue}%`;
        
        // Set color based on evaluation
        if (evaluation > 0.2) {
            this.evaluationMeter.className = 'progress-bar bg-success';
        } else if (evaluation < -0.2) {
            this.evaluationMeter.className = 'progress-bar bg-danger';
        } else {
            this.evaluationMeter.className = 'progress-bar bg-warning';
        }
    }
    
    /**
     * Reset the agent's state
     */
    reset() {
        // Reset visualization if available
        if (this.networkVisualization) {
            this.networkVisualization.drawEmptyNetwork();
        }
        
        // Reset evaluation meter if available
        if (this.evaluationMeter) {
            this.evaluationMeter.style.width = '50%';
            this.evaluationMeter.className = 'progress-bar bg-warning';
        }
    }
}
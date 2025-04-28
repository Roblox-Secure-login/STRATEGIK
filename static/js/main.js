/**
 * Main application script
 * Handles game initialization and AI interaction
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the chess board
    const board = createChessboard('chessboard', 400);
    
    // Initialize the DQN visualization
    const networkViz = new NetworkVisualization('network-visualization');
    
    // Get the buttons
    const newGameBtn = document.getElementById('new-game-btn');
    const undoMoveBtn = document.getElementById('undo-move-btn');
    const flipBoardBtn = document.getElementById('flip-board-btn');
    
    // Add button event listeners
    if (newGameBtn) {
        newGameBtn.addEventListener('click', newGame);
    }
    
    if (undoMoveBtn) {
        undoMoveBtn.addEventListener('click', undoMove);
    }
    
    if (flipBoardBtn) {
        flipBoardBtn.addEventListener('click', flipBoard);
    }
    
    // Game state
    let gameInProgress = true;
    let playerColor = 'w'; // 'w' for white, 'b' for black
    
    // Listen for moves from the player
    document.addEventListener('chess-move', handlePlayerMove);
    
    // Initialize a new game
    function newGame() {
        board.resetBoard();
        dqnAgent.reset();
        gameInProgress = true;
        playerColor = 'w'; // Player starts as white
        
        // Update game status
        updateGameStatus('Your turn to move (White)');
    }
    
    // Undo the last moves (both player and AI)
    function undoMove() {
        board.undoMove();
        dqnAgent.reset();
    }
    
    // Flip the board orientation
    function flipBoard() {
        board.flipBoard();
    }
    
    // Handle a move made by the player
    async function handlePlayerMove(event) {
        if (!gameInProgress) return;
        
        const { fen } = event.detail;
        
        // Check game state
        const gameState = await dqnAgent.checkGameState(fen);
        
        if (gameState && gameState.state !== 'ongoing') {
            // Game is over
            gameInProgress = false;
            updateGameStatus(gameState.message);
            
            // Save the completed game to the database for AI learning
            saveGameToDatabase(board.game, gameState.state);
            return;
        }
        
        // Evaluate the position
        await dqnAgent.evaluatePosition(fen);
        
        // If it's the AI's turn, get its move
        if (board.game.turn() !== playerColor) {
            setTimeout(async () => {
                // Get AI move
                const aiMoveData = await dqnAgent.getMove(fen);
                
                if (aiMoveData && aiMoveData.move) {
                    // Make the AI's move on the board
                    board.makeMove({
                        from: aiMoveData.move.substring(0, 2),
                        to: aiMoveData.move.substring(2, 4),
                        promotion: aiMoveData.move.length > 4 ? aiMoveData.move.substring(4, 5) : undefined
                    });
                    
                    // Check game state again after AI move
                    const newGameState = await dqnAgent.checkGameState(board.game.fen());
                    
                    if (newGameState && newGameState.state !== 'ongoing') {
                        // Game is over
                        gameInProgress = false;
                        updateGameStatus(newGameState.message);
                        
                        // Save the completed game to the database for AI learning
                        saveGameToDatabase(board.game, newGameState.state);
                    } else {
                        // Continue game
                        updateGameStatus('Your turn to move');
                    }
                }
            }, 500); // Small delay to make AI "thinking" visible
        }
    }
    
    // Update the game status display
    function updateGameStatus(message) {
        const statusElement = document.getElementById('game-status');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }
    
    // Initialize with a new game
    newGame();
    
    // Add window resize handler for responsive board
    window.addEventListener('resize', function() {
        const container = document.getElementById('chessboard-container');
        if (container) {
            // Calculate the maximum available width with a padding
            const containerWidth = container.clientWidth;
            const isMobile = window.innerWidth < 576;
            
            // On mobile, use nearly full width; on desktop, cap at 400px
            const width = isMobile ? 
                Math.min(320, containerWidth - 10) : // Mobile (smaller padding)
                Math.min(400, containerWidth - 20);  // Desktop
            
            board.size = width;
            board.squareSize = width / 8;
            board.renderBoard();
            
            // On very small screens, ensure the container is centered
            if (isMobile) {
                container.style.display = 'flex';
                container.style.justifyContent = 'center';
            }
        }
    });
    
    // Trigger initial resize
    window.dispatchEvent(new Event('resize'));
    
    /**
     * Save a completed game to the database for AI training
     * @param {Chess} chessGame - The chess.js game object
     * @param {string} gameState - The game state (checkmate, stalemate, etc.)
     */
    async function saveGameToDatabase(chessGame, gameState) {
        try {
            // Get all moves in UCI format
            const history = chessGame.history({verbose: true});
            const movesList = history.map(move => {
                return move.from + move.to + (move.promotion || '');
            });
            
            // Determine the game result
            let result = '1/2-1/2'; // Default to draw
            if (gameState === 'checkmate') {
                // Winner is the opposite of the current turn (since current turn lost)
                result = chessGame.turn() === 'w' ? '0-1' : '1-0';
            }
            
            // Get the final position evaluation
            let evaluation = 0;
            try {
                const evalData = await dqnAgent.evaluatePosition(chessGame.fen());
                evaluation = evalData.evaluation;
            } catch (e) {
                console.error('Error getting evaluation:', e);
            }
            
            // Prepare game data
            const gameData = {
                moves: movesList,
                result: result,
                white_player: playerColor === 'w' ? 'User' : 'AI',
                black_player: playerColor === 'w' ? 'AI' : 'User',
                final_position: chessGame.fen(),
                evaluation: evaluation,
                game_type: 'user-vs-ai'
            };
            
            // Send to server
            const response = await fetch('/api/save-game', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(gameData)
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('Game saved successfully for AI training:', data.game_id);
            } else {
                console.error('Error saving game:', data.message);
            }
        } catch (error) {
            console.error('Error saving game to database:', error);
        }
    }
});

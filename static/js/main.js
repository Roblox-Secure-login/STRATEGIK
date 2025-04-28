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
            const width = Math.min(400, container.clientWidth - 20);
            board.size = width;
            board.squareSize = width / 8;
            board.renderBoard();
        }
    });
    
    // Trigger initial resize
    window.dispatchEvent(new Event('resize'));
});

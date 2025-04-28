/**
 * Interactive Chess Board Implementation
 * Handles rendering and user interactions with the chess board
 */

class ChessBoard {
    constructor(containerId, size = 400, flipped = false) {
        this.containerId = containerId;
        this.size = size;
        this.flipped = flipped;
        this.squareSize = size / 8;
        this.selectedPiece = null;
        this.game = new Chess(); // Initialize with chess.js
        this.highlightedSquares = [];
        this.moveHistory = [];
        this.pieceImages = {
            'wP': 'P', 'wN': 'N', 'wB': 'B', 'wR': 'R', 'wQ': 'Q', 'wK': 'K',
            'bP': 'p', 'bN': 'n', 'bB': 'b', 'bR': 'r', 'bQ': 'q', 'bK': 'k'
        };
        
        this.initBoard();
        this.renderBoard();
        this.setupEventListeners();
    }
    
    /**
     * Initialize the chess board container
     */
    initBoard() {
        const container = document.getElementById(this.containerId);
        container.innerHTML = '';
        container.style.width = `${this.size}px`;
        container.style.height = `${this.size}px`;
        container.style.position = 'relative';
        container.style.userSelect = 'none';
        
        this.boardElement = container;
    }
    
    /**
     * Render the chess board with pieces
     */
    renderBoard() {
        this.boardElement.innerHTML = '';
        
        // Create squares and pieces
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                // Calculate actual row and column based on flipped state
                const actualRow = this.flipped ? 7 - row : row;
                const actualCol = this.flipped ? 7 - col : col;
                
                // Create square
                const square = document.createElement('div');
                square.classList.add('chess-square');
                
                // Alternating colors for chess board
                if ((row + col) % 2 === 0) {
                    square.classList.add('white-square');
                } else {
                    square.classList.add('black-square');
                }
                
                // Position square
                square.style.position = 'absolute';
                square.style.width = `${this.squareSize}px`;
                square.style.height = `${this.squareSize}px`;
                square.style.top = `${row * this.squareSize}px`;
                square.style.left = `${col * this.squareSize}px`;
                
                // Add square identifier attribute
                const squareId = this.getSquareId(actualRow, actualCol);
                square.setAttribute('data-square', squareId);
                
                // Add square to board
                this.boardElement.appendChild(square);
                
                // Add piece if there is one on this square
                const piece = this.game.get(squareId);
                if (piece) {
                    this.addPiece(square, piece, squareId);
                }
                
                // Add highlight if this square is highlighted
                if (this.highlightedSquares.includes(squareId)) {
                    square.classList.add('highlight-square');
                }
            }
        }
    }
    
    /**
     * Add a chess piece to a square
     */
    addPiece(square, piece, squareId) {
        const pieceElement = document.createElement('div');
        pieceElement.classList.add('chess-piece');
        
        // Set piece color and type
        const pieceType = (piece.color === 'w' ? 'w' : 'b') + piece.type.toUpperCase();
        pieceElement.setAttribute('data-piece', pieceType);
        
        // Use SVG representation for the piece
        const pieceChar = this.pieceImages[pieceType];
        pieceElement.innerHTML = this.getPieceSVG(pieceChar);
        
        // Make piece draggable if it's the current player's turn
        if ((piece.color === 'w' && this.game.turn() === 'w') || 
            (piece.color === 'b' && this.game.turn() === 'b')) {
            pieceElement.classList.add('draggable-piece');
            pieceElement.setAttribute('draggable', 'true');
            pieceElement.addEventListener('dragstart', (e) => this.handleDragStart(e, squareId));
        }
        
        // Add piece to square
        square.appendChild(pieceElement);
    }
    
    /**
     * Set up event listeners for drag and drop, clicks
     */
    setupEventListeners() {
        // Delegate events to the board container
        this.boardElement.addEventListener('click', (e) => this.handleSquareClick(e));
        this.boardElement.addEventListener('dragover', (e) => e.preventDefault());
        this.boardElement.addEventListener('drop', (e) => this.handleDrop(e));
    }
    
    /**
     * Handle piece selection and highlighting available moves
     */
    handleSquareClick(e) {
        const square = e.target.closest('.chess-square');
        if (!square) return;
        
        const squareId = square.getAttribute('data-square');
        const piece = this.game.get(squareId);
        
        // If we already have a selected piece, try to move it
        if (this.selectedPiece) {
            // Try to make a move
            if (this.selectedPiece !== squareId) {
                const move = {
                    from: this.selectedPiece,
                    to: squareId,
                    promotion: 'q' // Always promote to queen for simplicity
                };
                
                this.makeMove(move);
            }
            
            // Clear selection and highlights regardless
            this.clearHighlights();
            this.selectedPiece = null;
            return;
        }
        
        // If clicked on a piece that can move
        if (piece && 
            ((piece.color === 'w' && this.game.turn() === 'w') || 
             (piece.color === 'b' && this.game.turn() === 'b'))) {
            this.selectedPiece = squareId;
            this.highlightLegalMoves(squareId);
        }
    }
    
    /**
     * Handle drag start for a piece
     */
    handleDragStart(e, squareId) {
        this.selectedPiece = squareId;
        this.highlightLegalMoves(squareId);
        
        // Set data for drag operation
        e.dataTransfer.setData('text/plain', squareId);
        e.dataTransfer.effectAllowed = 'move';
        
        // For better drag image (optional)
        const img = new Image();
        img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'; // 1px transparent GIF
        e.dataTransfer.setDragImage(img, 0, 0);
    }
    
    /**
     * Handle dropping a piece
     */
    handleDrop(e) {
        e.preventDefault();
        
        const targetSquare = e.target.closest('.chess-square');
        if (!targetSquare) return;
        
        const targetSquareId = targetSquare.getAttribute('data-square');
        
        // Try to make the move
        if (this.selectedPiece && this.selectedPiece !== targetSquareId) {
            const move = {
                from: this.selectedPiece,
                to: targetSquareId,
                promotion: 'q' // Always promote to queen for simplicity
            };
            
            this.makeMove(move);
        }
        
        // Clear selection and highlights
        this.clearHighlights();
        this.selectedPiece = null;
    }
    
    /**
     * Make a move on the board
     */
    makeMove(moveObj) {
        try {
            // Try to make the move using chess.js
            const move = this.game.move(moveObj);
            
            if (move) {
                // Successful move
                this.moveHistory.push(move);
                this.renderBoard();
                
                // Update game status
                this.updateGameStatus();
                
                // Trigger move event for AI response
                const event = new CustomEvent('chess-move', { 
                    detail: { 
                        move: move,
                        fen: this.game.fen(),
                        pgn: this.game.pgn()
                    } 
                });
                document.dispatchEvent(event);
            }
        } catch (e) {
            console.error('Invalid move:', e);
        }
    }
    
    /**
     * Update the game status display
     */
    updateGameStatus() {
        const statusElement = document.getElementById('game-status');
        if (!statusElement) return;
        
        let statusText = '';
        
        // Check for checkmate
        if (this.game.in_checkmate()) {
            statusText = `Checkmate! ${this.game.turn() === 'w' ? 'Black' : 'White'} wins.`;
            statusElement.className = 'alert alert-success text-center mb-3';
        }
        // Check for draw
        else if (this.game.in_draw()) {
            statusText = 'Draw!';
            if (this.game.in_stalemate()) {
                statusText += ' (Stalemate)';
            } else if (this.game.in_threefold_repetition()) {
                statusText += ' (Threefold Repetition)';
            } else if (this.game.insufficient_material()) {
                statusText += ' (Insufficient Material)';
            }
            statusElement.className = 'alert alert-info text-center mb-3';
        }
        // Check for check
        else if (this.game.in_check()) {
            statusText = `${this.game.turn() === 'w' ? 'White' : 'Black'} is in check`;
            statusElement.className = 'alert alert-warning text-center mb-3';
        }
        // Normal status
        else {
            statusText = `${this.game.turn() === 'w' ? 'White' : 'Black'} to move`;
            statusElement.className = 'alert alert-info text-center mb-3';
        }
        
        statusElement.textContent = statusText;
    }
    
    /**
     * Highlight legal moves for a piece
     */
    highlightLegalMoves(squareId) {
        this.clearHighlights();
        
        // Get legal moves for the piece
        const legalMoves = this.game.moves({
            square: squareId,
            verbose: true
        });
        
        // Highlight source square
        this.highlightedSquares.push(squareId);
        
        // Highlight target squares
        legalMoves.forEach(move => {
            this.highlightedSquares.push(move.to);
        });
        
        // Apply highlights
        this.highlightedSquares.forEach(square => {
            const squareElement = this.boardElement.querySelector(`[data-square="${square}"]`);
            if (squareElement) {
                squareElement.classList.add('highlight-square');
                
                // Add move indicator dot if it's a destination square
                if (square !== squareId) {
                    const moveIndicator = document.createElement('div');
                    moveIndicator.classList.add('highlight-move');
                    squareElement.appendChild(moveIndicator);
                }
            }
        });
    }
    
    /**
     * Clear all highlights from the board
     */
    clearHighlights() {
        this.highlightedSquares.forEach(square => {
            const squareElement = this.boardElement.querySelector(`[data-square="${square}"]`);
            if (squareElement) {
                squareElement.classList.remove('highlight-square');
                
                // Remove move indicator dots
                const moveIndicator = squareElement.querySelector('.highlight-move');
                if (moveIndicator) {
                    moveIndicator.remove();
                }
            }
        });
        
        this.highlightedSquares = [];
    }
    
    /**
     * Get the algebraic notation for a square
     */
    getSquareId(row, col) {
        const files = 'abcdefgh';
        const ranks = '87654321';
        return files.charAt(col) + ranks.charAt(row);
    }
    
    /**
     * Flip the board orientation
     */
    flipBoard() {
        this.flipped = !this.flipped;
        this.renderBoard();
    }
    
    /**
     * Reset the board to starting position
     */
    resetBoard() {
        this.game.reset();
        this.selectedPiece = null;
        this.highlightedSquares = [];
        this.moveHistory = [];
        this.renderBoard();
        this.updateGameStatus();
    }
    
    /**
     * Undo the last move
     */
    undoMove() {
        this.game.undo();
        if (this.game.turn() !== 'w') {
            // If it's black's turn after undo, undo one more time to get back to white's turn
            this.game.undo();
        }
        this.selectedPiece = null;
        this.highlightedSquares = [];
        this.renderBoard();
        this.updateGameStatus();
    }
    
    /**
     * Set a position from FEN notation
     */
    setPosition(fen) {
        try {
            this.game.load(fen);
            this.selectedPiece = null;
            this.highlightedSquares = [];
            this.renderBoard();
            this.updateGameStatus();
        } catch (e) {
            console.error('Invalid FEN:', e);
        }
    }
    
    /**
     * Get SVG representation for chess pieces
     */
    getPieceSVG(piece) {
        // SVG definitions for chess pieces
        const pieceSVG = {
            'K': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6M20 8h5" stroke-linejoin="miter"></path><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#fff" stroke-linecap="butt" stroke-linejoin="miter"></path><path d="M12.5 37c5.5 3.5 14.5 3.5 20 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-2.5-7.5-12-10.5-16-4-3 6 6 10.5 6 10.5v7" fill="#fff"></path><path d="M12.5 30c5.5-3 14.5-3 20 0m-20 3.5c5.5-3 14.5-3 20 0m-20 3.5c5.5-3 14.5-3 20 0"></path></g></svg>`,
            'Q': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 26c8.5-1.5 21-1.5 27 0l2-12-7 11V11l-5.5 13.5-3-15-3 15-5.5-14V25L7 14l2 12z"></path><path d="M9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z"></path><path d="M11.5 30c3.5-1 18.5-1 22 0M12 33.5c6-1 15-1 21 0" fill="none"></path></g></svg>`,
            'R': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 39h27v-3H9v3zm3-3v-4h21v4H12zm-1-22V9h4v2h5V9h5v2h5V9h4v5" stroke-linecap="butt"></path><path d="M34 14l-3 3H14l-3-3"></path><path d="M31 17v12.5H14V17" stroke-linecap="butt" stroke-linejoin="miter"></path><path d="M31 29.5l1.5 2.5h-20l1.5-2.5"></path><path d="M11 14h23" fill="none" stroke-linejoin="miter"></path></g></svg>`,
            'B': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><g fill="#fff" stroke-linecap="butt"><path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.35.49-2.32.47-3-.5 1.35-1.46 3-2 3-2z"></path><path d="M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"></path><path d="M25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"></path></g><path d="M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5" stroke-linejoin="miter"></path></g></svg>`,
            'N': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#fff"></path><path d="M24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.042-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4.003 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-.994-.5-2-.5-3 1-1 3 2.5 3 2.5h2s.78-1.992 2.5-3c1 0 1 3 1 3" fill="#fff"></path><path d="M9.5 25.5a.5.5 0 1 1-1 0 .5.5 0 1 1 1 0zm5.433-9.75a.5 1.5 30 1 1-.866-.5.5 1.5 30 1 1 .866.5z" fill="#000"></path></g></svg>`,
            'P': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><path d="M22 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38-1.95 1.12-3.28 3.21-3.28 5.62 0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#fff" stroke="#000" stroke-width="1.5" stroke-linecap="round"></path></svg>`,
            'k': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6" stroke-linejoin="miter"></path><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#000" stroke-linecap="butt" stroke-linejoin="miter"></path><path d="M12.5 37c5.5 3.5 14.5 3.5 20 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-2.5-7.5-12-10.5-16-4-3 6 6 10.5 6 10.5v7" fill="#000"></path><path d="M20 8h5" stroke-linejoin="miter"></path><path d="M32 29.5s8.5-4 6.03-9.65C34.15 14 25 18 22.5 24.5l.01 2.1-.01-2.1C20 18 9.906 14 6.997 19.85c-2.497 5.65 4.853 9 4.853 9M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0" stroke="#fff"></path></g></svg>`,
            'q': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="#000" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><g stroke="none"><circle cx="6" cy="12" r="2.75"></circle><circle cx="14" cy="9" r="2.75"></circle><circle cx="22.5" cy="8" r="2.75"></circle><circle cx="31" cy="9" r="2.75"></circle><circle cx="39" cy="12" r="2.75"></circle></g><path d="M9 26c8.5-1.5 21-1.5 27 0l2.5-12.5L31 25l-.3-14.1-5.2 13.6-3-14.5-3 14.5-5.2-13.6L14 25 6.5 13.5 9 26z" stroke-linecap="butt"></path><path d="M9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z" stroke-linecap="butt"></path><path d="M11.5 30c3.5-1 18.5-1 22 0M12 33.5c6-1 15-1 21 0" fill="none" stroke="#fff"></path></g></svg>`,
            'r': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 39h27v-3H9v3zm3.5-7l1.5-2.5h17l1.5 2.5h-20zm-.5 4v-4h21v4H12z" stroke-linecap="butt" fill="#000"></path><path d="M14 29.5v-13h17v13H14z" stroke-linecap="butt" stroke-linejoin="miter" fill="#000"></path><path d="M14 16.5L11 14h23l-3 2.5H14zM11 14V9h4v2h5V9h5v2h5V9h4v5H11z" stroke-linecap="butt" fill="#000"></path><path d="M12 35.5h21M13 31.5h19M14 29.5h17M14 16.5h17M11 14h23" fill="none" stroke="#fff" stroke-width="1" stroke-linejoin="miter"></path></g></svg>`,
            'b': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><g fill="#000" stroke-linecap="butt"><path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.35.49-2.32.47-3-.5 1.35-1.46 3-2 3-2z"></path><path d="M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"></path><path d="M25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"></path></g><path d="M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5" stroke="#fff" stroke-linejoin="miter"></path></g></svg>`,
            'n': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#000"></path><path d="M24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.042-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4.003 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-.994-.5-2-.5-3 1-1 3 2.5 3 2.5h2s.78-1.992 2.5-3c1 0 1 3 1 3" fill="#000"></path><path d="M9.5 25.5a.5.5 0 1 1-1 0 .5.5 0 1 1 1 0zm5.433-9.75a.5 1.5 30 1 1-.866-.5.5 1.5 30 1 1 .866.5z" fill="#fff" stroke="#fff"></path><path d="M24.55 10.4l-.45 1.45.5.15c3.15 1 5.65 2.49 7.9 6.75S35.75 29.06 35.25 39l-.05.5h2.25l.05-.5c.5-10.06-.88-16.85-3.25-21.34-2.37-4.49-5.79-6.64-9.19-7.16l-.51-.1z" fill="#fff" stroke="none"></path></g></svg>`,
            'p': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><path d="M22 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38-1.95 1.12-3.28 3.21-3.28 5.62 0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#000" stroke="#000" stroke-width="1.5" stroke-linecap="round"></path></svg>`
        };
        
        return pieceSVG[piece] || '';
    }
}

/**
 * Create and return a ChessBoard instance
 * @param {string} containerId - The ID of the container element
 * @param {number} size - The size of the board in pixels
 * @param {boolean} flipped - Whether the board should be flipped
 * @returns {ChessBoard} A new ChessBoard instance
 */
function createChessboard(containerId, size = 400, flipped = false) {
    return new ChessBoard(containerId, size, flipped);
}

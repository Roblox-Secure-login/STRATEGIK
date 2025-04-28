/**
 * Neural Network Visualization for DQN Chess AI
 * Visualizes the neural network's state and evaluation process
 */

class NetworkVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container element with ID "${containerId}" not found.`);
            return;
        }
        
        // Initialize properties
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight || 200;
        this.layerCount = 4;
        this.maxNeuronsPerLayer = 10;
        
        // Initialize the visualization
        this.initVisualization();
        
        // Listen for network updates
        document.addEventListener('network-update', (e) => this.updateVisualization(e.detail));
    }
    
    /**
     * Initialize the visualization SVG
     */
    initVisualization() {
        // Clear existing content
        this.container.innerHTML = '';
        
        // Create SVG container
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.setAttribute('viewBox', `0 0 ${this.width} ${this.height}`);
        this.container.appendChild(svg);
        
        // Store reference to SVG
        this.svg = svg;
        
        // Create layers group
        this.layersGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.svg.appendChild(this.layersGroup);
        
        // Create connections group (drawn first to appear behind neurons)
        this.connectionsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.svg.insertBefore(this.connectionsGroup, this.layersGroup);
        
        // Create text group (drawn last to appear on top)
        this.textGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.svg.appendChild(this.textGroup);
        
        // Draw initial network structure
        this.drawEmptyNetwork();
    }
    
    /**
     * Draw an empty network structure
     */
    drawEmptyNetwork() {
        // Define layers
        const layerNames = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
        const neuronCounts = [8, 6, 4, 1];
        
        // Store neuron coordinates for connections
        const neuronPositions = [];
        
        // Calculate layer spacing
        const layerSpacing = this.width / (this.layerCount + 1);
        
        // Draw each layer
        for (let l = 0; l < this.layerCount; l++) {
            const layerX = (l + 1) * layerSpacing;
            const neuronsInLayer = neuronCounts[l];
            const neuronPositionsInLayer = [];
            
            // Draw layer label
            const layerLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            layerLabel.setAttribute('x', layerX);
            layerLabel.setAttribute('y', 15);
            layerLabel.setAttribute('text-anchor', 'middle');
            layerLabel.setAttribute('font-size', '12px');
            layerLabel.textContent = layerNames[l];
            this.textGroup.appendChild(layerLabel);
            
            // Draw neurons
            const neuronSpacing = this.height / (neuronsInLayer + 1);
            
            for (let n = 0; n < neuronsInLayer; n++) {
                const neuronY = (n + 1) * neuronSpacing;
                
                // Draw neuron
                const neuron = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                neuron.setAttribute('cx', layerX);
                neuron.setAttribute('cy', neuronY);
                neuron.setAttribute('r', 6);
                neuron.setAttribute('fill', '#ccc');
                neuron.setAttribute('class', `layer-${l} neuron-${n}`);
                this.layersGroup.appendChild(neuron);
                
                // Store position
                neuronPositionsInLayer.push({ x: layerX, y: neuronY });
            }
            
            neuronPositions.push(neuronPositionsInLayer);
        }
        
        // Draw connections between layers
        for (let l = 0; l < neuronPositions.length - 1; l++) {
            const fromLayer = neuronPositions[l];
            const toLayer = neuronPositions[l + 1];
            
            for (let from = 0; from < fromLayer.length; from++) {
                for (let to = 0; to < toLayer.length; to++) {
                    const connection = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    connection.setAttribute('x1', fromLayer[from].x);
                    connection.setAttribute('y1', fromLayer[from].y);
                    connection.setAttribute('x2', toLayer[to].x);
                    connection.setAttribute('y2', toLayer[to].y);
                    connection.setAttribute('stroke', '#ddd');
                    connection.setAttribute('stroke-width', '1');
                    connection.setAttribute('class', `connection from-${l}-${from} to-${l+1}-${to}`);
                    this.connectionsGroup.appendChild(connection);
                }
            }
        }
    }
    
    /**
     * Update the visualization with new network state data
     * @param {Object} data - Network state and evaluation data
     */
    updateVisualization(data) {
        if (!data || !data.networkState) return;
        
        const { networkState, evaluation, isThinking } = data;
        
        // Update neuron colors based on activation values
        for (let l = 0; l < networkState.length; l++) {
            const layer = networkState[l];
            
            if (!layer || !layer.neurons) continue;
            
            // Update neurons in this layer
            for (let n = 0; n < layer.neurons.length; n++) {
                const neuron = layer.neurons[n];
                
                // Find the neuron element
                const neuronElement = this.svg.querySelector(`.layer-${l}.neuron-${n}`);
                if (neuronElement) {
                    // Calculate color based on activation (red for negative, green for positive)
                    const value = neuron.activation;
                    let color;
                    
                    if (value >= 0) {
                        // Green intensity based on value (0-1)
                        const intensity = Math.min(255, Math.round(value * 255));
                        color = `rgb(0, ${intensity}, 0)`;
                    } else {
                        // Red intensity based on absolute value (0-1)
                        const intensity = Math.min(255, Math.round(Math.abs(value) * 255));
                        color = `rgb(${intensity}, 0, 0)`;
                    }
                    
                    // Update neuron color
                    neuronElement.setAttribute('fill', color);
                    
                    // Add pulsing animation if AI is thinking
                    if (isThinking) {
                        neuronElement.setAttribute('opacity', '0.8');
                        neuronElement.innerHTML = `
                            <animate attributeName="opacity" 
                                values="0.5;1;0.5" 
                                dur="1s" 
                                repeatCount="indefinite" />
                        `;
                    } else {
                        neuronElement.removeAttribute('opacity');
                        neuronElement.innerHTML = '';
                    }
                }
            }
            
            // Update connections to the next layer
            if (l < networkState.length - 1) {
                for (let from = 0; from < layer.neurons.length; from++) {
                    const fromNeuron = layer.neurons[from];
                    
                    for (let to = 0; to < Math.min(10, networkState[l+1].neurons.length); to++) {
                        // Find the connection element
                        const connectionElement = this.svg.querySelector(`.connection.from-${l}-${from}.to-${l+1}-${to}`);
                        
                        if (connectionElement) {
                            // Calculate weight (use random value for demo)
                            // In a real implementation, you would use the actual weight
                            const weight = fromNeuron.weight || Math.random() * 2 - 1;
                            
                            // Calculate color and thickness based on weight
                            let color;
                            let width;
                            
                            if (weight >= 0) {
                                // Blue for positive weights
                                const intensity = Math.min(255, Math.round(weight * 200));
                                color = `rgba(0, 0, ${intensity}, 0.5)`;
                                width = Math.max(0.5, Math.min(3, weight * 3));
                            } else {
                                // Red for negative weights
                                const intensity = Math.min(255, Math.round(Math.abs(weight) * 200));
                                color = `rgba(${intensity}, 0, 0, 0.5)`;
                                width = Math.max(0.5, Math.min(3, Math.abs(weight) * 3));
                            }
                            
                            // Update connection
                            connectionElement.setAttribute('stroke', color);
                            connectionElement.setAttribute('stroke-width', width);
                        }
                    }
                }
            }
        }
        
        // Add a pulse effect on the evaluation display if thinking
        const evaluationMeter = document.getElementById('evaluation-meter');
        if (evaluationMeter) {
            if (isThinking) {
                evaluationMeter.classList.add('thinking');
            } else {
                evaluationMeter.classList.remove('thinking');
            }
        }
    }
    
    /**
     * Resize the visualization
     */
    resize() {
        if (!this.container) return;
        
        // Update width and height
        this.width = this.container.clientWidth;
        
        // Reinitialize visualization with new dimensions
        this.initVisualization();
    }
}

// Initialize the network visualization when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const networkViz = new NetworkVisualization('network-visualization');
    
    // Handle window resize
    window.addEventListener('resize', () => {
        networkViz.resize();
    });
});

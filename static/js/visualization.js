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
        
        this.initVisualization();
        this.drawEmptyNetwork();
        
        // Add window resize listener
        window.addEventListener('resize', this.resize.bind(this));
    }
    
    /**
     * Initialize the visualization SVG
     */
    initVisualization() {
        // Set dimensions
        this.width = this.container.clientWidth;
        this.height = 300;
        this.margin = { top: 20, right: 20, bottom: 20, left: 20 };
        
        // Create SVG container
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);
        
        // Create main group with margins
        this.g = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
            
        // Add color legend
        this.addColorLegend();
    }
    
    /**
     * Add a color legend for neuron activations
     */
    addColorLegend() {
        const legendWidth = 150;
        const legendHeight = 15;
        const legendX = this.width - this.margin.right - legendWidth;
        const legendY = this.height - this.margin.bottom - 30;
        
        // Create gradient
        const defs = this.svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'activation-gradient')
            .attr('x1', '0%')
            .attr('y1', '0%')
            .attr('x2', '100%')
            .attr('y2', '0%');
            
        gradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', d3.interpolateRdYlGn(0));
            
        gradient.append('stop')
            .attr('offset', '50%')
            .attr('stop-color', d3.interpolateRdYlGn(0.5));
            
        gradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', d3.interpolateRdYlGn(1));
            
        // Draw legend rectangle
        this.svg.append('rect')
            .attr('x', legendX)
            .attr('y', legendY)
            .attr('width', legendWidth)
            .attr('height', legendHeight)
            .style('fill', 'url(#activation-gradient)');
            
        // Add labels
        this.svg.append('text')
            .attr('x', legendX)
            .attr('y', legendY - 5)
            .attr('text-anchor', 'start')
            .attr('font-size', '10px')
            .text('Low Activation');
            
        this.svg.append('text')
            .attr('x', legendX + legendWidth)
            .attr('y', legendY - 5)
            .attr('text-anchor', 'end')
            .attr('font-size', '10px')
            .text('High Activation');
            
        this.svg.append('text')
            .attr('x', legendX + legendWidth/2)
            .attr('y', legendY + legendHeight + 15)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .text('Neuron Activation Level');
    }
    
    /**
     * Draw an empty network structure
     */
    drawEmptyNetwork() {
        // Clear any existing elements
        this.g.selectAll('*').remove();
        
        // Default network structure (will be updated with real data)
        this.networkStructure = [
            { name: 'input', neurons: 10 },
            { name: 'hidden1', neurons: 8 },
            { name: 'hidden2', neurons: 6 },
            { name: 'output', neurons: 1 }
        ];
        
        const availableWidth = this.width - this.margin.left - this.margin.right;
        const availableHeight = this.height - this.margin.top - this.margin.bottom - 50; // Leave space for legend
        
        // Calculate x-position of each layer
        const layerGap = availableWidth / (this.networkStructure.length + 1);
        
        // For each layer
        this.networkStructure.forEach((layer, layerIndex) => {
            const x = (layerIndex + 1) * layerGap;
            const neuronGap = availableHeight / (layer.neurons + 1);
            
            // Draw layer label
            this.g.append('text')
                .attr('x', x)
                .attr('y', 10)
                .attr('text-anchor', 'middle')
                .attr('font-size', '12px')
                .text(layer.name);
            
            // For each neuron in the layer
            for (let neuronIndex = 0; neuronIndex < layer.neurons; neuronIndex++) {
                const y = (neuronIndex + 1) * neuronGap;
                
                // Draw neuron
                this.g.append('circle')
                    .attr('cx', x)
                    .attr('cy', y)
                    .attr('r', 8)
                    .attr('fill', '#e0e0e0')
                    .attr('stroke', '#000')
                    .attr('stroke-width', 1)
                    .attr('data-layer', layer.name)
                    .attr('data-neuron', neuronIndex);
                
                // If not first layer, draw connections to previous layer
                if (layerIndex > 0) {
                    const prevLayer = this.networkStructure[layerIndex - 1];
                    const prevX = (layerIndex) * layerGap;
                    const prevNeuronGap = availableHeight / (prevLayer.neurons + 1);
                    
                    for (let prevNeuronIndex = 0; prevNeuronIndex < prevLayer.neurons; prevNeuronIndex++) {
                        const prevY = (prevNeuronIndex + 1) * prevNeuronGap;
                        
                        // Draw connection
                        this.g.append('line')
                            .attr('x1', prevX)
                            .attr('y1', prevY)
                            .attr('x2', x)
                            .attr('y2', y)
                            .attr('stroke', '#ccc')
                            .attr('stroke-width', 1)
                            .attr('data-from-layer', prevLayer.name)
                            .attr('data-from-neuron', prevNeuronIndex)
                            .attr('data-to-layer', layer.name)
                            .attr('data-to-neuron', neuronIndex);
                    }
                }
            }
        });
    }
    
    /**
     * Update the visualization with new network state data
     * @param {Array} data - Network state and evaluation data from the DQN
     */
    updateVisualization(data) {
        if (!data || !Array.isArray(data)) {
            console.error('Invalid network state data:', data);
            return;
        }
        
        // Update network structure if it's different
        if (this.networkStructure.length !== data.length) {
            this.networkStructure = data.map(layer => ({
                name: layer.layer,
                neurons: layer.neurons.length
            }));
            this.drawEmptyNetwork();
        }
        
        const availableWidth = this.width - this.margin.left - this.margin.right;
        const availableHeight = this.height - this.margin.top - this.margin.bottom - 50;
        
        // Calculate x-position of each layer
        const layerGap = availableWidth / (this.networkStructure.length + 1);
        
        // For each layer
        data.forEach((layerData, layerIndex) => {
            const layer = this.networkStructure[layerIndex];
            const x = (layerIndex + 1) * layerGap;
            const neuronGap = availableHeight / (layer.neurons + 1);
            
            // For each neuron
            layerData.neurons.forEach((neuron, neuronIndex) => {
                if (neuronIndex >= layer.neurons) return; // Skip if outside our visualization
                
                const y = (neuronIndex + 1) * neuronGap;
                const activation = neuron.activation;
                
                // Update neuron color based on activation
                this.g.select(`circle[data-layer="${layer.name}"][data-neuron="${neuronIndex}"]`)
                    .transition()
                    .duration(500)
                    .attr('fill', d3.interpolateRdYlGn(activation));
                
                // If not first layer, update connection weights
                if (layerIndex > 0) {
                    const prevLayer = this.networkStructure[layerIndex - 1];
                    const prevLayerData = data[layerIndex - 1];
                    
                    prevLayerData.neurons.forEach((prevNeuron, prevNeuronIndex) => {
                        if (prevNeuronIndex >= prevLayer.neurons) return;
                        
                        // Calculate weight color and thickness (would come from the API in a real implementation)
                        const weight = (prevNeuron.weight + 1) / 2; // Normalize from [-1,1] to [0,1]
                        
                        // Find the connection
                        this.g.select(`line[data-from-layer="${prevLayer.name}"][data-from-neuron="${prevNeuronIndex}"][data-to-layer="${layer.name}"][data-to-neuron="${neuronIndex}"]`)
                            .transition()
                            .duration(500)
                            .attr('stroke', d3.interpolateRdYlGn(weight))
                            .attr('stroke-width', 1 + Math.abs(prevNeuron.weight) * 2);
                    });
                }
            });
        });
    }
    
    /**
     * Resize the visualization
     */
    resize() {
        this.width = this.container.clientWidth;
        this.svg.attr('width', this.width);
        this.drawEmptyNetwork();
    }
}
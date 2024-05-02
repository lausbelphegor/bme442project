from graphviz import Digraph

def visualize_network():
    dot = Digraph(comment='EEG Conformer Architecture')

    # Adding nodes
    dot.node('A', 'Input Layer\n(Batch of EEG trials)')
    dot.node('B', 'Temporal Convolution\nKernel: (1, 25), Stride: (1, 1), Activation: ELU')
    dot.node('C', 'Spatial Convolution\nKernel: (ch, 1), Stride: (1, 1), Activation: ELU')
    dot.node('D', 'Batch Normalization')
    dot.node('E', 'Average Pooling\nKernel: (1, 75), Stride: (1, 15)')
    dot.node('F', 'Token Formation')
    dot.node('G', 'Multi-Head Attention\nHeads: h, Activation: Softmax')
    dot.node('H', 'Fully Connected Layers\nActivation: Softmax')
    dot.node('I', 'Output Layer\n(M-dimensional vector)')

    # Adding edges
    dot.edge('A', 'B', 'flow')
    dot.edge('B', 'C', 'flow')
    dot.edge('C', 'D', 'flow')
    dot.edge('D', 'E', 'flow')
    dot.edge('E', 'F', 'flow')
    dot.edge('F', 'G', 'flow')
    dot.edge('G', 'H', 'flow')
    dot.edge('H', 'I', 'flow')

    print(dot.source)  # To print the generated source code (optional)
    dot.render('EEG_Conformer_Architecture', format='png', cleanup=True)  # Save and render as PNG

visualize_network()

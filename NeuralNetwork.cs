namespace NeuralNetwork {
    public class NeuralNetwork {
        private readonly List<Layer> layers;
        private readonly IHiddenFunction hiddenFunction;
        private readonly IOutputFunction outputFunction;

        public NeuralNetwork( List<int> layerSizes, IHiddenFunction hiddenFunction, IOutputFunction outputFunction ) {
            layers = [];
            this.hiddenFunction = hiddenFunction;
            this.outputFunction = outputFunction;

            for ( int i = 0; i < layerSizes.Count; i++ ) {
                int numInputs = ( i == 0 ) ? 0 : layerSizes[i - 1];
                layers.Add(new Layer(layerSizes[i], numInputs, hiddenFunction));
            }
        }

        public void FeedForward( List<float> inputs ) {
            for ( int i = 0; i < layers.Count; i++ ) {
                var layer = layers[i];
                if ( i == 0 ) {
                    for ( int j = 0; j < inputs.Count; j++ ) {
                        layer.Neurons[j].SetInput(inputs[j]);
                    }
                } else {
                    var prevLayerOutputs = layers[i - 1].Neurons.Select(neuron => neuron.GetRawOutput()).ToList();
                    foreach ( var neuron in layer.Neurons ) {
                        neuron.SetInput(prevLayerOutputs);
                    }
                }
            }
        }

        public void Backpropagate( List<float> expectedOutputs, float learningRate ) {
            var outputLayer = layers[^1];
            List<float> outputErrors = [.. outputLayer.Neurons.Select(( neuron, i ) => expectedOutputs[i] - neuron.GetRawOutput())];

            List<float> nextLayerErrors = outputErrors;
            for ( int i = layers.Count - 1; i > 0; i-- ) {
                var layer = layers[i];
                var prevLayer = layers[i - 1];
                List<float> currentLayerErrors = new(prevLayer.Neurons.Count);

                for ( int j = 0; j < layer.Neurons.Count; j++ ) {
                    var neuron = layer.Neurons[j];
                    float delta = nextLayerErrors[j] * hiddenFunction.Derivative(neuron.GetRawOutput());
                    for ( int k = 0; k < neuron.Weights.Count; k++ ) {
                        if ( neuron.Weights.Count == 0 )
                            continue;
                        Console.WriteLine(k);
                        Console.WriteLine(currentLayerErrors.Count);
                        Console.WriteLine(neuron.Weights.Count);
                        currentLayerErrors[k] += neuron.Weights[k] * delta;
                        neuron.Weights[k] += learningRate * delta * neuron.Inputs[k];
                    }
                    neuron.Bias += learningRate * delta;
                }
                nextLayerErrors = currentLayerErrors;
            }
        }

        public List<float> GetOutput() {
            return outputFunction.Compute([.. layers[^1].Neurons.Select(neuron => neuron.GetRawOutput())]);
        }
    }
}

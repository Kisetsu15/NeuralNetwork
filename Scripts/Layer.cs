namespace NeuralNetwork {
    public class Layer {
        public List<Neuron> Neurons { get; private set; }

        public Layer( int numInputs, int numNeurons, Random rand, IHiddenFunction hiddenFunction, IOutputFunction outputFunction ) {
            Neurons = [];

            for ( int i = 0; i < numNeurons; i++ ) {
                Neurons.Add(new Neuron(numInputs, rand, hiddenFunction, outputFunction));
            }
        }

        public List<float> Forward( List<float> inputs ) {
            return Neurons.Select(neuron => neuron.Forward(inputs)).ToList();
        }

        public List<float> GetOutputErrors( List<float> expectedOutputs ) {
            return [.. Neurons.Select(( neuron, i ) => expectedOutputs[i] - neuron.Output)];
        }

        public List<float> Backpropagate( List<float> nextLayerErrors, float learningRate ) {
            List<float> errors = [.. new float[Neurons[0].Weights.Count]];

            for ( int i = 0; i < Neurons.Count; i++ ) {
                Neurons[i].Backpropagate(nextLayerErrors[i], learningRate, errors);
            }

            return errors;
        }
    }
}

namespace NeuralNetwork {
    public class Layer {

        public List<Neuron> Neurons { get; private set; }

        public Layer( int numNeurons, int numInputsPerNeuron, IHiddenFunction function) {
            Neurons = [];
            for ( int i = 0; i < numNeurons; i++ ) {
                var neuron = new Neuron.Builder()
                    .WithInputs([.. new float[numInputsPerNeuron]])
                    .WithWeights([.. new float[numInputsPerNeuron]])
                    .WithBias(0.1f)
                    .WithFunction(function)
                    .Build();
                Neurons.Add(neuron);
            }
        }

    }
}

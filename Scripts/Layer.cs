namespace NeuralNetwork {
    public class Layer {
        public List<Neuron> Neurons { get; private set; }

        private Layer( int numNeurons, int numInputsPerNeuron, IHiddenFunction function, Random random ) {
            Neurons = [];

            for ( int i = 0; i < numNeurons; i++ ) {
                var neuron = new Neuron.Builder()
                    .WithInputs(new float[numInputsPerNeuron].ToList()) 
                    .WithWeights([.. Enumerable.Range(0, numInputsPerNeuron).Select(_ => (float)( random.NextDouble() * 2 - 1 ))])
                    .WithBias((float)( random.NextDouble() * 2 - 1 )) 
                    .WithFunction(function)
                    .Build();
                Neurons.Add(neuron);
            }
        }

        public class Builder {
            private int numNeurons;
            private int numInputsPerNeuron;
            private IHiddenFunction? function;
            private Random? random;

            public Builder WithNumNeurons( int numNeurons ) {
                this.numNeurons = numNeurons;
                return this;
            }
            public Builder WithNumInputsPerNeuron( int numInputsPerNeuron ) {
                this.numInputsPerNeuron = numInputsPerNeuron;
                return this;
            }
            public Builder WithFunction( IHiddenFunction function ) {
                this.function = function;
                return this;
            }
            public Builder WithRandom( Random random ) {
                this.random = random;
                return this;
            }
            public Layer Build() => new(numNeurons, numInputsPerNeuron, function!, random!);           
        }
    }

}

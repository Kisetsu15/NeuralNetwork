namespace NeuralNetwork {
    public class NeuralNetwork {
        private readonly List<Layer> layers;
        private readonly float learningRate;

        public NeuralNetwork( List<int> layerSizes, IHiddenFunction hiddenFunction, IOutputFunction outputFunction, float learningRate = 0.1f ) {
            this.learningRate = learningRate;
            layers = [];

            Random rand = new(42); // Fixed seed for reproducibility

            for ( int i = 0; i < layerSizes.Count - 1; i++ ) {
                layers.Add(new Layer(layerSizes[i], layerSizes[i + 1], rand, hiddenFunction, outputFunction));
            }
        }

        public List<float> Forward( List<float> inputs ) {
            foreach ( var layer in layers ) {
                inputs = layer.Forward(inputs);
            }
            return inputs;
        }

        public void Backpropagate( List<float> expectedOutputs ) {
            List<float> errors = layers[^1].GetOutputErrors(expectedOutputs);

            for ( int i = layers.Count - 1; i >= 0; i-- ) {
                errors = layers[i].Backpropagate(errors, learningRate);
            }
        }

        public void Train( List<List<float>> inputs, List<List<float>> outputs, int epochs ) {
            for ( int epoch = 0; epoch < epochs; epoch++ ) {
                float totalError = 0;

                for ( int i = 0; i < inputs.Count; i++ ) {
                    List<float> output = Forward(inputs[i]);
                    Backpropagate(outputs[i]);

                    totalError += MathF.Abs(outputs[i][0] - output[0]); // XOR has single output
                }

                if ( epoch % 1000 == 0 ) {
                    Console.WriteLine($"Epoch {epoch}, Accuracy: {100f - totalError}");
                }
            }
        }

        public class Builder {
            private List<int> layerSizes = [];
            private float learningRate;
            private IHiddenFunction hiddenFunction = new Sigmoid();      // Default Hidden function ReLu
            private IOutputFunction outputFunction = new Softmax();   // Default Output function Softmax
            public Builder WithLayerSizes( List<int> layerSizes ) {
                this.layerSizes = layerSizes;
                return this;
            }
            public Builder WithLearningRate( float rate ) {
                this.learningRate = rate;
                return this;
            }
            public Builder WithHiddenFunction( IHiddenFunction hiddenFunction ) {
                this.hiddenFunction = hiddenFunction;
                return this;
            }
            public Builder WithOutputFunction( IOutputFunction outputFunction ) {
                this.outputFunction = outputFunction;
                return this;
            }
            public NeuralNetwork Build() => new(layerSizes, hiddenFunction, outputFunction, learningRate);
        }
    }

}
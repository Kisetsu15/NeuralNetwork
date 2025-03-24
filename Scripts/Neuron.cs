namespace NeuralNetwork {
    public class Neuron {
        public List<float> Inputs { get; private set; }
        public List<float> Weights { get; private set; }
        public float Bias { get; set; }

        private readonly IHiddenFunction function;
        private float? output;

        private Neuron( List<float> inputs, List<float> weights, float bias, IHiddenFunction function ) {
            Inputs = inputs;
            Weights = weights;
            Bias = bias;
            this.function = function;
        }

        public void SetInputs( List<float> inputs ) {
            Inputs = inputs;
        }
        
        public void SetInput( float input ) {
            Inputs = [input];
        }

        public float Calculate() {
            float sum = 0;

            for ( int i = 0; i < Inputs.Count; i++ ) {
                if ( Weights.Count == 0 ) {
                    sum += Inputs[i];
                    continue;
                }
                sum += Inputs[i] * Weights[i];
            }
            return function.Activate(sum + Bias);
        }

        public float GetRawOutput() {
            output ??= Calculate();
            return (float) output!;
        }

        public class Builder {
            private List<float> inputs = [];
            private List<float> weights = [];
            private float bias = 0;
            private IHiddenFunction? function;

            public Builder WithInputs( List<float> inputs ) {
                this.inputs = inputs;
                return this;
            }
            public Builder WithWeights( List<float> weights ) {
                this.weights = weights;
                return this;
            }
            public Builder WithBias( float bias ) {
                this.bias = bias;
                return this;
            }
            public Builder WithFunction( IHiddenFunction function ) {
                this.function = function;
                return this;
            }
            public Neuron Build() => new(inputs, weights, bias, function!);
        }
    }
}

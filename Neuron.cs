namespace NeuralNetwork {
    public class Neuron {
        public List<float> Inputs { get; private set; }
        public List<float> Weights { get; private set; }
        public float Bias { get; set; }

        private float? output = null;
        private readonly IHiddenFunction function;

        private Neuron( List<float> inputs, List<float> weights, float bias, IHiddenFunction function ) {
            this.Inputs = inputs;
            this.Weights = weights;
            this.Bias = bias;
            this.function = function;
        }

        public void SetInput( List<float> inputs ) {
            this.Inputs = inputs;
        }

        public void SetWeight( float weight ) {
            this.Weights = [weight];
        }  
        
        public void SetInput( float input ) {
            this.Inputs = [input];
        }

        public void Calculate() {
            float sum = 0;

            for ( int i = 0; i < Inputs.Count; i++ ) {
                if (Weights.Count == 0) {
                    sum += Inputs[i];
                    continue;
                }
                sum += Inputs[i] * Weights[i];
            }
            output = function.Activate(sum + Bias); // Apply the Sigmoid activation hiddenFunction
        }

        public float GetRawOutput() {
            if ( output == null ) {
                Calculate();
            }

            return (float)output! ;    
        }
        

        // Builder class for constructing the Neuron
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

            public Neuron Build() {
                return new Neuron(inputs, weights, bias, function!);
            }
        }
    }
}

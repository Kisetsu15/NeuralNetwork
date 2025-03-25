
namespace NeuralNetwork {
    public class Neuron {
        public List<float> Weights { get; private set; }
        public float Bias { get; private set; }
        public float Output { get; private set; }
        private List<float> Inputs = [];
        private IHiddenFunction HiddenFunction = new Sigmoid();

        public Neuron( int numInputs, Random rand, IHiddenFunction hiddenFunction, IOutputFunction outputFunciton ) {
            Weights = [];

            for ( int i = 0; i < numInputs; i++ ) {
                Weights.Add((float)( rand.NextDouble() * 2 - 1 )); // Random weight (-1 to 1)
            }

            Bias = (float)( rand.NextDouble() * 2 - 1 );
        }

        public float Forward( List<float> inputs ) {
            Inputs = inputs;
            float sum = Inputs.Zip(Weights, ( input, weight ) => input * weight).Sum() + Bias;
            Output = HiddenFunction.Activate(sum);
            return Output;
        }

        public void Backpropagate( float error, float learningRate, List<float> prevLayerErrors ) {
            float delta = error * HiddenFunction.Derivative(Output);

            for ( int i = 0; i < Weights.Count; i++ ) {
                prevLayerErrors[i] += Weights[i] * delta;
                Weights[i] += learningRate * delta * Inputs[i];
            }

            Bias += learningRate * delta;
        }
    }
}
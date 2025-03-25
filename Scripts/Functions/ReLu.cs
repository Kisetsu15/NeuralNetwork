namespace NeuralNetwork {
    public class ReLu : IHiddenFunction {
        public float Activate( float value ) {
            return value > 0f ? value : 0f;
        }

        public float Derivative( float value ) {
            return value > 0f ? 1f : 0f;
        }
    }
}

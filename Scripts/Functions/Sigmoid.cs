namespace NeuralNetwork {
    public class Sigmoid : IHiddenFunction {
        public float Activate( float value ) {
            return 1.0f / ( 1.0f + MathF.Exp(-value) );
        }
        public float Derivative( float value ) {
            return value * ( 1.0f - value );
        }
    }
}

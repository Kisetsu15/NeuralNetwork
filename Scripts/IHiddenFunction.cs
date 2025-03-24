namespace NeuralNetwork {
    public interface IHiddenFunction {
        public abstract float Activate( float value );
        public abstract float Derivative( float value );
    }
}
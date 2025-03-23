namespace NeuralNetwork {
    public interface IOutputFunction {
        public abstract List<float> Compute( List<float> list );
    }
}
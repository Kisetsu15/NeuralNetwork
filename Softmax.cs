namespace NeuralNetwork {
    public class Softmax : IOutputFunction {
        public List<float> Compute( List<float> list ) {
            float sum = list.Select(x => MathF.Exp(x)).Sum();
            return [.. list.Select(z => MathF.Exp(z) / sum)];
        }
    }
}

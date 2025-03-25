
namespace NeuralNetwork {
    class Program {
        static void Main() {

            NeuralNetwork nn = new NeuralNetwork.Builder()
                .WithHiddenFunction(new Sigmoid())
                .WithLearningRate(0.1f)
                .WithLayerSizes([ 2, 100, 1 ])
                .Build();


            List<List<float>> trainingInputs = [
                [0f, 0f],
                [0f, 1f],
                [1f, 0f],
                [1f, 1f]
            ];

            List<List<float>> expectedOutputs = [
                [0f],
                [1f],
                [1f],
                [0f]
            ];

            nn.Train(trainingInputs, expectedOutputs, 10000);

            Console.WriteLine("\nTesting XOR Network:");
            foreach ( var input in trainingInputs ) {
                List<float> output = nn.Forward(input);
                Console.WriteLine($"Input: {string.Join(",", input)} => Output: {output[0]:F4}");
            }
        }
    }
}


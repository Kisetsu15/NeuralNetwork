namespace NeuralNetwork
{
    public static class Train {
        private static void Main( string[] args ) {
            // Create a simple neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
            NeuralNetwork neuralNetwork = new([2, 2, 1], new Sigmoid(), new Softmax());

            // Training data: XOR problem
            List<List<float>> trainingInputs = [
                [ 0f, 0f ],
                [ 0f, 1f ],
                [ 1f, 0f ],
                [ 1f, 1f ]
            ];

            List<List<float>> expectedOutputs = [
                [0f],
                [1f],
                [1f],
                [0f]
            ];

            // Train the network
            for ( int epoch = 0; epoch < 100; epoch++ ) {
                for ( int i = 0; i < trainingInputs.Count; i++ ) {
                    neuralNetwork.FeedForward(trainingInputs[i]);
                    neuralNetwork.Backpropagate(expectedOutputs[i], 0.1f);
                }
            }

            // Test the network
            Console.WriteLine("Testing XOR Network:");

            for ( int i = 0; i < trainingInputs.Count; i++ ) {
                var input = trainingInputs[i];
                neuralNetwork.FeedForward(input);
                var output = neuralNetwork.GetOutput();
                Console.WriteLine($"Input: {string.Join(",", input)} => Output: {output[0]}");
            }
        }
    }
}

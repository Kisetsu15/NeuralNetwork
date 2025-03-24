namespace NeuralNetwork {
    public static class Demo {
        private static void Main( string[] args ) {

            var neuralNetwork = new NeuralNetwork.Builder()
                .WithLayerSizes([2, 2, 1])
                .WithHiddenFunction(new ReLu())
                .WithOutputFunction(new Softmax())
                .Build();

            List<List<float>> trainingInputs = [
                [0f, 0f],
                [0f, 1f],
                [1f, 0f],
                [1f, 1f]
            ];

            List<List<float>> expectedOutputs = [
                [0.01f],
                [0.99f],
                [0.99f],
                [0.01f]
            ];
            
            int epochs = 10000;
            float learningRate = 0.1f;

            for ( int epoch = 0; epoch < epochs; epoch++ ) {
                float totalError = 0f;

                for ( int i = 0; i < trainingInputs.Count; i++ ) {
                    neuralNetwork.Forwardpropagate(trainingInputs[i]);
                    neuralNetwork.Backpropagate(expectedOutputs[i], learningRate);

                    var output = neuralNetwork.GetOutput(out _);
                    totalError += MathF.Abs(expectedOutputs[i][0] - output[0]);
                }

                if ( epoch % 1000 == 0 ) {
                    Console.WriteLine($"Epoch {epoch}, Accuracy: {(1 - (totalError / trainingInputs.Count)) * 100f}");
                }
            }

            Console.WriteLine("\nTesting XOR Network:");
            foreach ( var input in trainingInputs ) {
                neuralNetwork.Forwardpropagate(input);
                var output = neuralNetwork.GetOutput(out _);
                Console.WriteLine($"Input: {string.Join(",", input)} => Output: {output[0]:F4}");
            }
        }

    }
}

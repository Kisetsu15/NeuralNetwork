using NeuralNetwork;

public class Tanh : IHiddenFunction {
    public float Activate( float value ) {
        return MathF.Tanh(value);
    }

    public float Derivative( float value ) {
        return 1 - ( value * value );
    }
}
namespace Cogni.MultilayerPerceptronNetwork;

public class Perceptron
{
    private double[] _inputWeights { get; set; }
    private double _bias { get; set; }

    public Perceptron(int numberOfInputs)
    {
        _inputWeights = new double[numberOfInputs];
        
        Random rand = new Random();
        for(int i = 0 ; i < numberOfInputs ; i++) {
            _inputWeights[i] = rand.NextDouble() * 2 - 1;
        }

        _bias = rand.NextDouble() * 2 - 1;
    }

    public double Predict(double[] input) {
        var result = 0.0;

        for(int i = 0 ; i < input.Count() ; i++) {
            result += input[i] * _inputWeights[i];
        }

        result += _bias;

        return SigmoidFunction(result);
    }

    // TODO: export to IActivationFunction interface
    private double SigmoidFunction(double input)
    {
        return 1.0 / (1.0 + Math.Exp(-input));
    }
}
namespace Cogni.MultilayerPerceptronNetwork;

public class Perceptron
{
    public double[] Weights { get; set; }
    public int Index { get; set; }
    public double Bias { get; set; }
    public double LastOutput { get; set; }

    private PerceptronLayerType LayerType;

    public Perceptron(int numberOfInputs, PerceptronLayerType layerType, int index)
    {
        LayerType = layerType;
        Weights = new double[numberOfInputs];

        if (LayerType == PerceptronLayerType.Input)
        {
            Weights[0] = 1;
            Bias = 0;
        }
        else
        {
            Random rand = new Random();
            for (int i = 0; i < numberOfInputs; i++)
            {
                Weights[i] = rand.NextDouble() * 2 - 1;
            }

            Bias = rand.NextDouble() * 2 - 1;
        }

    }

    public double Predict(double[] input)
    {
        var result = 0.0;

        for (int i = 0; i < input.Count(); i++)
        {
            result += input[i] * Weights[i];
        }

        result += Bias;

        var output = SigmoidFunction(result);
        LastOutput = output;
        return output;
    }

    // TODO: export to IActivationFunction interface
    private double SigmoidFunction(double input)
    {
        return 1.0 / (1.0 + Math.Exp(-input));
    }

    private double ReLUFunction(double input)
    {
        if (input > 0)
            return input;
        return 0;
    }
}
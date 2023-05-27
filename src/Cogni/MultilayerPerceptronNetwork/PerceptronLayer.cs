namespace Cogni.MultilayerPerceptronNetwork;

public class PerceptronLayer
{
    public PerceptronLayerType LayerType { get; set; }
    public int NumberOfPerceptrons => Perceptrons.Length;
    public Perceptron[] Perceptrons { get; set; }

    private PerceptronLayer() { }
    
    public static PerceptronLayer CreateInputLayer(int numberOfInputs)
    {
        var layer = new PerceptronLayer();
        layer.Perceptrons = Enumerable.Range(0, numberOfInputs)
            .Select(i => new Perceptron(1, PerceptronLayerType.Input, i))
            .ToArray();

        layer.LayerType = PerceptronLayerType.Input;
        return layer;
    }

    public static PerceptronLayer CreateHiddenLayer(int numberOfPerceptrons, int numberOfPreviousLayerPerceptrons)
    {
        var layer = new PerceptronLayer();
        layer.Perceptrons = Enumerable.Range(0, numberOfPerceptrons)
            .Select(i => new Perceptron(numberOfPreviousLayerPerceptrons, PerceptronLayerType.Hidden, i))
            .ToArray();

        layer.LayerType = PerceptronLayerType.Hidden;
        return layer;
    }

    public static PerceptronLayer CreateOutputLayer(int numberOfOutputs, int numberOfPreviousLayerPerceptrons)
    {
        var layer = new PerceptronLayer();
        layer.Perceptrons = Enumerable.Range(0, numberOfOutputs)
            .Select(i => new Perceptron(numberOfPreviousLayerPerceptrons, PerceptronLayerType.Output, i))
            .ToArray();

        layer.LayerType = PerceptronLayerType.Output;
        return layer;
    }

    public double[] CalculateOutput(double[] input)
    {
        var output = new double[Perceptrons.Length];

        if(LayerType == PerceptronLayerType.Input)
        {
            for (int i = 0; i < Perceptrons.Length; i++)
            {
                output[i] = Perceptrons[i].Predict(new double[] { input[i] });
            }
        }
        else
        {
            for (int i = 0; i < Perceptrons.Length; i++)
            {
                output[i] = Perceptrons[i].Predict(input);
            }
        }

        return output;
    }
}

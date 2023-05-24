namespace Cogni.MultilayerPerceptronNetwork;

public class PerceptronLayer
{
    public PerceptronLayerType LayerType { get; set; }
    public int NumberOfPerceptrons => Perceptrons.Count;
    public List<Perceptron> Perceptrons { get; set; }

    private PerceptronLayer() {
        Perceptrons = new List<Perceptron>();
     }
    
    public static PerceptronLayer CreateInputLayer(int numberOfInputs)
    {
        var layer = new PerceptronLayer();
        layer.Perceptrons = Enumerable.Range(0, numberOfInputs)
            .Select(i => new Perceptron(1, PerceptronLayerType.Input, i))
            .ToList();

        layer.LayerType = PerceptronLayerType.Input;
        return layer;
    }

    public static PerceptronLayer CreateHiddenLayer(int numberOfPerceptrons, int numberOfPreviousLayerPerceptrons)
    {
        var layer = new PerceptronLayer();
        layer.Perceptrons = Enumerable.Range(0, numberOfPerceptrons)
            .Select(i => new Perceptron(numberOfPreviousLayerPerceptrons, PerceptronLayerType.Hidden, i))
            .ToList();

        layer.LayerType = PerceptronLayerType.Hidden;
        return layer;
    }

    public static PerceptronLayer CreateOutputLayer(int numberOfOutputs, int numberOfPreviousLayerPerceptrons)
    {
        var layer = new PerceptronLayer();
        layer.Perceptrons = Enumerable.Range(0, numberOfOutputs)
            .Select(i => new Perceptron(numberOfPreviousLayerPerceptrons, PerceptronLayerType.Output, i))
            .ToList();

        layer.LayerType = PerceptronLayerType.Output;
        return layer;
    }

    public double[] CalculateOutput(double[] input)
    {
        var output = new double[Perceptrons.Count];

        if(LayerType == PerceptronLayerType.Input)
        {
            for (int i = 0; i < Perceptrons.Count; i++)
            {
                output[i] = Perceptrons[i].Predict(new double[] { input[i] });
            }
        }
        else
        {
            for (int i = 0; i < Perceptrons.Count; i++)
            {
                output[i] = Perceptrons[i].Predict(input);
            }
        }

        return output;
    }
}

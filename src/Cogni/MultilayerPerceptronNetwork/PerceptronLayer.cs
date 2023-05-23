namespace Cogni.MultilayerPerceptronNetwork;

public class PerceptronLayer
{
    public PerceptronLayerType LayerType { get; set; }
    public int NumberOfPerceptrons => _perceptrons.Count;
    private List<Perceptron> _perceptrons { get; set; }

    private PerceptronLayer() {
        _perceptrons = new List<Perceptron>();
     }
    
    public static PerceptronLayer CreateInputLayer(int numberOfInputs)
    {
        var layer = new PerceptronLayer();
        layer._perceptrons = Enumerable.Range(0, numberOfInputs)
            .Select(i => new Perceptron(1))
            .ToList();

        layer.LayerType = PerceptronLayerType.Input;
        return layer;
    }

    public static PerceptronLayer CreateHiddenLayer(int numberOfPerceptrons, int numberOfPreviousLayerPerceptrons)
    {
        var layer = new PerceptronLayer();
        layer._perceptrons = Enumerable.Range(0, numberOfPerceptrons)
            .Select(i => new Perceptron(numberOfPreviousLayerPerceptrons))
            .ToList();

        layer.LayerType = PerceptronLayerType.Hidden;
        return layer;
    }

    public static PerceptronLayer CreateOutputLayer(int numberOfOutputs, int numberOfPreviousLayerPerceptrons)
    {
        var layer = new PerceptronLayer();
        layer._perceptrons = Enumerable.Range(0, numberOfOutputs)
            .Select(i => new Perceptron(numberOfPreviousLayerPerceptrons))
            .ToList();

        layer.LayerType = PerceptronLayerType.Output;
        return layer;
    }

    public double[] Calculate(double[] input)
    {
        var output = new double[_perceptrons.Count];

        if(LayerType == PerceptronLayerType.Input)
        {
            for (int i = 0; i < _perceptrons.Count; i++)
            {
                output[i] = _perceptrons[i].Predict(new double[] { input[i] });
            }
        }
        else
        {
            for (int i = 0; i < _perceptrons.Count; i++)
            {
                output[i] = _perceptrons[i].Predict(input);
            }
        }

        return output;
    }
}

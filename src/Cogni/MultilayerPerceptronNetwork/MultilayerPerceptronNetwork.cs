namespace Cogni.MultilayerPerceptronNetwork;

public class MultilayerPerceptronNetwork
{
    private List<PerceptronLayer> _layers { get; set; }

    // TODO: Create network builder
    public MultilayerPerceptronNetwork()
    {
        _layers = new List<PerceptronLayer>();
    }

    public MultilayerPerceptronNetwork AddInputLayer(int numberOfInputs)
    {
        if(_layers.Any(layer => layer.LayerType == PerceptronLayerType.Input))
            throw new InvalidOperationException("Cannot add more than one input layer.");

        _layers.Add(PerceptronLayer.CreateInputLayer(numberOfInputs));
        return this;
    }

    public MultilayerPerceptronNetwork AddHiddenLayer(int numberOfPerceptrons)
    {
        if(!_layers.Any(layer => layer.LayerType == PerceptronLayerType.Input))
            throw new InvalidOperationException("Cannot add hidden layer while no input layer is defined.");

        if(_layers.Any(layer => layer.LayerType == PerceptronLayerType.Output))
            throw new InvalidOperationException("Cannot add hidden layer while output layer is already defined.");

        _layers.Add(PerceptronLayer.CreateHiddenLayer(numberOfPerceptrons, _layers.Last().NumberOfPerceptrons));
        return this;
    }

    public MultilayerPerceptronNetwork AddOutputLayer(int numberOfOutputs)
    {      
        if(_layers.Any(layer => layer.LayerType == PerceptronLayerType.Output))
            throw new InvalidOperationException("Cannot add more than one output layer.");

        _layers.Add(PerceptronLayer.CreateOutputLayer(numberOfOutputs, _layers.Last().NumberOfPerceptrons));
        return this;
    }

    public double[] Predict(double[] input)
    {
        // TODO: validate if network has Input and Output layers

        var nextInput = input;
        for(int i = 0 ; i < _layers.Count ; i++) {
            nextInput = _layers[i].Calculate(nextInput);
        }

        return nextInput;
    }

    public void Train(double[] input, double[] expectedOutput)
    {
        // TODO: implement back propagation algorithm
    }
}

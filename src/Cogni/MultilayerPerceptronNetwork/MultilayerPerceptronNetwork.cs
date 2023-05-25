using System;

namespace Cogni.MultilayerPerceptronNetwork;

public class MultilayerPerceptronNetwork
{
    public List<PerceptronLayer> Layers { get; set; }
    private int NumberOfLayers => Layers.Count;
    private double LearningRate => 0.7;

    // TODO: Create network builder
    public MultilayerPerceptronNetwork()
    {
        Layers = new List<PerceptronLayer>();
    }

    public MultilayerPerceptronNetwork AddInputLayer(int numberOfInputs)
    {
        if(Layers.Any(layer => layer.LayerType == PerceptronLayerType.Input))
            throw new InvalidOperationException("Cannot add more than one input layer.");

        Layers.Add(PerceptronLayer.CreateInputLayer(numberOfInputs));
        return this;
    }

    public MultilayerPerceptronNetwork AddHiddenLayer(int numberOfPerceptrons)
    {
        if(!Layers.Any(layer => layer.LayerType == PerceptronLayerType.Input))
            throw new InvalidOperationException("Cannot add hidden layer while no input layer is defined.");

        if(Layers.Any(layer => layer.LayerType == PerceptronLayerType.Output))
            throw new InvalidOperationException("Cannot add hidden layer while output layer is already defined.");

        Layers.Add(PerceptronLayer.CreateHiddenLayer(numberOfPerceptrons, Layers.Last().NumberOfPerceptrons));
        return this;
    }

    public MultilayerPerceptronNetwork AddOutputLayer(int numberOfOutputs)
    {      
        if(Layers.Any(layer => layer.LayerType == PerceptronLayerType.Output))
            throw new InvalidOperationException("Cannot add more than one output layer.");

        Layers.Add(PerceptronLayer.CreateOutputLayer(numberOfOutputs, Layers.Last().NumberOfPerceptrons));
        return this;
    }

    public double[] Predict(double[] input)
    {
        // TODO: validate if network has Input and Output layers

        var nextInput = input;
        for(int i = 0 ; i < NumberOfLayers; i++)
        {
            var output = Layers[i].CalculateOutput(nextInput);
            nextInput = output;
        }

        return nextInput;
    }

    public void Train(double[] input, double[] expectedOutput)
    {
        // Feed forward
        var prediction = Predict(input);

        // Output errors
        var outputErrors = new double[expectedOutput.Length];
        for(int i = 0; i < expectedOutput.Length; i++)
        {
            outputErrors[i] = 0.5 * Math.Pow(expectedOutput[i] - prediction[i], 2);
        }

        // Output layer weights update
        double[][] deltas = new double[NumberOfLayers][];

        var outputLayerIndex = NumberOfLayers - 1;
        var outputLayer = Layers[outputLayerIndex];

        deltas[outputLayerIndex] = new double[outputLayer.NumberOfPerceptrons];
        for(int i = 0; i < outputLayer.NumberOfPerceptrons; i++)
        {
            deltas[outputLayerIndex][i] = (prediction[i] - expectedOutput[i]) * SigmoidDerivative(prediction[i]);
        }

        // Output layer weights update
        for (int i = 0; i < outputLayer.NumberOfPerceptrons; i++)
        {
            for (int j = 0; j < outputLayer.Perceptrons[i].Weights.Length; j++)
            {
                var errorDerivative = deltas[outputLayerIndex][i] * Layers[outputLayerIndex - 1].Perceptrons[j].LastOutput;
                outputLayer.Perceptrons[i].Weights[j] -= LearningRate * errorDerivative;
            }

            outputLayer.Perceptrons[i].Bias -= LearningRate * deltas[outputLayerIndex][i];
        }

        // Hidden layers weigths update
        for (int i = outputLayerIndex - 1; i > 0; i--)
        {
            deltas[i] = new double[Layers[i].NumberOfPerceptrons];
            for (int j = 0; j < Layers[i].NumberOfPerceptrons; j++)
            {
                var errorToOutputDiff = 0.0;
                for (int k = 0; k < Layers[i + 1].NumberOfPerceptrons; k++)
                {
                    errorToOutputDiff += deltas[i + 1][k] * Layers[i + 1].Perceptrons[k].Weights[j];
                }

                deltas[i][j] = errorToOutputDiff * SigmoidDerivative(Layers[i].Perceptrons[j].LastOutput);

                for (int k = 0; k < Layers[i].Perceptrons[j].Weights.Length; k++)
                {
                    var errorDerivative = deltas[i][j] * Layers[i - 1].Perceptrons[k].LastOutput;
                    Layers[i].Perceptrons[j].Weights[k] -= LearningRate * errorDerivative;
                }

                Layers[i].Perceptrons[j].Bias -= LearningRate * deltas[i][j];
            }
        }
    }

    private double SigmoidDerivative(double value)
    {
        return value * (1 - value);
    }

    private double ReLUDerivative(double value)
    {
        if (value > 0)
            return 1;
        return 0;
    }
}

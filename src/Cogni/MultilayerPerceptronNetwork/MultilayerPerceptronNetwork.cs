using Cogni.MultilayerPerceptronNetwork.ActivationFunctions;

namespace Cogni.MultilayerPerceptronNetwork;

public class MultilayerPerceptronNetwork
{
    private int NumberOfLayers => Weights.Length;
    private double LearningRate => 0.7;
    internal int InputsCount { get; set; }
    internal int OutputsCount { get; set; }

    internal WeightMatrix[] Weights { get; set; }
    internal BiasMatrix[] Bias { get; set; }
    internal Matrix[] Outputs { get; set; }
    internal IActivationFunction ActivationFunction { get; set; }

    // TODO: Create network builder
    public MultilayerPerceptronNetwork()
    {

    }

    public MultilayerPerceptronNetwork AddInputLayer(int numberOfInputs)
    {
        Weights = new WeightMatrix[1];
        Weights[0] = new WeightMatrix(InputsCount, numberOfInputs, 1.0);

        InputsCount = numberOfInputs;
        return this;
    }

    public MultilayerPerceptronNetwork AddHiddenLayer(int numberOfPerceptrons)
    {
        var newWeights = new WeightMatrix[Weights.Length + 1];
        for(int i = 0; i < Weights.Length; i++)
        {
            newWeights[i] = Weights[i];
        }
        newWeights[Weights.Length] = new WeightMatrix(newWeights[Weights.Length - 1].Columns, numberOfPerceptrons, 0.5);
        Weights = newWeights;

        return this;
    }

    public MultilayerPerceptronNetwork AddOutputLayer(int numberOfOutputs)
    {
        var newWeights = new WeightMatrix[Weights.Length + 1];
        for (int i = 0; i < Weights.Length; i++)
        {
            newWeights[i] = Weights[i];
        }
        newWeights[Weights.Length] = new WeightMatrix(newWeights[Weights.Length - 1].Columns, numberOfOutputs, 0.5);
        Weights = newWeights;

        OutputsCount = numberOfOutputs;
        return this;
    }

    public MultilayerPerceptronNetwork AddBias()
    {
        // INFO: should be called after layers definition
        Bias = new BiasMatrix[Weights.Length];
        for(int i = 0; i < Bias.Length; i++)
        {
            Bias[i] = new BiasMatrix(LayerPerceptronsCount(i), i == 0 ? 0.0 : 0.5);
        }
        return this;
    }

    public MultilayerPerceptronNetwork WithActivationFunction(IActivationFunction function)
    {
        ActivationFunction = new SigmoidFunction();
        return this;
    }

    public double[] Predict(double[] input)
    {
        // TODO: validate if network has Input and Output layers
        Outputs = new Matrix[NumberOfLayers];
        Outputs[0] = new Matrix(input);

        for (int i = 1; i < Weights.Length; i++)
        {
            Outputs[i] = Outputs[i - 1].MultiplyBy(Weights[i]);
            
            if(Bias != null)
            {
                for (int j = 0; j < Outputs[i].Columns; j++)
                {
                    Outputs[i].Values[0, j] += Bias[i].Values[0, j];
                }
            }

            for (int j = 0; j < Outputs[i].Columns; j++)
            {
                Outputs[i].Values[0, j] = ActivationFunction.GetValueFor(Outputs[i].Values[0, j]);
            }
        }
        var result = Outputs[NumberOfLayers - 1].Flatten();

        return result;
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
        var outputLayerPerceptronsCount = LayerPerceptronsCount(outputLayerIndex);

        deltas[outputLayerIndex] = new double[outputLayerPerceptronsCount];
        for(int i = 0; i < outputLayerPerceptronsCount; i++)
        {
            deltas[outputLayerIndex][i] = (prediction[i] - expectedOutput[i]) * ActivationFunction.GetDerivativeValueFor(prediction[i]);
        }

        // Output layer weights update
        for (int i = 0; i < outputLayerPerceptronsCount; i++)
        {
            for (int j = 0; j < LayerPerceptronsCount(outputLayerIndex - 1); j++)
            {
                var errorDerivative = deltas[outputLayerIndex][i] * Outputs[outputLayerIndex - 1].Values[0, j];
                Weights[outputLayerIndex].Values[j, i] -= LearningRate * errorDerivative;
            }

            if(Bias != null)
                Bias[outputLayerIndex].Values[0, i] -= LearningRate * deltas[outputLayerIndex][i];
        }

        // Hidden layers weigths update
        for (int i = outputLayerIndex - 1; i > 0; i--)
        {
            deltas[i] = new double[LayerPerceptronsCount(i)];
            for (int j = 0; j < LayerPerceptronsCount(i); j++)
            {
                var errorToOutputDiff = 0.0;
                for (int k = 0; k < LayerPerceptronsCount(i + 1); k++)
                {
                    errorToOutputDiff += deltas[i + 1][k] * Weights[i + 1].Values[j, k];
                }

                deltas[i][j] = errorToOutputDiff * ActivationFunction.GetDerivativeValueFor(Outputs[i].Values[0, j]);

                for (int k = 0; k < LayerPerceptronsCount(i - 1); k++)
                {
                    var errorDerivative = deltas[i][j] * Outputs[i - 1].Values[0, k];
                    Weights[i].Values[k, j] -= LearningRate * errorDerivative;
                }

                if (Bias != null)
                    Bias[i].Values[0, j] -= LearningRate * deltas[i][j];
            }
        }
    }

    private int LayerPerceptronsCount(int layerNumber) => 
        Weights[layerNumber].OutputsCount;
}

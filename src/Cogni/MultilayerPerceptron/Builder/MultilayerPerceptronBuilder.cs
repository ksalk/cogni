using Cogni.MultilayerPerceptron.ActivationFunctions;

namespace Cogni.MultilayerPerceptron.Builder
{
    public class MultilayerPerceptronBuilder
    {
        private List<LayerDefinition> _layers;
        private bool _bias;
        private IActivationFunction _activationFunction;

        private MultilayerPerceptronBuilder()
        {
            _layers = new List<LayerDefinition>();
            _bias = false;
            _activationFunction = new SigmoidFunction();
        }

        public MultilayerPerceptronBuilder AddLayer(int perceptronsCount)
        {
            _layers.Add(new LayerDefinition(perceptronsCount));
            return this;
        }

        public static MultilayerPerceptronBuilder CreateNetwork()
        {
            return new MultilayerPerceptronBuilder();
        }


        public MultilayerPerceptronBuilder WithBias()
        {
            _bias = true;
            return this;
        }

        public MultilayerPerceptronBuilder WithActivationFunction(IActivationFunction function)
        {
            _activationFunction = function;
            return this;
        }

        public MultilayerPerceptronBuilder WithActivationFunction<TActivationFunction>() where TActivationFunction : IActivationFunction
        {
            var type = typeof(TActivationFunction);
            _activationFunction = (IActivationFunction)Activator.CreateInstance(type);
            return this;
        }

        public MultilayerPerceptron Build()
        {
            if (_layers.Count < 2)
                throw new InvalidOperationException("Network should have at least 2 layers.");

            var network = new MultilayerPerceptron();

            network.AddInputLayer(_layers[0].PerceptronsCount);

            for (int i = 1; i < _layers.Count - 1; i++)
                network.AddHiddenLayer(_layers[i].PerceptronsCount);

            network.AddOutputLayer(_layers[_layers.Count - 1].PerceptronsCount);

            if (_bias)
                network.AddBias();

            network.WithActivationFunction(_activationFunction);

            return network;
        }
    }

    internal class LayerDefinition
    {
        public int PerceptronsCount { get; private set; }

        public LayerDefinition(int perceptronsCount)
        {
            PerceptronsCount = perceptronsCount;
        }
    }
}

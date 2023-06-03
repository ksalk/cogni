using Cogni.MultilayerPerceptron.ActivationFunctions;

namespace Cogni.MultilayerPerceptron.Builder
{
    public class MultilayerPerceptronBuilder
    {
        private int _layers;
        private Boolean _bias;
        private IActivationFunction _activationFunction;

        private MultilayerPerceptronBuilder()
        {
            _layers = 3;
            _bias = false;
            _activationFunction = new SigmoidFunction();
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
            // TODO : pass network parameters
            return new MultilayerPerceptron();
        }
    }
}

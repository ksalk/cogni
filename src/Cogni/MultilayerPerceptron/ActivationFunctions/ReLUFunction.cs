namespace Cogni.MultilayerPerceptron.ActivationFunctions
{
    public class ReLUFunction : IActivationFunction
    {
        public ReLUFunction()
        {
        }

        public double GetDerivativeValueFor(double argument)
        {
            if (argument > 0)
                return 1;
            return 0;
        }

        public double GetValueFor(double argument)
        {
            if (argument > 0)
                return argument;
            return 0;
        }
    }
}

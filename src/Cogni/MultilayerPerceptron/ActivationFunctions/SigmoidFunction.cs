namespace Cogni.MultilayerPerceptron.ActivationFunctions
{
    public class SigmoidFunction : IActivationFunction
    {
        public SigmoidFunction()
        {
        }

        public double GetDerivativeValueFor(double argument)
        {
            return argument * (1 - argument);
        }

        public double GetValueFor(double argument)
        {
            return 1.0 / (1.0 + Math.Exp(-argument));
        }
    }
}

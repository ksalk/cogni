namespace Cogni.MultilayerPerceptron.ActivationFunctions
{
    public interface IActivationFunction
    {
        double GetValueFor(double argument);
        double GetDerivativeValueFor(double argument);
    }
}

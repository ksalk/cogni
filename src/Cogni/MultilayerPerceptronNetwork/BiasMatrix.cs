namespace Cogni.MultilayerPerceptronNetwork
{
    internal class BiasMatrix : Matrix
    {
        public BiasMatrix(int perceptrons, double? defaultValue = null) : base(1, perceptrons)
        {
            for (int i = 0; i < perceptrons; i++)
            {
                Values[0, i] = defaultValue.HasValue ?
                    defaultValue.Value :
                    Random.Shared.NextDouble() * 2 - 1;
            }
        }
    }
}

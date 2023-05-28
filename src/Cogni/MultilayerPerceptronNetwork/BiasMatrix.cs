namespace Cogni.MultilayerPerceptronNetwork
{
    internal class BiasMatrix : Matrix
    {
        public BiasMatrix(int perceptrons) : base(1, perceptrons)
        {
            for (int i = 0; i < perceptrons; i++)
            {
                Values[0, i] = Random.Shared.NextDouble() * 2 - 1;
            }
        }
    }
}

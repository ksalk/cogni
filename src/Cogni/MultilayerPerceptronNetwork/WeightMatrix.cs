namespace Cogni.MultilayerPerceptronNetwork
{
    internal class WeightMatrix : Matrix
    {
        public int InputsCount => Rows;
        public int OutputsCount => Columns;

        public WeightMatrix(int rows, int cols) : base(rows, cols)
        {
            for(int i = 0; i < rows; i++)
            {
                for(int j = 0; j < cols; j++)
                {
                    Values[i, j] = Random.Shared.NextDouble() * 2 - 1;
                }
            }
        }
    }
}

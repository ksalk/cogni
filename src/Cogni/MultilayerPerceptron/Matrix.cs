namespace Cogni.MultilayerPerceptron
{
    internal class Matrix
    {
        public int Rows { get; set; }
        public int Columns { get; set; }
        public double[,] Values { get; set; }

        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            Values = new double[rows, columns];
        }

        public Matrix(double[] values)
        {
            Rows = 1;
            Columns = values.Length;
            Values = new double[Rows, Columns];

            for (int i = 0; i < Columns; i++)
            {
                Values[0, i] = values[i];
            }
        }

        public Matrix MultiplyBy(Matrix matrix)
        {
            var result = new Matrix(Rows, matrix.Columns);

            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Columns; j++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < Columns; k++)
                    {
                        sum += Values[i, k] * matrix.Values[k, j];
                    }
                    result.Values[i, j] = sum;
                }
            }

            return result;
        }

        public double[] Flatten()
        {
            var result = new double[Rows * Columns];
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    result[i * Columns + j] = Values[i, j];
                }
            }
            return result;
        }
    }
}

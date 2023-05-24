using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cogni.MultilayerPerceptronNetwork
{
    public class DiffData
    {
        public double[] ErrorToInputDiff { get; set; }

        public double[][] Weights { get; set; }

        public DiffData(int numberOfPerceptrons)
        {
            ErrorToInputDiff = new double[numberOfPerceptrons];
            Weights = new double[numberOfPerceptrons][];
        }
    }
}

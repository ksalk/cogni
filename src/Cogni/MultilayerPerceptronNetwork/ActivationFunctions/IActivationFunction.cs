using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Cogni.MultilayerPerceptronNetwork.ActivationFunctions
{
    public interface IActivationFunction
    {
        double GetValueFor(double argument);
        double GetDerivativeValueFor(double argument);
    }
}

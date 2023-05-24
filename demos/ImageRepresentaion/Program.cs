using Cogni.MultilayerPerceptronNetwork;

var mlp = new MultilayerPerceptronNetwork()
    .AddInputLayer(2)
    .AddHiddenLayer(5)
    .AddHiddenLayer(5)
    .AddOutputLayer(1);

var rand = new Random();
var trainingData = Enumerable.Range(0, 100_000).Select(_ =>
{
    double[] input = new double[] { rand.NextDouble(), rand.NextDouble() };
    var value = TestMethod(input[0], input[1]);
    return new TrainingSet(input, value);
}).ToArray();

for(int i = 0; i < trainingData.Length; i++)
{
    mlp.Train(trainingData[i].Input, new double[] { trainingData[i].Target });
}

for (double value = 0.1; value < 1.0; value += 0.1)
{
    var result1 = mlp.Predict(new double[] { value, value }).First();
    Console.WriteLine($"Input: ({value:0.####}, {value:0.####})");
    Console.WriteLine($"Predicted = {result1:0.####}");
    Console.WriteLine($"Actual = {TestMethod(value, value):0.####}");
    Console.WriteLine($"Diff = {Math.Abs((TestMethod(value, value)) - result1):0.####}");
    Console.WriteLine();
}

double TestMethod(double x, double y)
{
    return (x + y) / 5;
}

class TrainingSet
{
    public double[] Input;
    public double Target;

    public TrainingSet(double[] input, double target)
    {
        Input = input;
        Target = target;
    }
}
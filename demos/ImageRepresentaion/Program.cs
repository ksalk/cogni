using Cogni.MultilayerPerceptronNetwork;

var mlp = new MultilayerPerceptronNetwork()
    .AddInputLayer(2)
    .AddHiddenLayer(8)
    .AddHiddenLayer(8)
    .AddOutputLayer(1);

var result = mlp.Predict(new double[] { 0.1, 0.4 });
Console.WriteLine($"MLP Result = {string.Join(", ", result)}");
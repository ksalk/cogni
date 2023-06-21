using Cogni.MultilayerPerceptron;
using Cogni.MultilayerPerceptron.Builder;
using Cogni.MultilayerPerceptron.ActivationFunctions;
using System.Drawing;

var mlp = CreateMlpNetwork();

var rand = new Random();
TrainNetworkOnTestMethod(mlp, 1_000_000, rand);

for (double i = 0.0; i <= 1.0; i += 0.05)
{
    var result = mlp.PredictFor(i, i)[0];

    Console.WriteLine($"For (x, y): ({i:0.##}, {i:0.##})");
    Console.WriteLine($"Predicted: {result:0.####}");
    Console.WriteLine($"Actual: {TestMethod(i, i):0.####}");
    Console.WriteLine($"Diff: {Math.Abs(TestMethod(i, i) - result):0.####}");
    Console.WriteLine("--------------------------------------------------");
}

double GetPixelColor(int x, int y, Bitmap bmp)
{
    var color = bmp.GetPixel(x, y);
    return (double)(color.R) / 255;
}

MultilayerPerceptron CreateMlpNetwork()
{
    return MultilayerPerceptronBuilder.CreateNetwork()
        .AddLayer(2)
        .AddLayer(8)
        .AddLayer(8)
        .AddLayer(1)
        .WithBias()
        .WithActivationFunction<SigmoidFunction>()
        .Build();
}

List<ImageTrainingSet> GetTrainingSetsFromImage(string imagePath)
{
    var image = new Bitmap(imagePath);
    var result = new List<ImageTrainingSet>();

    for (int i = 0; i < image.Width; i++)
    {
        for (int j = 0; j < image.Height; j++)
        {
            var xNormalized = (double)i / image.Width;
            var yNormalized = (double)j / image.Height;
            var pixelColor = GetPixelColor(i, j, image);
            result.Add(new ImageTrainingSet(xNormalized, yNormalized, pixelColor));
        }
    }

    return result;
}

void TrainNetworkOnTestMethod(MultilayerPerceptron network, int iterations, Random r)
{
    Console.WriteLine($"Started training the network over {iterations} iterations");
    int updateInterval = iterations / 20;
    for (int i = 0; i < iterations; i++)
    {
        var x = r.NextDouble();
        var y = r.NextDouble();

        var result = TestMethod(x, y);

        network.Train(new double[] { x, y }, new double[] { result });
        if(i % updateInterval == 1) {
            Console.WriteLine($"{100 * (double)i / (double)iterations:0} %");
        }
    }
    
    Console.WriteLine($"Finished training the network");
}

void TrainNetwork(MultilayerPerceptron network, List<ImageTrainingSet> trainingData, int iterations, string imageName)
{
    var rand = new Random();
    for (int i = 0; i < iterations; i++)
    {
        var dataIndex = rand.Next(trainingData.Count);

        mlp.Train(trainingData[dataIndex].Input, trainingData[dataIndex].Target);

        if (i % 200_000 == 0)
        {
            var newImagePath = @$"C:\dev\cogni_resource\{imageName}_generated_{i:0000000}.bmp";
            WriteGeneratedImageToFile(mlp, newImagePath, 100, 100);
        }
    }
}

void WriteGeneratedImageToFile(MultilayerPerceptron network, string filePath, int width, int height)
{
    var imageData = GenerateImageFromNetwork(mlp, width, height);
    var image = new Bitmap(width, height);
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int pixelColor = (int)(imageData[i, j] * 255);
            image.SetPixel(i, j, Color.FromArgb(pixelColor, pixelColor, pixelColor));
        }
    }

    image.Save(filePath);
}

double[,] GenerateImageFromNetwork(MultilayerPerceptron network, int width, int height)
{
    var imageData = new double[width, height];
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            var xNormalized = (double)i / width;
            var yNormalized = (double)j / height;
            var pixelColorNormalized = mlp.PredictFor(xNormalized, yNormalized)[0];
            imageData[i, j] = pixelColorNormalized;
        }
    }

    return imageData;
}

double TestMethod(double x, double y)
{
    return (x + y) / 5;
}

class ImageTrainingSet
{
    public double[] Input;
    public double[] Target;

    public ImageTrainingSet(double x, double y, double target)
    {
        Input = new[] { x, y };
        Target = new[] { target };
    }

    public override string? ToString()
    {
        return $"({Input[0]:0.####}, {Input[1]:0.####}) => {Target[0]:0.####}";
    }
}
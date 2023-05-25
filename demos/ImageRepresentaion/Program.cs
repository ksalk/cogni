using Cogni.MultilayerPerceptronNetwork;
using System.Drawing;

var mlp = CreateMlpNetwork();

var imageName = "lines";
var imagePath = @$"C:\dev\cogni_resource\{imageName}.bmp";

if (!File.Exists(imagePath))
{
    throw new FileNotFoundException("You've passed wrong file path dummy...");
}

var trainingData = GetTrainingSetsFromImage(imagePath);

TrainNetwork(mlp, trainingData, 20_000_000);

var newImagePath = @$"C:\dev\cogni_resource\{imageName}_generated.bmp";
WriteGeneratedImageToFile(mlp, newImagePath, 100, 100);

Console.WriteLine("Finished generating file!");

double GetPixelColor(int x, int y, Bitmap bmp)
{
    var color = bmp.GetPixel(x, y);
    return (double)(color.R) / 255;
}

MultilayerPerceptronNetwork CreateMlpNetwork()
{
    return new MultilayerPerceptronNetwork()
        .AddInputLayer(2)
        .AddHiddenLayer(8)
        .AddHiddenLayer(8)
        .AddOutputLayer(1);
}

List<ImageTrainingSet> GetTrainingSetsFromImage(string imagePath)
{
    var image = new Bitmap(imagePath);
    var result = new List<ImageTrainingSet>();

    for(int i = 0; i < image.Width; i++)
    {
        for(int j = 0; j < image.Height; j++)
        {
            var xNormalized = (double)i / image.Width;
            var yNormalized = (double)j / image.Height;
            var pixelColor = GetPixelColor(i, j, image);
            result.Add(new ImageTrainingSet(xNormalized, yNormalized, pixelColor));
        }
    }
   
    return result;
}

void TrainNetwork(MultilayerPerceptronNetwork network, List<ImageTrainingSet> trainingData, int iterations)
{
    var rand = new Random();
    for (int i = 0; i < iterations; i++)
    {
        var dataIndex = rand.Next(trainingData.Count);

        mlp.Train(trainingData[dataIndex].Input, trainingData[dataIndex].Target);

        if(i % 200_000 == 0)
        {
            var newImagePath = @$"C:\dev\cogni_resource\{imageName}_generated_{i:0000000}.bmp";
            WriteGeneratedImageToFile(mlp, newImagePath, 100, 100);
        }
    }
}

void WriteGeneratedImageToFile(MultilayerPerceptronNetwork network, string filePath, int width, int height)
{
    var imageData = GenerateImageFromNetwork(mlp, width, height);
    var image = new Bitmap(width, height);
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < height; j++)
        {
            int pixelColor = (int)(imageData[i, j] * 255);
            image.SetPixel(i, j, Color.FromArgb(pixelColor, pixelColor, pixelColor));
        }
    }

    image.Save(filePath);
}

double[,] GenerateImageFromNetwork(MultilayerPerceptronNetwork network, int width, int height)
{
    var imageData = new double[width, height];
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            var xNormalized = (double)i / width;
            var yNormalized = (double)j / height;
            var pixelColorNormalized = mlp.Predict(new double[] { xNormalized, yNormalized })[0];
            imageData[i, j] = pixelColorNormalized;
        }
    }

    return imageData;
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
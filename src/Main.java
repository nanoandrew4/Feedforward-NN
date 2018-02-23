import java.util.List;

public class Main {
    public static void main(String... args) {

        long start = System.currentTimeMillis();

        System.out.println("Initializing NN...");
        NeuralNetwork nn = new NeuralNetwork(784, 256, 10);
        System.out.println("NN initialized, reading in test data...");

        int[] labels = MNISTReader.getLabels("training/labels.idx1-ubyte");
        List<int[][]> images = MNISTReader.getImages("training/nums.idx3-ubyte");

        System.out.println("Test data read, preparing data for NN...");

        float[][] expectedOut = new float[labels.length][10];
        for (int i = 0; i < labels.length; i++)
            for (int j = 0; j < expectedOut[0].length; j++)
                if (j == labels[i])
                    expectedOut[i][j] = 1f;

        float[][] numberData = convertImages(images);

        System.out.println("Data ready, starting training... This may take a long time");

        nn.train(numberData, expectedOut);

        System.out.println("Training complete");

        System.out.println("Loading testing data for test run");

        labels = MNISTReader.getLabels("testing/labels.idx1-ubyte");
        numberData = convertImages(MNISTReader.getImages("testing/nums.idx3-ubyte"));

        System.out.println("Loaded test data, starting test run... This may take a long time");

        float accuracy = nn.test(numberData, labels);

        System.out.println("Testing finished. Network accuracy: " + accuracy + "%");
        System.out.println("Total time taken: " + (float)(System.currentTimeMillis() - start) / 60000f);
    }

    private static float[][] convertImages(List<int[][]> images) {
        float[][] inputs = new float[images.size()][images.get(0).length * images.get(0).length];
        for (int i = 0; i < images.size(); i++)
            for (int c = 0; c < images.get(0).length; c++)
                for (int r = 0; r < images.get(0).length; r++)
                    inputs[i][(c * images.get(0).length) + r] = (float)images.get(i)[c][r] / 255f;
        return inputs;
    }
}

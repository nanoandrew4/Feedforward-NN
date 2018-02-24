package main.java;

import java.util.List;

import static main.java.IOHandler.*;

public class Main {
    public static void main(String... args) {

        long start = System.currentTimeMillis();

        NeuralNetwork nn;

        if (args.length > 1) {
            float LEARNING_RATE = Float.valueOf(args[0]);
            int[] neuronsPerLayer = new int[args.length - 1];
            for (int i = 1; i < args.length; i++)
                neuronsPerLayer[i - 1] = Integer.valueOf(args[i]);
            nn = new NeuralNetwork(LEARNING_RATE, neuronsPerLayer);
        } else {
            nn = (NeuralNetwork) deserialize("ffnn.nn");
        }

        ClassLoader classLoader = Main.class.getClassLoader();

        int[] labels = deserializeLabels(classLoader.getResourceAsStream("training/data.la"));
        float[][] numberData = deserializeNums(classLoader.getResourceAsStream("training/data.nu"));

        float[][] expectedOut = new float[labels.length][10];
        for (int i = 0; i < labels.length; i++)
            for (int j = 0; j < expectedOut[0].length; j++)
                if (j == labels[i])
                    expectedOut[i][j] = 1f;

        nn.train(numberData, expectedOut);

        labels = deserializeLabels(classLoader.getResourceAsStream("testing/data.la"));
        numberData = deserializeNums(classLoader.getResourceAsStream("testing/data.nu"));

        float accuracy = nn.test(numberData, labels);

        if (accuracy > 90.0)
            serialize("ffnn.nn", nn);

        System.out.println("Testing finished. Network accuracy: " + accuracy + "%");
        System.out.println("Total time taken: " + (float) (System.currentTimeMillis() - start) / 60000f + " minutes");
    }

    private static float[][] convertImages(List<int[][]> images) {
        float[][] inputs = new float[images.size()][images.get(0).length * images.get(0).length];
        for (int i = 0; i < images.size(); i++)
            for (int c = 0; c < images.get(0).length; c++)
                for (int r = 0; r < images.get(0).length; r++)
                    inputs[i][(c * images.get(0).length) + r] = (float) images.get(i)[c][r] / 255f;
        return inputs;
    }
}
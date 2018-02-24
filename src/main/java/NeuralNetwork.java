package main.java;

import java.io.Serializable;

public class NeuralNetwork implements Serializable {

    private float[][] activ;

    // layer - previous layer neuron - current layer neuron
    private float[][][] weights;
//    private float[][][] biases;

    private float LEARNING_RATE = 0.35f;
    private int MAX_ITERATIONS = 10;

    NeuralNetwork(float LEARNING_RATE, int[] neuronsPerLayer) {
        Logger.print("Initializing NN... ");

        this.LEARNING_RATE = LEARNING_RATE;

        if (neuronsPerLayer.length < 2) {
            System.out.println("Network must be composed of at least 2 layers");
            return;
        }

        activ = new float[neuronsPerLayer.length][];
        weights = new float[neuronsPerLayer.length - 1][][];
//        biases = new float[neuronsPerLayer.length - 1][][];

        for (int i = 0; i < neuronsPerLayer.length; i++)
            activ[i] = new float[neuronsPerLayer[i]];

        for (int i = 0; i < neuronsPerLayer.length - 1; i++) {
            weights[i] = new float[neuronsPerLayer[i]][neuronsPerLayer[i + 1]];
//            biases[i] = new float[neuronsPerLayer[i]][neuronsPerLayer[i + 1]];

            System.out.println("Layer: " + i + " with weights: " + neuronsPerLayer[i] + " -> " + neuronsPerLayer[i + 1]);

            for (int j = 0; j < neuronsPerLayer[i]; j++) {
                for (int k = 0; k < neuronsPerLayer[i + 1]; k++) {
                    weights[i][j][k] = (float) (Math.random() - 0.5) * 2;
//                    biases[i][j][k] = (float) (Math.random() - 0.5) * 2;
                }
            }
        }

        Logger.print("Done\n");
    }

    private NeuralNetwork(float LEARNING_RATE, float[][] activ, float[][][] weights) {
        this.LEARNING_RATE = LEARNING_RATE;
        this.activ = activ;
        this.weights = weights;
    }

    private float sigmoid(float f) {
        return (float) (1f / (1f + Math.exp(-f)));
    }

    private float dSigmoid(float f) {
        return f * (1 - f);
    }

    void forward(float[] inputs) {
        System.arraycopy(inputs, 0, activ[0], 0, inputs.length);

        for (int l = 1; l < activ.length; l++) {
            for (int n = 0; n < activ[l].length; n++) {
                for (int pn = 0; pn < activ[l - 1].length; pn++) {
                    activ[l][n] += activ[l - 1][pn] * weights[l - 1][pn][n]/* + biases[l - 1][pn][n]*/;
                }
                activ[l][n] = sigmoid(activ[l][n]);
            }
        }
    }

    private float cost(float[] inputs, float[] expectedOut) {
        float cost = 0f;
        forward(inputs);

        for (int n = 0; n < activ[activ.length - 1].length; n++)
            cost += Math.pow(activ[activ.length - 1][n] - expectedOut[n], 2);
        return cost / 2;
    }

    private float dCost(int neuron, float[] expected) {
        return activ[activ.length - 1][neuron] - expected[neuron];
    }

    private void gradientDescent(float[] expected) {
        int maxSize = activ[0].length;
        for (int i = 1; i < activ.length; i++)
            maxSize = Math.max(maxSize, activ[i].length);

        float[] dweights = new float[maxSize];
        // float[][][] dbiases = new float[biases.length][biases[0].length][biases[0][0].length];

        for (int n = 0; n < activ[activ.length - 1].length; n++) {
            dweights[n] = LEARNING_RATE * dSigmoid(activ[activ.length - 1][n]) * dCost(n, expected);

            for (int pn = 0; pn < activ[activ.length - 2].length; pn++) {
                weights[weights.length - 1][pn][n] -= dweights[n] * activ[activ.length - 2][pn];
            }
        }

        for (int l = activ.length - 3; l >= 0; l--) {
            for (int n = 0; n < activ[l + 1].length; n++) {
                float oldDWeights = dweights[n];
                dweights[n] = 0;
                for (int nn = 0; nn < activ[l + 2].length; nn++)
                    dweights[n] = LEARNING_RATE * dSigmoid(activ[l + 1][n]) * weights[l + 1][n][nn] * oldDWeights;

                for (int pn = 0; pn < activ[l].length; pn++)
                    weights[l][pn][n] -= dweights[n] * activ[l][pn];
            }
        }
    }

    void train(float[][] inputs, float[][] outputs) {
        if (inputs.length != outputs.length) {
            System.out.println("Must input as many inputs as outputs");
            return;
        }

        Logger.print("Starting training... This may take a long time\n");
        System.out.println("Training network on " + inputs.length + " samples");
        System.out.println("Total iterations on this training data: " + MAX_ITERATIONS);

        for (int iters = 0; iters < MAX_ITERATIONS; iters++) {
            for (int i = 0; i < inputs.length; i++) {
                forward(inputs[i]);
                gradientDescent(outputs[i]);
            }
        }
        Logger.print("Training finished\n");
    }

    float test(float[][] testInputs, int[] expectedOut) {
        float hits = 0;

        Logger.print("Starting testing...\n");
        System.out.println("Testing network on " + testInputs.length + " samples");

        for (int i = 0; i < testInputs.length; i++) {
            forward(testInputs[i]);
            int index = 0;
            for (int j = 1; j < activ[activ.length - 1].length; j++)
                if (activ[activ.length - 1][j] > activ[activ.length - 1][index])
                    index = j;

            if (index == expectedOut[i])
                hits++;
        }

        Logger.print("Testing finished\n");
        return (hits / testInputs.length) * 100f;
    }

    float[] getResults() {
        return activ[activ.length - 1];
    }

    NeuralNetwork copy() {
        return new NeuralNetwork(LEARNING_RATE, activ, weights);
    }
}

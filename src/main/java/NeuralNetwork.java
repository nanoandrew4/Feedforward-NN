package main.java;

import java.io.Serializable;

public class NeuralNetwork implements Serializable {

    /*
     * Stores all the activations in the neurons that compose the network.
     * In essence, these are the neurons.
     */
    private double[][] activ;

    /*
     * Stores all the weights for this given network.
     * In essence, these are the synapses connecting each layer of neurons to the next.
     * Stored as an array of 2D arrays, which connect the neurons in two layers to each other.
     * weights[layer][neuron][nextLayerNeuron]
     */
    private double[][][] weights;

    // Rate at which the network learns. Values > 0.35 seem to degrade accuracy a lot, so small steps are optimal.
    private double LEARNING_RATE = 0.3f;

    /*
     * Iterations to train the network on. Although the same data is used on each iteration, it significantly improves
     * accuracy to train it multiple times, so it has multiple chances to adjust.
     */
    private int MAX_ITERATIONS = 4;

    /**
     * Initializes Neural Network according to passed in parameters, and randomizes all the weights.
     *
     * @param LEARNING_RATE   Rate at which the network learns ( < 0.4 recommended)
     * @param neuronsPerLayer Number of neurons per layer that will make up the network
     */
    NeuralNetwork(double LEARNING_RATE, int[] neuronsPerLayer) {
        Logger.print("Initializing NN... Structure: ");
        for (int i = 0; i < neuronsPerLayer.length; i++)
            Logger.print(neuronsPerLayer[i] + (i == neuronsPerLayer.length - 1 ? " | " : " -> "));

        this.LEARNING_RATE = LEARNING_RATE;

        if (neuronsPerLayer.length < 2) {
            System.out.println("Network must be composed of at least 2 layers");
            return;
        }

        activ = new double[neuronsPerLayer.length][];
        weights = new double[neuronsPerLayer.length - 1][][];

        for (int i = 0; i < neuronsPerLayer.length; i++)
            activ[i] = new double[neuronsPerLayer[i]];

        for (int i = 0; i < neuronsPerLayer.length - 1; i++) {
            weights[i] = new double[neuronsPerLayer[i]][neuronsPerLayer[i + 1]];

            for (int j = 0; j < neuronsPerLayer[i]; j++)
                for (int k = 0; k < neuronsPerLayer[i + 1]; k++)
                    weights[i][j][k] = (Math.random() - 0.5);
        }

        Logger.print("Done\n");
    }

    private NeuralNetwork(double LEARNING_RATE, double[][] activ, double[][][] weights) {
        this.LEARNING_RATE = LEARNING_RATE;
        this.activ = activ;
        this.weights = weights;
    }

    private double sigmoid(double f) {
        return (1d / (1d + Math.exp(-f)));
    }

    // Derivative of the sigmoid function
    private double dSigmoid(double f) {
        return f * (1 - f);
    }

    /**
     * Forward propagation through the network. Takes a set of inputs, and sets them as the input layer
     * activation values. Proceeds to iteratively calculate the activation values for each layer, finishing at the
     * output layer.
     *
     * @param inputs Array of data to be input (should be the same size as the first layer of the network)
     */
    void forward(double[] inputs) {

        // Zero array
        for (int l = 0; l < activ.length; l++)
            for (int n = 0; n < activ[l].length; n++)
                activ[l][n] = 0;

        System.arraycopy(inputs, 0, activ[0], 0, inputs.length);

        for (int l = 1; l < activ.length; l++) {
            for (int n = 0; n < activ[l].length; n++) {
                for (int pn = 0; pn < activ[l - 1].length; pn++)
                    activ[l][n] += activ[l - 1][pn] * weights[l - 1][pn][n]/* + biases[l - 1][pn][n]*/;
                activ[l][n] = sigmoid(activ[l][n]); // Apply sigmoid to sum of previous layer activations
            }
        }
    }

    /**
     * @param input
     * @param expectedOut
     * @return
     */
    private double cost(double[] input, double[] expectedOut) {
        double cost = 0f;
        forward(input);

        for (int n = 0; n < activ[activ.length - 1].length; n++)
            cost += Math.pow(activ[activ.length - 1][n] - expectedOut[n], 2);
        return cost / 2;
    }

    /**
     * Gradient descent algorithm to back propagate through the network.
     * In essence, this function is what makes the network 'learn', by adjusting the weights connecting the various
     * layers.
     *
     * @param expected Values to be expected in last layer of activations array after forward propagation
     */
    private double[][] gradientDescent(double[] input, double[] expected) {
        forward(input);

        /*
         * Initialize array that will hold the changes to be applied at each layer of the network
         * Read as "delta weights"
         */
        double[][] dweights = new double[weights.length][];
        dweights[dweights.length - 1] = new double[activ[activ.length - 1].length];

        // Adjust second-last-to-last layer weights, using dCost
        for (int n = 0; n < activ[activ.length - 1].length; n++) {
            dweights[dweights.length - 1][n] = LEARNING_RATE * dSigmoid(activ[activ.length - 1][n]) * (activ[activ.length - 1][n] - expected[n]);
            for (int pn = 0; pn < activ[activ.length - 2].length; pn++)
                weights[weights.length - 1][pn][n] -= dweights[dweights.length - 1][n] * activ[activ.length - 2][pn];
        }

        // Adjust all other weights in the network, using previous dweights values
        for (int l = weights.length - 2; l >= 0; l--) {
            dweights[l] = new double[activ[l + 1].length];
            for (int n = 0; n < activ[l + 1].length; n++) {
                // Recalculate delta for connection between 'n' and all neurons on the next layer ('nn')
                for (int nn = 0; nn < activ[l + 2].length; nn++)
                    dweights[l][n] += LEARNING_RATE * dSigmoid(activ[l + 1][n]) * weights[l + 1][n][nn] * dweights[l + 1][nn];
                // Apply delta to appropriate weights
                for (int pn = 0; pn < activ[l].length; pn++)
                    weights[l][pn][n] -= dweights[l][n] * activ[l][pn];
            }
        }

        return dweights;
    }

    /**
     * Trains the network given an array of inputs and an array of expected outputs. For each element, calls forward()
     * with a given input, and backpropagates with its respective expected output.
     * <p>
     * Will iterate MAX_ITERATIONS times through the same data, so the network can learn better.
     *
     * @param inputs  Array of inputs. Each sub-array should contain the same number of elements as the input layer
     * @param outputs Array of expected outputs. Each sub-array should contain the same number of elements as the
     *                output layer
     */
    void train(double[][] inputs, double[][] outputs) {
        if (inputs.length != outputs.length) {
            System.out.println("Must input as many inputs as outputs");
            return;
        }

        Logger.print("Starting training... This may take a long time\n");
        System.out.println("Training network on " + inputs.length + " samples");
        System.out.println("Total iterations on this training data: " + MAX_ITERATIONS);

        for (int iters = 0; iters < MAX_ITERATIONS; iters++)
            for (int i = 0; i < inputs.length; i++)
                gradientDescent(inputs[i], outputs[i]);
        Logger.print("Training finished\n");
    }

    /**
     * Tests the network given an array of inputs and an array of expected outputs. For each element, calls forward()
     * with a given input and checks that the last right neuron in the output layer was activated.
     *
     * @param testInputs  Array of inputs. Each sub-array should contain the same number of elements as the input layer
     * @param expectedOut Array of expected outputs. Each sub-array should contain the same number of elements as the
     *                    output layer
     * @return Float between 0-100 representing the accuracy of the network on the test data
     */
    double test(double[][] testInputs, int[] expectedOut) {

        gradientCheck(testInputs[0], desiredOutputLayerActiv(expectedOut[0], activ[activ.length - 1].length));

        double hits = 0;

        Logger.print("Starting testing... Testing network on " + testInputs.length + " samples | ");

        for (int i = 1; i < testInputs.length; i++) {
            forward(testInputs[i]);

            if (getOutputNeuron() == expectedOut[i])
                hits++;
        }

        Logger.print("Done\n");
        return (hits / testInputs.length) * 100f;
    }

    private void gradientCheck(double[] input, double[] output) {
        double e = Math.pow(1, -4);

        double[][] dweights = gradientDescent(input, output);
        double[][] dweightsApprox = new double[dweights.length][];

        for (int l = 0; l < dweights.length; l++) {
            dweightsApprox[l] = new double[dweights[l].length];
            for (int n = 0; n < activ[l + 1].length; n++) {
                for (int pn = 0; pn < activ[l].length; pn++) {

                    // Slight perturbation to weight
                    weights[l][pn][n] += e;
                    double loss1 = cost(input, output);

                    // Cancel previous perturbation and add another in opposite direction
                    weights[l][pn][n] -= 2 * e;
                    double loss2 = cost(input, output);

                    // Calculates what the gradient should be
                    dweightsApprox[l][n] = (loss1 - loss2) / (2 * e);
                    if (Math.abs(dweightsApprox[l][n] - dweights[l][n]) > Math.pow(1, -8))
                        System.out.println(Math.abs(dweightsApprox[l][n] - dweights[l][n]));

                    // Reset weight
                    weights[l][pn][n] += e;
                }
            }
        }
    }

    private double[] desiredOutputLayerActiv(int n, int size) {
        double[] d = new double[size];
        d[n] = 1.0;
        return d;
    }

    /**
     * Returns the index of the neuron on the output layer with the highest value
     *
     * @return Index corresponding to the neuron on the output layer with the highest value
     */
    private int getOutputNeuron() {
        int index = 0;
        // Searches through output layer for brightest neuron
        for (int j = 1; j < activ[activ.length - 1].length; j++)
            if (activ[activ.length - 1][j] > activ[activ.length - 1][index])
                index = j;
        return index;
    }

    String getResults() {
        StringBuilder str = new StringBuilder("[");
        for (int i = 0; i < activ[activ.length - 1].length; i++)
            str.append(activ[activ.length - 1][i]).append(i == activ[activ.length - 1].length - 1 ? "" : ", ");
        str.append("]");
        return str.toString();
    }
}
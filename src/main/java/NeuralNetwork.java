package main.java;

import java.io.Serializable;
import java.util.Arrays;

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

    // Rate at which the network learns. 0.3 seems to yield pretty good results.
    private double learningRate;

    /*
     * Iterations to train the network on. Although the same data is used on each iteration, it significantly improves
     * accuracy to train it multiple times, so it has multiple chances to adjust. However, too many iterations will
     * lead it to memorize the training data, and will generalize poorly.
     */
    private int iterations;

    /**
     * Initializes Neural Network according to passed in parameters, and randomizes all the weights.
     *
     * @param learningRate   Rate at which the network learns ( < 0.4 recommended)
     * @param neuronsPerLayer Number of neurons per layer that will make up the network
     */
    NeuralNetwork(double learningRate, int iterations, int[] neuronsPerLayer) {
        Logger.print("Initializing NN... Structure: ");
        for (int i = 0; i < neuronsPerLayer.length; i++)
            Logger.print(neuronsPerLayer[i] + (i == neuronsPerLayer.length - 1 ? " | " : " -> "));

        this.learningRate = learningRate;
        this.iterations = iterations;

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
    private void forward(double[] inputs) {

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
     * Computes and returns the cost of the network. This particular implementation sums the difference between the
     * output layer and the expected output layer squared. The higher the value, the worse the performance of the
     * network
     *
     * @param input Input to forward propagate with
     * @param expectedOut Expected output to compute cost
     * @return Cost of the network
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
     * @param input Input to perform forward propagation with
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

        // Adjust second-last-to-last layer weights
        for (int n = 0; n < activ[activ.length - 1].length; n++) {
            dweights[dweights.length - 1][n] = learningRate * dSigmoid(activ[activ.length - 1][n]) * (activ[activ.length - 1][n] - expected[n]);
            for (int pn = 0; pn < activ[activ.length - 2].length; pn++)
                weights[weights.length - 1][pn][n] -= dweights[dweights.length - 1][n] * activ[activ.length - 2][pn];
        }

        // Adjust all other weights in the network, using previous dweights values
        for (int l = weights.length - 2; l >= 0; l--) {
            dweights[l] = new double[activ[l + 1].length];
            for (int n = 0; n < activ[l + 1].length; n++) {
                // Recalculate delta for connection between 'n' and all neurons on the next layer ('nn')
                for (int nn = 0; nn < activ[l + 2].length; nn++)
                    dweights[l][n] += learningRate * dSigmoid(activ[l + 1][n]) * weights[l + 1][n][nn] * dweights[l + 1][nn];
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
     * Will iterate iterations times through the same data, so the network can learn better.
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

        Logger.print("Training network over " + iterations + " iterations on " + inputs.length + " samples... ");
        Spinner.spin();

        for (int iters = 0; iters < iterations; iters++)
            for (int i = 0; i < inputs.length; i++)
                gradientDescent(inputs[i], outputs[i]);

        Spinner.spin();
        Logger.print("Done\n");
    }

    /**
     * Tests the network given an array of inputs and an array of expected outputs. For each element, calls forward()
     * with a given input and checks that the last right neuron in the output layer was activated.
     *
     * @param testInputs  Array of inputs. Each sub-array should contain the same number of elements as the input layer
     * @param expectedOut Array of expected outputs. Each sub-array should contain the same number of elements as the
     *                    output layer
     */
    void test(double[][] testInputs, double[][] expectedOut) {

        gradientCheck(testInputs[0], expectedOut[0]);

        double hits = 0;

        Logger.print("Starting testing... Testing network on " + testInputs.length + " samples... ");
        Spinner.spin();

        for (int i = 1; i < testInputs.length; i++)
            if (Arrays.equals(expectedOut[i], getOutputLayerActiv(testInputs[i], 1)))
                hits++;

        Spinner.spin();
        Logger.print("Done\n");
        System.out.println("Network accuracy: " + ((hits / testInputs.length) * 100f) + "%");
    }

    /**
     * Compare the gradient for a set of input values to approximations. Each weight is perturbed positively and
     * negatively, and the cost calculated after each perturbation. The slope of the line connecting the two costs
     * should be within Math.pow(1, -8) of the gradient computed using gradient descent of that same weight.
     * The perturbation should be a small number.
     * If the difference between the actual and approximate gradients is more than Math.pow(1, -8), there might be
     * an error in the gradient descent algorithm.
     *
     * @param input Input set to check gradient descent algorithm with
     * @param output Expected output for the 'input' set, to calculate gradient and cost
     */
    private void gradientCheck(double[] input, double[] output) {
        Logger.print("Checking gradient descent algorithm quality... ");
        Spinner.spin();

        double e = Math.pow(1, -4);

        double[][] dweights = gradientDescent(input, output);
        double[][] dweightsApprox = new double[dweights.length][];

        boolean pass = true;

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
                        pass = false;

                    // Reset weight
                    weights[l][pn][n] += e;
                }
            }
        }

        Spinner.spin();
        Logger.print(pass ? "Good\n" : "Bad\n");
    }

    /**
     * Returns an approximation of the output layer, with 'numOfActivations' neurons set to 1.0, and the rest to 0.0
     *
     * @param input Input to forward propagate with
     * @param numOfActivations Number of neurons that should be activated (same number as in your expected output)
     * @return Array representing an approximation of the last layer
     */
    private double[] getOutputLayerActiv(double[] input, int numOfActivations) {
        forward(input);

        double[] approxOut = new double[activ[activ.length - 1].length];
        int[] indices = new int[numOfActivations];
        int ll = activ.length - 1; // Last layer

        for (int n = 0; n < activ[activ.length - 1].length; n++) {
            int activNeuron = -1;
            for (int i = numOfActivations - 1; i >= 0; i--) {
                if (activ[ll][n] <= activ[ll][indices[i]])
                    break;
                else
                    activNeuron = i;
            }
            if (activNeuron != -1)
                indices[activNeuron] = n;
        }
        
        for (int i : indices)
            approxOut[i] = 1d;
        return approxOut;
    }

    String getOutputLayerStr() {
        StringBuilder str = new StringBuilder("[");
        for (int i = 0; i < activ[activ.length - 1].length; i++)
            str.append(activ[activ.length - 1][i]).append(i == activ[activ.length - 1].length - 1 ? "" : ", ");
        str.append("]");
        return str.toString();
    }
}
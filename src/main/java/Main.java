package main.java;

import java.util.Scanner;

public class Main {

    private static String layout = "784 -> 128 -> 10", iters = "4", learningRate = "0.3",
            trainingInFile = "training/nums.tr", trainingOutFile = "training/labels.tr",
            testingInFile = "testing/nums.te", testingOutFile = "testing/labels.te";

    private static double[][] trainingIn = {}, testingIn = {}, trainingOut = {}, testingOut = {};

    private static Scanner in = new Scanner(System.in);

    public static void main(String... args) {

        printNetworkInfo();

        while (true) {
            System.out.println("Hyper-parameter setup");
            System.out.println("1. Modify network layout");
            System.out.println("2. Modify iterations");
            System.out.println("3. Modify learning rate");
            System.out.println("8. Start using this network");
            System.out.println("9. Print network layout");
            System.out.println("0. Exit\n");

            System.out.print(">> ");
            int opt = in.nextInt();
            in.nextLine();

            switch (opt) {
                case 1:
                    System.out.print("Enter number of neurons per layer, separated by \'->\': ");
                    layout = in.nextLine().trim();
                    break;
                case 2:
                    System.out.print("Enter number of iterations to perform on training set: ");
                    iters = in.nextLine().trim();
                    break;
                case 3:
                    System.out.print("Enter desired learning rate: ");
                    learningRate = in.nextLine().trim();
                    break;
                case 8:
                    System.out.println();
                    startNetwork();
                    break;
                case 9:
                    System.out.println("\n");
                    printNetworkInfo();
                    break;
                case 0:
                    return;
            }

            System.out.println("\n");
        }
    }
    
    private static void startNetwork() {
        NeuralNetwork nn = new NeuralNetwork(Double.valueOf(learningRate), Integer.valueOf(iters), getLayout(layout));
        String str;

        while (true) {
            System.out.println("\nNetwork options: ");
            System.out.println("1. Train and test");
            System.out.println("2. Train");
            System.out.println("3. Test");
            System.out.println("4. Modify training sets");
            System.out.println("5. Modify testing sets");
            System.out.println("6. (re)Load data sets");
            System.out.println("0. Exit\n");

            System.out.print(">> ");
            int opt = in.nextInt();
            in.nextLine();
            System.out.println();

            switch (opt) {
                case 1:
                    if (trainingIn.length == 0 || trainingOut.length == 0 || testingIn.length == 0 || testingOut.length == 0)
                        loadData("t");
                    System.out.println("Training data files: \"" + trainingInFile + "\" - \"" + trainingOutFile + "\"");
                    System.out.println("Testing data files: \"" + testingInFile + "\" - \"" + testingOutFile + "\"");
                    nn.train(trainingIn, trainingOut);
                    nn.test(testingIn, testingOut);
                    break;
                case 2:
                    if (trainingIn.length == 0 || trainingOut.length == 0)
                        loadData("tr");
                    System.out.println("Training data files: \"" + trainingInFile + "\" - \"" + trainingOutFile + "\"");
                    nn.train(trainingIn, trainingOut);
                    break;
                case 3:
                    if (testingIn.length == 0 || testingOut.length == 0)
                        loadData("te");
                    System.out.println("Testing data files: \"" + testingInFile + "\" - \"" + testingOutFile + "\"");
                    nn.test(testingIn, testingOut);
                    break;
                case 4:
                    requestFiles("training");
                    str = in.nextLine().trim();
                    trainingInFile = str.split(" ")[0];
                    trainingOutFile = str.split(" ")[1];
                    break;
                case 5:
                    requestFiles("testing");
                    str = in.nextLine().trim();
                    Main.testingInFile = str.split(" ")[0];
                    Main.testingOutFile = str.split(" ")[1];
                    break;
                case 6:
                    System.out.println("Enter \'t\' to load both training and testing data, \'tr\' to load training data" +
                            "only, or \'te\' to load testing data only");
                    String dataOpt = in.nextLine().trim().toLowerCase();
                    loadData(dataOpt);
                    break;
                case 0:
                    return;
            }
        }
    }

    private static void printNetworkInfo() {
        System.out.println("Network information:\n");
        System.out.println("Network layout: \"" + layout + "\"");
        System.out.println("Serialized training files (input, output): \"" + trainingInFile + "\" - \"" + trainingOutFile + "\"");
        System.out.println("Serialized testing files (input, output): \"" + testingInFile + "\" - \"" + testingOutFile + "\"");
        System.out.println("Iterations through the training set: " + iters);
        System.out.println("Learning rate of the network: " + learningRate);
        System.out.println();
    }

    private static void requestFiles(String fileType) {
        System.out.println("Enter the filename of the serialized " + fileType
                + " input, and the filename of serialized expected " + fileType + " output");
        System.out.println("The input and output data must be serialized as a double[][]");
    }

    private static int[] getLayout(String layoutStr) {
        String[] str = layoutStr.split("->");
        int[] layout = new int[str.length];

        for (int l = 0; l < str.length; l++)
            layout[l] = Integer.valueOf(str[l].trim());

        return layout;
    }

    private static void loadData(String dataToLoad) {
        Logger.print("Loading data... ");
        Spinner.spin();
        if ("t".equals(dataToLoad) || "tr".equals(dataToLoad)) {
            trainingIn = IOHandler.deserialize2DDoubleArr(trainingInFile);
            trainingOut = IOHandler.deserialize2DDoubleArr(trainingOutFile);
        }
        if ("t".equals(dataToLoad) || "te".equals(dataToLoad)) {
            testingIn = IOHandler.deserialize2DDoubleArr(Main.testingInFile);
            testingOut = IOHandler.deserialize2DDoubleArr(Main.testingOutFile);
        }
        Spinner.spin();
        Logger.print("Done\n");
    }
}

/**
 * Simple spinning icon made out of characters. Used while program is running long computations, so the user
 * feels like the program is running something vs just being stuck. Pretty meaningless, but hey, looks good!
 */
class Spinner extends Thread {
    private static Thread t;
    private static Spinner s;

    private char[] chars = {'|', '/', '-', '\\'};
    private boolean spinning = true;

    @Override
    public void run() {
        for (int i = 0; spinning; i++) {
            System.out.print(chars[i % chars.length]);
            try {
                sleep(500);
            } catch (InterruptedException e) {e.printStackTrace();}
            System.out.print("\b \b");
        }
    }

    private void end() {
        spinning = false;
        try {
            t.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void spin() {
        if (t == null || !s.spinning) {
            t = new Thread(s = new Spinner());
            t.setDaemon(true);
            t.start();
        } else
            s.end();
    }
}
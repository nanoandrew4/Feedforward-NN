package main.java;

import java.io.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import static main.java.Logger.print;

/*
 * Serialized MNIST dataset was initially read and processed thanks to jeffgriffith.
 * See his MNISTReader here: https://github.com/jeffgriffith/mnist-reader
 */
public class IOHandler {

    /**
     * Compresses and serializes an object, and outputs it to a file called 'filename'.
     *
     * @param filename Desired output file name
     * @param o        Object to serialize
     */
    protected static void serialize(String filename, Object o) {
        print("Serializing -> " + filename + "... ");
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(filename)));
            oos.writeObject(o);
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        print("Done\n");
    }

    /**
     * See deserializeLabels(InputStream is).
     *
     * @param path Path to serialized file
     * @return 1D integer array stored in file at 'path', representing the labels for a set of data
     */
    protected static int[] deserializeLabels(String path) {
        return (int[]) deserialize(path);
    }

    /**
     * Converts returned object to a 1D int array and returns it.
     * Used to deserialize the labels from MNIST dataset (they have been converted to a 1D array and serialized)
     * The values stored in the array represent the labels for their respective digits.
     *
     * @param is Input stream to read file from
     * @return 1D integer array read from 'is', representing the labels for a set of data
     */
    protected static int[] deserializeLabels(InputStream is) {
        return (int[]) deserialize(is);
    }

    /**
     * See deserializeNums(InputStream is).
     *
     * @param path Path to serialized file
     * @return 2D float array stored in file at 'path', representing the digit images
     */
    protected static double[][] deserializeNums(String path) {
        return (double[][]) deserialize(path);
    }

    /**
     * Converts returned object to a 2D float array and returns it.
     * Used to deserialize the digits from MNIST dataset (they have been converted to a 2D array and serialized)
     * The values stored in the file represent 8 bit color values divided by 255, read from left to right starting
     * at the top left corner. Each image is 28x28 pixels, which combine to make a 784 length 1D array.
     *
     * @param is Input stream to read file from
     * @return 2D float array read from 'is', representing the digit images
     */
    protected static double[][] deserializeNums(InputStream is) {
        return (double[][]) deserialize(is);
    }

    /**
     * See deserialize(InputStream is) below. Instead of loading data from a stream, this method takes a filename
     * and runs deserialize(InputStream is), returning the deserialized data as an Object type.
     *
     * @param path Path to file
     * @return Object stored in file at 'path'
     */
    protected static Object deserialize(String path) {
        try {
            return deserialize(new FileInputStream(path));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * Deserializes a file and returns the object contained within it. Primarily used to read NN instances which have
     * been serialized, as well as for reading in data which has been properly converted from its original format
     * though the serialization functions available in this program.
     *
     * @param is Input stream for serialized data. Useful to use with Class.class.getClassLoader().getResourceAsStream()
     * @return Object contained within the file
     */
    protected static Object deserialize(InputStream is) {
        print("Loading serialized data... ");
        try {
            ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(is));
            Object o = ois.readObject();
            ois.close();
            print("Done\n");
            return o;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }
}

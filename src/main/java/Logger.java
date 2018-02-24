package main.java;

public class Logger {
    private static boolean printStatus = true;

    static void disablePrintStatus() {
        printStatus = false;
    }

    static void print(String message) {
        if (printStatus)
            System.out.print(message);
    }
}

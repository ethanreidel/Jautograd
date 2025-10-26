package com.example.lib;

/** Tiny example to verify :app -> :lib wiring. Replace with your ML code. */
public class MathUtils {
    /** Returns the mean of an array; NaN for empty arrays. */
    public static double mean(double[] xs) {
        if (xs == null || xs.length == 0) return Double.NaN;
        double s = 0.0;
        for (double x : xs) s += x;
        return s / xs.length;
    }
}

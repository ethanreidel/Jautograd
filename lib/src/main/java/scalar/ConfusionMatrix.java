package scalar;

/**
 * Multi-class confusion matrix.
 *
 * <p>Stores counts for (actualClass, predictedClass) pairs.
 */
public final class ConfusionMatrix {

    private final int numClasses;
    // counts[actual][predicted]
    private final long[][] counts;

    /**
     * Creates a confusion matrix for the given number of classes.
     *
     * @param numClasses number of distinct classes (must be >= 2)
     */
    public ConfusionMatrix(int numClasses) {
        if (numClasses <= 1) {
            throw new IllegalArgumentException("numClasses must be >= 2, but was: " + numClasses);
        }
        this.numClasses = numClasses;
        this.counts = new long[numClasses][numClasses];
    }

    /**
     * Increments the count for a single (actual, predicted) pair.
     *
     * @param actual    actual class index
     * @param predicted predicted class index
     */
    public void accumulate(int actual, int predicted) {
        checkIndex(actual, "actual");
        checkIndex(predicted, "predicted");
        counts[actual][predicted]++;
    }

    /**
     * Returns the count for a given (actual, predicted) pair.
     */
    public long get(int actual, int predicted) {
        checkIndex(actual, "actual");
        checkIndex(predicted, "predicted");
        return counts[actual][predicted];
    }

    /**
     * Returns the number of classes.
     */
    public int numClasses() {
        return numClasses;
    }

    /**
     * Optional helper: total number of samples accumulated.
     */
    public long totalCount() {
        long total = 0L;
        for (int a = 0; a < numClasses; a++) {
            for (int p = 0; p < numClasses; p++) {
                total += counts[a][p];
            }
        }
        return total;
    }

    /**
     * Optional helper: diagonal (correct predictions) count.
     */
    public long correctCount() {
        long correct = 0L;
        for (int c = 0; c < numClasses; c++) {
            correct += counts[c][c];
        }
        return correct;
    }

    private void checkIndex(int index, String name) {
        if (index < 0 || index >= numClasses) {
            throw new IllegalArgumentException(
                    name + " index out of range: " + index + " (numClasses=" + numClasses + ")"
            );
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder("ConfusionMatrix(")
                .append(numClasses)
                .append(" classes)\n");

        for (int actual = 0; actual < numClasses; actual++) {
            builder.append("actual ")
                   .append(actual)
                   .append(": ");
            for (int predicted = 0; predicted < numClasses; predicted++) {
                builder.append(counts[actual][predicted]);
                if (predicted < numClasses - 1) {
                    builder.append(' ');
                }
            }
            builder.append('\n');
        }
        return builder.toString();
    }
}

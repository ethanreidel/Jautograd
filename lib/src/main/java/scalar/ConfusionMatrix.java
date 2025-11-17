package scalar;



public final class ConfusionMatrix {
    private final int numClasses;
    private final long[][] counts; //actual, predicted

    public ConfusionMatrix(int numClasses) {
        if (numClasses <= 1) {
            throw new IllegalArgumentException("num classes must be >= 2");
        }
        this.numClasses = numClasses;
        this.counts = new long[numClasses][numClasses];
    }

    private void checkRange(int idx, String name) {
        if (idx < 0 || idx >= numClasses) {
            throw new IllegalArgumentException(name + " out of range: " + idx);
        }
    }

    public void accumulate(int actual, int predicted) {
        checkRange(actual, "actual");
        checkRange(predicted, "predicted");
        counts[actual][predicted]++;
    }
    public long get(int actual, int predicted) {
        checkRange(actual, "actual");
        checkRange(predicted, "predicted");
        return counts[actual][predicted];
    }
    public int numClasses() {return numClasses;}
}

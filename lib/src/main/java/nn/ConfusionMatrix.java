package nn;

/**
 * Binary confusion matrix for labels 0 (negative) and 1 (positive).
 */
public class ConfusionMatrix {

    private int tp;
    private int fp;
    private int fn;
    private int tn;

    public ConfusionMatrix() {
        reset();
    }

    public void reset() {
        tp = 0;
        fp = 0;
        fn = 0;
        tn = 0;
    }

    /**
     * Update the confusion matrix with a single prediction.
     *
     * @param yTrue ground-truth label (0 or 1)
     * @param yPred predicted label (0 or 1)
     */
    public void update(int yTrue, int yPred) {
        validateBinaryLabel(yTrue, "yTrue");
        validateBinaryLabel(yPred, "yPred");

        if (yPred == 1 && yTrue == 1) {
            tp++;
        } else if (yPred == 1 && yTrue == 0) {
            fp++;
        } else if (yPred == 0 && yTrue == 1) {
            fn++;
        } else {
            tn++;
        }
    }

    /**
     * Update the confusion matrix with arrays of predictions.
     */
    public void update(int[] yTrue, int[] yPred) {
        if (yTrue == null || yPred == null) {
            throw new IllegalArgumentException("yTrue and yPred must not be null");
        }
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("yTrue and yPred length mismatch");
        }
        for (int i = 0; i < yTrue.length; i++) {
            update(yTrue[i], yPred[i]);
        }
    }

    public int getTP() {
        return tp;
    }

    public int getFP() {
        return fp;
    }

    public int getFN() {
        return fn;
    }

    public int getTN() {
        return tn;
    }

    public int total() {
        return tp + fp + fn + tn;
    }

    public double accuracy() {
        int tot = total();
        return tot == 0 ? 0.0 : (tp + tn) / (double) tot;
    }

    public double precision() {
        int denom = tp + fp;
        return denom == 0 ? 0.0 : tp / (double) denom;
    }

    public double recall() {
        int denom = tp + fn;
        return denom == 0 ? 0.0 : tp / (double) denom;
    }

    public double f1() {
        double p = precision();
        double r = recall();
        double sum = p + r;
        if (sum == 0.0) {
            return 0.0;
        }
        return 2.0 * p * r / sum;
    }

    public void merge(ConfusionMatrix other) {
        if (other == null) {
            return;
        }
        this.tp += other.tp;
        this.fp += other.fp;
        this.fn += other.fn;
        this.tn += other.tn;
    }

    public int[][] toArray() {
        return new int[][]{
                {tp, fp},
                {fn, tn}
        };
    }

    @Override
    public String toString() {
        return String.format(
                "ConfusionMatrix%n" +
                        "           Pred 1   Pred 0%n" +
                        "Actual 1:   %6d   %6d%n" +
                        "Actual 0:   %6d   %6d%n" +
                        "Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f",
                tp,
                fn,
                fp,
                tn,
                accuracy(),
                precision(),
                recall(),
                f1()
        );
    }

    private void validateBinaryLabel(int label, String name) {
        if (label != 0 && label != 1) {
            throw new IllegalArgumentException(name + " must be 0 or 1, but was: " + label);
        }
    }
}

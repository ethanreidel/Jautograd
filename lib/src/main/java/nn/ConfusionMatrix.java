package nn;

public class ConfusionMatrix {
    private int tp, fp, fn, tn;

    public void reset() { tp = fp = fn = tn = 0; }

    public void update(int yTrue, int yPred) {
        if ((yTrue & ~1) != 0 || (yPred & ~1) != 0) {
            throw new IllegalArgumentException("Labels must be 0 or 1");
        }
        if (yPred == 1 && yTrue == 1) tp++;
        else if (yPred == 1 && yTrue == 0) fp++;
        else if (yPred == 0 && yTrue == 1) fn++;
        else tn++;
    }

    public void update(int[] yTrue, int[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("yTrue and yPred length mismatch");
        }
        for (int i = 0; i < yTrue.length; i++) update(yTrue[i], yPred[i]);
    }

    public int getTP() { return tp; }
    public int getFP() { return fp; }
    public int getFN() { return fn; }
    public int getTN() { return tn; }

    public int total() { return tp + fp + fn + tn; }

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
        double p = precision(), r = recall();
        return (p + r == 0.0) ? 0.0 : 2.0 * p * r / (p + r);
    }

    public void merge(ConfusionMatrix other) {
        this.tp += other.tp;
        this.fp += other.fp;
        this.fn += other.fn;
        this.tn += other.tn;
    }

    public int[][] toArray() {
        return new int[][] {{tp, fp}, {fn, tn}};
    }

    @Override
    public String toString() {
        return String.format(
                "ConfusionMatrix\n" +
                        "           Pred 1   Pred 0\n" +
                        "Actual 1:   %6d   %6d\n" +
                        "Actual 0:   %6d   %6d\n" +
                        "Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f",
                tp, fn, fp, tn, accuracy(), precision(), recall(), f1()
        );
    }
}

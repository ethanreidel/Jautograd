package nn;

/**
 * Simple container for training metrics such as accuracy, epoch, and loss.
 */
public class Metrics {

    private double accuracy;
    private int epoch;
    private double loss;

    public Metrics() {
        this.accuracy = 0.0;
        this.epoch = 0;
        this.loss = 0.0;
    }

    public void updateAccuracy(double newAcc) {
        this.accuracy = newAcc;
    }

    public void updateEpoch() {
        this.epoch++;
    }

    public void updateLoss(double newLoss) {
        this.loss = newLoss;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public int getEpoch() {
        return epoch;
    }

    public double getLoss() {
        return loss;
    }

    @Override
    public String toString() {
        return "Metrics{"
                + "accuracy=" + accuracy
                + ", epoch=" + epoch
                + ", loss=" + loss
                + '}';
    }
}

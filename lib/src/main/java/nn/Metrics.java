package nn;

public class Metrics {
    private double accuracy;
    private int epoch;
    private double loss;

    public Metrics() {}

    public void updateAccuracy(double newAcc) {
        accuracy = newAcc;
    }
    public void updateEpoch() {
        epoch++;
    }
    public void updateLoss(double newLoss) {
        loss = newLoss;
    }

}

package nn;

public class ExponentialDecayScheduler implements LearningRateScheduler {
    private final double initialRate;
    private final double decayRate;

    public ExponentialDecayScheduler(double initialRate, double decayRate) {
        this.initialRate = initialRate;
        this.decayRate = decayRate;
    }

    @Override
    public double getLearningRate(int epoch) {
        return initialRate * Math.pow(decayRate, epoch);
    }
}
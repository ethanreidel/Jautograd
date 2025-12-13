package nn;

/**
 * Exponential decay learning rate scheduler.
 *
 * <p>learningRate(epoch) = initialRate * decayRate^epoch
 */
public class ExponentialDecayScheduler implements LearningRateScheduler {

    private final double initialRate;
    private final double decayRate;

    public ExponentialDecayScheduler(double initialRate, double decayRate) {
        this.initialRate = initialRate;
        this.decayRate = decayRate;
    }

    @Override
    public double getLearningRate(int epoch) {
        if (epoch < 0) {
            throw new IllegalArgumentException("epoch must be non-negative, but was: " + epoch);
        }
        return initialRate * Math.pow(decayRate, epoch);
    }

    public double getInitialRate() {
        return initialRate;
    }

    public double getDecayRate() {
        return decayRate;
    }
}

package nn;

/**
 * Step decay learning rate scheduler.
 *
 * <p>learningRate(epoch) = initialRate * decay^(epoch / stepSize)
 */
public class StepDecayScheduler implements LearningRateScheduler {

    private final double initialRate;
    private final int stepSize;
    private final double decay;

    public StepDecayScheduler(double initialRate, int stepSize, double decay) {
        if (initialRate <= 0.0) {
            throw new IllegalArgumentException("initialRate must be > 0");
        }
        if (stepSize <= 0) {
            throw new IllegalArgumentException("stepSize must be > 0");
        }
        if (decay <= 0.0) {
            throw new IllegalArgumentException("decay must be > 0");
        }

        this.initialRate = initialRate;
        this.stepSize = stepSize;
        this.decay = decay;
    }

    @Override
    public double getLearningRate(int epoch) {
        if (epoch < 0) {
            throw new IllegalArgumentException("epoch must be >= 0, but was: " + epoch);
        }
        int stepIndex = epoch / stepSize;
        return initialRate * Math.pow(decay, stepIndex);
    }

    public double getInitialRate() {
        return initialRate;
    }

    public int getStepSize() {
        return stepSize;
    }

    public double getDecay() {
        return decay;
    }
}

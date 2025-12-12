package nn;

public class StepDecayScheduler implements LearningRateScheduler {
    private final double initialRate;
    private final int stepSize;
    private final double decay;

    public StepDecayScheduler(double initialRate, int stepSize, double decay) {
        this.initialRate = initialRate;
        this.stepSize = stepSize;
        this.decay = decay;
    }

    @Override
    public double getLearningRate(int epoch) {
        return initialRate * Math.pow(decay, epoch / stepSize);
    }
}
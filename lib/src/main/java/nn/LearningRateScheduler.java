package nn;

/**
 * Strategy interface for learning rate schedules.
 */
public interface LearningRateScheduler {

    /**
     * Returns the learning rate for the given epoch.
     *
     * @param epoch non-negative training epoch index (0-based or 1-based by convention)
     * @return learning rate value for this epoch
     */
    double getLearningRate(int epoch);
}

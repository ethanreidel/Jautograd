package nn;

/**
 * Observer for training events.
 */
public interface TrainingObserver {

    /**
     * Called after each optimization step.
     */
    default void onStep(int globalStep, double loss, double accuracy) {
    }

    /**
     * Called after each epoch.
     */
    default void onEpoch(
            int epoch,
            ConfusionMatrix confusionMatrix,
            double avgLoss,
            double accuracy) {
    }
}

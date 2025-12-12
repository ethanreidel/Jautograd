package nn;

public interface TrainingObserver {
    default void onStep(int globalStep, double loss, double accuracy) {}
    default void onEpoch(int epoch, ConfusionMatrix confusionMatrix, double avgLoss, double accuracy) {}
}

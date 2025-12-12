package nn;

public interface LearningRateScheduler {
    double getLearningRate(int epoch);
}
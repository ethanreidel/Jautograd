package nn;

import java.util.List;
import scalar.Scalar;

/**
 * Base class for optimizers operating on a collection of parameters.
 */
public abstract class AbstractOptimizer implements Optimizer {

    private double learningRate;

    protected AbstractOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public abstract void step(List<Scalar> parameters);
}

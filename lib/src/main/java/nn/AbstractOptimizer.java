package nn;

import java.util.List;
import scalar.Scalar;

public abstract class AbstractOptimizer implements Optimizer{
    protected double learningRate;

    protected AbstractOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setLearningRate(double lr) {
        this.learningRate = lr;
    }

    @Override
    public abstract void step(List<Scalar> parameters);

}
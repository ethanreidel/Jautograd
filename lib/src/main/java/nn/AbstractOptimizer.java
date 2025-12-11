package nn;

import java.util.List;
import scalar.Scalar;

public abstract class AbstractOptimizer implements Optimizer{
    protected final double learningRate;

    protected AbstractOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public abstract void step(List<Scalar> parameters);
    
}
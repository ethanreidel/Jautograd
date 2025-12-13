package nn;

import java.util.List;
import scalar.Scalar;

/**
 * Vanilla stochastic gradient descent optimizer.
 */
public class SGDOptimizer extends AbstractOptimizer {

    public SGDOptimizer(double learningRate) {
        super(learningRate);
    }

    @Override
    public void step(List<Scalar> parameters) {
        if (parameters == null || parameters.isEmpty()) {
            return;
        }

        double learningRate = getLearningRate();
        for (Scalar parameter : parameters) {
            parameter.data -= learningRate * parameter.grad();
        }
    }
}

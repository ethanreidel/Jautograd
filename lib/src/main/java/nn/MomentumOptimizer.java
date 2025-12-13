package nn;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import scalar.Scalar;

/**
 * Momentum-based gradient descent optimizer.
 */
public class MomentumOptimizer extends AbstractOptimizer {

    private static final double DEFAULT_MOMENTUM = 0.9;

    private final double momentum;
    private final Map<Scalar, Double> velocities = new HashMap<>();

    public MomentumOptimizer(double learningRate) {
        this(learningRate, DEFAULT_MOMENTUM);
    }

    public MomentumOptimizer(double learningRate, double momentum) {
        super(learningRate);
        this.momentum = momentum;
    }

    @Override
    public void step(List<Scalar> parameters) {
        if (parameters == null || parameters.isEmpty()) {
            return;
        }

        for (Scalar parameter : parameters) {
            updateParameter(parameter);
        }
    }

    private void updateParameter(Scalar parameter) {
        double gradient = parameter.grad();

        double previousVelocity = velocities.getOrDefault(parameter, 0.0);
        double newVelocity = computeVelocity(previousVelocity, gradient);

        velocities.put(parameter, newVelocity);

        parameter.data += newVelocity;
    }

    private double computeVelocity(double previousVelocity, double gradient) {
        return momentum * previousVelocity - getLearningRate() * gradient;
    }

    public double getMomentum() {
        return momentum;
    }
}

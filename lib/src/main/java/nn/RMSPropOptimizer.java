package nn;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import scalar.Scalar;

/**
 * RMSProp optimizer.
 */
public class RMSPropOptimizer extends AbstractOptimizer {

    private static final double DEFAULT_DECAY_RATE = 0.99;
    private static final double DEFAULT_EPSILON = 1e-8;

    private final double decayRate;
    private final double epsilon;

    private final Map<Scalar, Double> squaredGradientAverages = new HashMap<>();

    public RMSPropOptimizer(double learningRate) {
        this(learningRate, DEFAULT_DECAY_RATE, DEFAULT_EPSILON);
    }

    public RMSPropOptimizer(double learningRate, double decayRate, double epsilon) {
        super(learningRate);
        this.decayRate = decayRate;
        this.epsilon = epsilon;
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
        double previousAverage = squaredGradientAverages.getOrDefault(parameter, 0.0);

        double newAverage = computeSquaredGradientAverage(previousAverage, gradient);
        squaredGradientAverages.put(parameter, newAverage);

        double update =
                getLearningRate() * gradient / (Math.sqrt(newAverage) + epsilon);

        parameter.data -= update;
    }

    private double computeSquaredGradientAverage(double previousAverage, double gradient) {
        return decayRate * previousAverage + (1.0 - decayRate) * gradient * gradient;
    }

    public double getDecayRate() {
        return decayRate;
    }

    public double getEpsilon() {
        return epsilon;
    }
}

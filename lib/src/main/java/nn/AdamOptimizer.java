package nn;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import scalar.Scalar;

/**
 * Adam optimizer implementation.
 *
 * <p>Reference: Kingma &amp; Ba, "Adam: A Method for Stochastic Optimization".
 */
public class AdamOptimizer extends AbstractOptimizer {

    private static final double DEFAULT_BETA1 = 0.9;
    private static final double DEFAULT_BETA2 = 0.999;
    private static final double DEFAULT_EPSILON = 1e-8;

    private final double beta1;
    private final double beta2;
    private final double epsilon;

    private final Map<Scalar, Double> firstMomentEstimates = new HashMap<>();
    private final Map<Scalar, Double> secondMomentEstimates = new HashMap<>();

    private int timeStep = 0;

    public AdamOptimizer(double learningRate) {
        this(learningRate, DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON);
    }

    public AdamOptimizer(double learningRate, double beta1, double beta2, double epsilon) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public void step(List<Scalar> parameters) {
        if (parameters == null || parameters.isEmpty()) {
            return;
        }

        timeStep++;

        for (Scalar parameter : parameters) {
            updateParameter(parameter);
        }
    }

    private void updateParameter(Scalar parameter) {
        double gradient = parameter.grad();

        double previousFirstMoment =
                firstMomentEstimates.getOrDefault(parameter, 0.0);
        double previousSecondMoment =
                secondMomentEstimates.getOrDefault(parameter, 0.0);

        double firstMoment = computeFirstMoment(previousFirstMoment, gradient);
        double secondMoment = computeSecondMoment(previousSecondMoment, gradient);

        firstMomentEstimates.put(parameter, firstMoment);
        secondMomentEstimates.put(parameter, secondMoment);

        double biasCorrectedFirstMoment =
                computeBiasCorrectedMoment(firstMoment, beta1, timeStep);
        double biasCorrectedSecondMoment =
                computeBiasCorrectedMoment(secondMoment, beta2, timeStep);

        double stepSize =
                getLearningRate()
                        * biasCorrectedFirstMoment
                        / (Math.sqrt(biasCorrectedSecondMoment) + epsilon);

        parameter.data -= stepSize;
    }

    private double computeFirstMoment(double previousFirstMoment, double gradient) {
        return beta1 * previousFirstMoment + (1.0 - beta1) * gradient;
    }

    private double computeSecondMoment(double previousSecondMoment, double gradient) {
        return beta2 * previousSecondMoment + (1.0 - beta2) * gradient * gradient;
    }

    private double computeBiasCorrectedMoment(double moment, double beta, int timeStep) {
        double biasCorrection = 1.0 - Math.pow(beta, timeStep);
        return moment / biasCorrection;
    }
}

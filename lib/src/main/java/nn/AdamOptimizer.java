package nn;

import scalar.Scalar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AdamOptimizer extends AbstractOptimizer {

    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;

    private final Map<Scalar, Double> m = new HashMap<>();
    private final Map<Scalar, Double> v = new HashMap<>();

    private int timeStep = 0;

    public AdamOptimizer(double learningRate) {
        super(learningRate);
    }

    @Override
    public void step(List<Scalar> parameters) {
        timeStep++;

        for (Scalar p : parameters) {
            double g = p.grad();

            double prevM = m.getOrDefault(p, 0.0);
            double prevV = v.getOrDefault(p, 0.0);

            double mt = beta1 * prevM + (1 - beta1) * g;
            double vt = beta2 * prevV + (1 - beta2) * (g * g);

            m.put(p, mt);
            v.put(p, vt);

            double mtHat = mt / (1 - Math.pow(beta1, timeStep));
            double vtHat = vt / (1 - Math.pow(beta2, timeStep));

            p.data -= learningRate * mtHat / (Math.sqrt(vtHat) + epsilon);
        }
    }
}
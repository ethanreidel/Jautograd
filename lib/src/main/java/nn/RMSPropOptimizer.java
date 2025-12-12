package nn;

import scalar.Scalar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RMSPropOptimizer extends AbstractOptimizer {

    private final double decayRate = 0.99;
    private final double epsilon = 1e-8;

    private final Map<Scalar, Double> cache = new HashMap<>();

    public RMSPropOptimizer(double learningRate) {
        super(learningRate);
    }

    @Override
    public void step(List<Scalar> parameters) {
        for (Scalar p : parameters) {
            double g = p.grad();

            double prev = cache.getOrDefault(p, 0.0);
            double newCache = decayRate * prev + (1 - decayRate) * (g * g);
            cache.put(p, newCache);

            double update = learningRate * g / (Math.sqrt(newCache) + epsilon);

            p.data -= update;
        }
    }
}

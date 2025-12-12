package nn;

import scalar.Scalar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MomentumOptimizer extends AbstractOptimizer {

    private final double momentum = 0.9;
    private final Map<Scalar, Double> velocity = new HashMap<>();

    public MomentumOptimizer(double learningRate) {
        super(learningRate);
    }

    @Override
    public void step(List<Scalar> parameters) {
        for (Scalar p : parameters) {
            double g = p.grad();

            double vPrev = velocity.getOrDefault(p, 0.0);
            double vNew = momentum * vPrev - learningRate * g;
            velocity.put(p, vNew);

            p.data += vNew;
        }
    }
}

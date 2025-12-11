package nn;

import java.util.List;
import scalar.Scalar;

public class SGDOptimizer extends AbstractOptimizer{
    public SGDOptimizer(double learningRate) {
        super(learningRate);
    }
    @Override
    public void step(List<Scalar> parameters) {
        for (Scalar p : parameters) {
            p.data -= learningRate * p.grad();
        }
    }

}
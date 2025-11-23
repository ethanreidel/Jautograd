package nn;

import scalar.Scalar;
import java.util.List;


public class Optimizer {
    private final double learningRate;

    //start with SGD

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public void step(List<Scalar> parameters) {
        for (Scalar p : parameters) {
            p.data -= learningRate * p.grad();
        }
    }

}

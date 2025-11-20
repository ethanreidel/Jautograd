package nn;

import scalar.Scalar;
import java.util.List;


public class Optimizer {
    private final List<Scalar> parameters;
    private final double learningRate;

    //start with SGD

    public Optimizer(List<Scalar> parameters, double learningRate) {
        this.parameters = parameters;
        this.learningRate = learningRate;
    }

    public void SGD() {
        for (Scalar p : parameters) {
            p.data -= learningRate * p.grad();
        }
    }

}

package nn;

import java.util.List;
import scalar.Scalar;

/**
 * Optimization algorithm that updates parameters based on their gradients.
 */
public interface Optimizer {

    /**
     * Perform one optimization step on the given parameters.
     *
     * @param parameters list of parameters to update
     */
    void step(List<Scalar> parameters);
}

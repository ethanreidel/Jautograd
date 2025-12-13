package nn;

import java.util.List;
import scalar.Scalar;

/**
 * Base class for neural network modules.
 *
 * <p>Each module must implement a forward pass and expose its parameters.
 */
public abstract class Module {

    /**
     * Forward pass of the module.
     *
     * @param inputs list of input scalars
     * @return list of output scalars
     */
    public abstract List<Scalar> forward(List<Scalar> inputs);

    /**
     * All parameters that should receive gradients and be updated by optimizers.
     *
     * @return list of parameters
     */
    public abstract List<Scalar> parameters();

    /**
     * Resets gradients of all parameters to zero.
     */
    public void zeroGrad() {
        for (Scalar parameter : parameters()) {
            parameter.zeroGrad();
        }
    }
}

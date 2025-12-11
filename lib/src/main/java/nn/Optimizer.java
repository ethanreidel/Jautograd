package nn;

import scalar.Scalar;
import java.util.List;

public interface Optimizer {
    void step(List<Scalar> parameters);
}
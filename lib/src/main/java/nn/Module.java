package nn;

import java.util.List;
import scalar.Scalar;

public abstract class Module {
    public abstract List<Scalar> parameters();
    public void zeroGrad() {
        for (Scalar s : parameters()) {
            s.zeroGrad();
        }
    }
}

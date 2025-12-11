package nn;

import java.util.ArrayList;
import java.util.List;
import scalar.Scalar;

public class Layer extends Module {
    private final List<Neuron> neurons;

    public Layer(int nin, int nout, boolean nonlin) {
        neurons = new ArrayList<>();
        for (int i = 0; i < nout; i++) {
            neurons.add(new Neuron(nin, nonlin));
        }
    }

    public List<Scalar> forward(List<Scalar> x) {
        List<Scalar> out = new ArrayList<>();
        for (Neuron n : neurons) {
            out.add(n.forwardScalar(x));
        }
        return out.size() == 1 ? List.of(out.get(0)) : out;
    }

    @Override
    public List<Scalar> parameters() {
        List<Scalar> params = new ArrayList<>();
        for (Neuron n : neurons) {
            params.addAll(n.parameters());
        }
        return params;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Layer of [");
        for (int i = 0; i < neurons.size(); i++) {
            sb.append(neurons.get(i));
            if (i < neurons.size() - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}

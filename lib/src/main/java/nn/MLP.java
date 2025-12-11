package nn;

import java.util.ArrayList;
import java.util.List;
import scalar.Scalar;

public class MLP extends Module {
    private final List<Layer> layers;

    public MLP(int nin, List<Integer> nouts) {
        this.layers = new ArrayList<>();
        List<Integer> sz = new ArrayList<>();
        sz.add(nin);
        sz.addAll(nouts);
        for (int i = 0; i < nouts.size(); i++) {
            boolean nonlin = i != nouts.size() - 1;
            layers.add(new Layer(sz.get(i), sz.get(i + 1), nonlin));
        }
    }

    @Override
    public List<Scalar> forward(List<Scalar> x) {
        List<Scalar> out = x;
        for (Layer layer : layers) {
            out = layer.forward(out);
        }
        return out;
    }

    @Override
    public List<Scalar> parameters() {
        List<Scalar> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        return params;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("MLP of [");
        for (int i = 0; i < layers.size(); i++) {
            sb.append(layers.get(i));
            if (i < layers.size() - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}

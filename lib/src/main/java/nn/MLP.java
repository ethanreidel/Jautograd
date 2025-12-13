package nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import scalar.Scalar;

/**
 * Multi-layer perceptron composed of stacked {@link Layer} objects.
 */
public class MLP extends Module {

    private final List<Layer> layers;

    /**
     * Constructs an MLP.
     *
     * @param nin   number of input features
     * @param nouts list of output sizes for each layer
     */
    public MLP(int nin, List<Integer> nouts) {
        if (nouts == null || nouts.isEmpty()) {
            throw new IllegalArgumentException("nouts must not be null or empty");
        }

        this.layers = new ArrayList<>();
        List<Integer> sizes = new ArrayList<>(nouts.size() + 1);
        sizes.add(nin);
        sizes.addAll(nouts);

        for (int i = 0; i < nouts.size(); i++) {
            boolean nonlin = (i != nouts.size() - 1);
            layers.add(new Layer(sizes.get(i), sizes.get(i + 1), nonlin));
        }
    }

    @Override
    public List<Scalar> forward(List<Scalar> x) {
        if (x == null) {
            throw new IllegalArgumentException("input must not be null");
        }

        List<Scalar> output = x;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    @Override
    public List<Scalar> parameters() {
        List<Scalar> parameters = new ArrayList<>();
        for (Layer layer : layers) {
            parameters.addAll(layer.parameters());
        }
        return parameters;
    }

    public List<Layer> getLayers() {
        return Collections.unmodifiableList(layers);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder("MLP of [");
        for (int i = 0; i < layers.size(); i++) {
            builder.append(layers.get(i));
            if (i < layers.size() - 1) {
                builder.append(", ");
            }
        }
        builder.append(']');
        return builder.toString();
    }
}

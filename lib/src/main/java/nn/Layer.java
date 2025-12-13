package nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import scalar.Scalar;

/**
 * Fully-connected neural network layer (collection of neurons).
 */
public class Layer extends Module {

    private final List<Neuron> neurons;

    /**
     * Constructs a layer.
     *
     * @param nin    number of inputs
     * @param nout   number of outputs (neurons)
     * @param nonlin whether to apply non-linearity in neurons
     */
    public Layer(int nin, int nout, boolean nonlin) {
        this.neurons = new ArrayList<>();
        initializeNeurons(nin, nout, nonlin);
    }

    private void initializeNeurons(int nin, int nout, boolean nonlin) {
        for (int i = 0; i < nout; i++) {
            neurons.add(new Neuron(nin, nonlin));
        }
    }

    /**
     * Forward pass through the layer.
     *
     * @param inputs list of input scalars
     * @return list of output scalars
     */
    public List<Scalar> forward(List<Scalar> inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("inputs must not be null");
        }

        List<Scalar> outputs = new ArrayList<>(neurons.size());
        for (Neuron neuron : neurons) {
            outputs.add(neuron.forwardScalar(inputs));
        }

        // Preserve original behavior: if single output, return a singleton list.
        if (outputs.size() == 1) {
            return Collections.singletonList(outputs.get(0));
        }

        return outputs;
    }

    @Override
    public List<Scalar> parameters() {
        List<Scalar> parameters = new ArrayList<>();
        for (Neuron neuron : neurons) {
            parameters.addAll(neuron.parameters());
        }
        return parameters;
    }

    public List<Neuron> getNeurons() {
        return Collections.unmodifiableList(neurons);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder("Layer of [");
        for (int i = 0; i < neurons.size(); i++) {
            builder.append(neurons.get(i));
            if (i < neurons.size() - 1) {
                builder.append(", ");
            }
        }
        builder.append(']');
        return builder.toString();
    }
}

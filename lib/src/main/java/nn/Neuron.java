package nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import scalar.Scalar;

/**
 * Single fully-connected neuron with optional ReLU non-linearity.
 */
public class Neuron extends Module {

    private static final Random RANDOM = new Random();

    private final List<Scalar> weights;
    private final Scalar bias;
    private final boolean useNonLinearity;

    /**
     * Constructs a neuron with random weights in [-1, 1] and zero bias.
     *
     * @param inputSize     number of inputs
     * @param useNonLinearity whether to apply ReLU non-linearity
     */
    public Neuron(int inputSize, boolean useNonLinearity) {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be > 0, but was: " + inputSize);
        }

        this.weights = new ArrayList<>(inputSize);
        for (int i = 0; i < inputSize; i++) {
            double value = RANDOM.nextDouble() * 2.0 - 1.0;
            this.weights.add(new Scalar(value));
        }
        this.bias = new Scalar(0.0);
        this.useNonLinearity = useNonLinearity;
    }

    /**
     * Forward pass returning a single scalar.
     *
     * @param inputs list of input scalars
     * @return neuron output as scalar
     */
    public Scalar forwardScalar(List<Scalar> inputs) {
        validateInputs(inputs);

        Scalar activation = bias;
        for (int i = 0; i < weights.size(); i++) {
            activation = activation.add(weights.get(i).mul(inputs.get(i)));
        }

        if (useNonLinearity) {
            return activation.relu();
        }
        return activation;
    }

    @Override
    public List<Scalar> forward(List<Scalar> inputs) {
        List<Scalar> outputs = new ArrayList<>(1);
        outputs.add(forwardScalar(inputs));
        return outputs;
    }

    @Override
    public List<Scalar> parameters() {
        List<Scalar> parameters = new ArrayList<>(weights);
        parameters.add(bias);
        return parameters;
    }

    public List<Scalar> getWeights() {
        return Collections.unmodifiableList(weights);
    }

    public Scalar getBias() {
        return bias;
    }

    public boolean isUsingNonLinearity() {
        return useNonLinearity;
    }

    @Override
    public String toString() {
        String type = useNonLinearity ? "ReLU" : "Linear";
        return type + "Neuron(" + weights.size() + ")";
    }

    private void validateInputs(List<Scalar> inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("inputs must not be null");
        }
        if (inputs.size() != weights.size()) {
            throw new IllegalArgumentException(
                    "Input size (" + inputs.size() + ") must match weight size (" + weights.size() + ")");
        }
    }
}

package data;

/**
 * An immutable example: a feature vector and its integer label.
 */
public record Example(double[] features, int label) {

    /**
     * Canonical constructor with defensive copy and basic validation.
     */
    public Example {
        if (features == null) {
            throw new IllegalArgumentException("features must not be null");
        }
        // Defensive copy into the record's field
        features = features.clone();
    }

    /**
     * Returns a copy of the feature array to preserve immutability.
     */
    @Override
    public double[] features() {
        return features.clone();
    }

    /**
     * Label accessor (kept explicitly for clarity; identical to the generated one).
     */
    @Override
    public int label() {
        return label;
    }
}

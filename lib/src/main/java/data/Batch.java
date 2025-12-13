package data;

import java.util.List;
import java.util.Objects;
import scalar.Scalar;

/**
 * A mini-batch of data.
 *
 * <p>features: shape [numFeatures][batchSize], i.e. features.get(f).get(i)
 * is the scalar value of feature {@code f} for sample {@code i}.
 *
 * <p>label: labels for each sample in the batch.
 */
public record Batch(List<List<Scalar>> features, List<Integer> label) {

    /**
     * Canonical constructor with basic validation.
     */
    public Batch {
        Objects.requireNonNull(features, "features must not be null");
        Objects.requireNonNull(label, "label must not be null");
        if (features.isEmpty()) {
            throw new IllegalArgumentException("features must not be empty");
        }
        int batchSize = label.size();
        for (int i = 0; i < features.size(); i++) {
            List<Scalar> column = features.get(i);
            if (column == null) {
                throw new IllegalArgumentException("features[" + i + "] must not be null");
            }
            if (column.size() != batchSize) {
                throw new IllegalArgumentException(
                        "All feature columns must have same size as label list. "
                                + "Expected " + batchSize + " but got " + column.size()
                                + " at feature index " + i
                );
            }
        }
    }

    /**
     * Returns the feature column at index {@code i}.
     *
     * <p>Note: despite its name, this returns a feature slice, not a whole sample.
     * It preserves the original API to avoid breaking callers.
     */
    public List<Scalar> getBatch(int i) {
        return features.get(i);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder("Batch{features=[");
        for (int rowIndex = 0; rowIndex < features.size(); rowIndex++) {
            List<Scalar> row = features.get(rowIndex);
            builder.append('[');
            for (int i = 0; i < row.size(); i++) {
                builder.append(String.format("%.3f", row.get(i).data()));
                if (i < row.size() - 1) {
                    builder.append(", ");
                }
            }
            builder.append(']');
            if (rowIndex < features.size() - 1) {
                builder.append(", ");
            }
        }
        builder.append("], label=").append(label).append('}');
        return builder.toString();
    }
}

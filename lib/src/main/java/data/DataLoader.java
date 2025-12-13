package data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import scalar.Scalar;

/**
 * Creates mini-batches ({@link Batch}) from a {@link CsvDataset}.
 *
 * <p>Each batch has:
 * <ul>
 *   <li>features: [numFeatures][batchSize] of {@link Scalar}</li>
 *   <li>label: list of integer labels, one per sample</li>
 * </ul>
 */
public final class DataLoader {

    private final int batchSize;
    private final boolean shuffle;
    private final Random rng;

    public DataLoader(int batchSize, boolean shuffle) {
        this(batchSize, shuffle, null);
    }

    public DataLoader(int batchSize, boolean shuffle, Long seed) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be > 0");
        }
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.rng = (seed == null) ? null : new Random(seed);
    }

    public int batchSize() {
        return batchSize;
    }

    /**
     * Loads the entire dataset into memory and returns a list of batches.
     *
     * @throws IllegalArgumentException if dataset is null or inconsistent.
     */
    public List<Batch> loadData(CsvDataset dataset) {
        if (dataset == null) {
            throw new IllegalArgumentException("dataset must not be null");
        }

        List<Example> examples = new ArrayList<>();
        for (Example example : dataset) {
            examples.add(example);
        }

        if (examples.isEmpty()) {
            return Collections.emptyList();
        }

        if (shuffle) {
            if (rng != null) {
                Collections.shuffle(examples, rng);
            } else {
                Collections.shuffle(examples);
            }
        }

        int numFeatures = examples.get(0).features().length;
        List<Batch> batches = new ArrayList<>();

        for (int start = 0; start < examples.size(); start += batchSize) {
            int end = Math.min(start + batchSize, examples.size());
            int currentBatchSize = end - start;

            List<List<Scalar>> featureColumns = new ArrayList<>(numFeatures);
            for (int feat = 0; feat < numFeatures; feat++) {
                featureColumns.add(new ArrayList<>(currentBatchSize));
            }

            List<Integer> labels = new ArrayList<>(currentBatchSize);

            for (int index = start; index < end; index++) {
                Example example = examples.get(index);
                double[] feats = example.features();

                if (feats.length != numFeatures) {
                    throw new IllegalArgumentException(
                            "Inconsistent feature length at index " + index
                                    + ": expected " + numFeatures
                                    + " but got " + feats.length
                    );
                }

                for (int feat = 0; feat < numFeatures; feat++) {
                    featureColumns.get(feat).add(new Scalar(feats[feat]));
                }
                labels.add(example.label());
            }

            batches.add(new Batch(featureColumns, labels));
        }

        return batches;
    }
}

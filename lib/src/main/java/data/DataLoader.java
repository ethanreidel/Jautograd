package data;

import java.security.InvalidParameterException;
import java.util.*;
import scalar.Scalar;

public final class DataLoader {
    private final int batchSize;
    private final boolean shuffle;
    private final Random rng; // optional deterministic shuffle

    public DataLoader(int batchSize, boolean shuffle) {
        this(batchSize, shuffle, null);
    }

    public DataLoader(int batchSize, boolean shuffle, Long seed) {
        if (batchSize <= 0) throw new InvalidParameterException("batchSize must be > 0");
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.rng = (seed == null) ? null : new Random(seed);
    }

    public int batchSize() { return batchSize; }

    public List<Batch> loadData(CsvDataset ds) {
        if (ds == null) throw new InvalidParameterException("Dataset is not ready");

        List<Example> all = new ArrayList<>();
        for (Example e : ds) all.add(e);

        if (all.isEmpty()) return Collections.emptyList();

        if (shuffle) {
            if (rng != null) Collections.shuffle(all, rng);
            else Collections.shuffle(all);
        }

        int numFeatures = all.get(0).features().length;
        List<Batch> output = new ArrayList<>();

        for (int i = 0; i < all.size(); i += batchSize) {
            int j = Math.min(i + batchSize, all.size());
            int thisBatch = j - i;

            List<List<Scalar>> f = new ArrayList<>(numFeatures);
            for (int feat = 0; feat < numFeatures; feat++) {
                f.add(new ArrayList<>(thisBatch));
            }
            List<Integer> y = new ArrayList<>(thisBatch);

            for (int k = i; k < j; k++) {
                Example ex = all.get(k);
                double[] feats = ex.features();
                if (feats.length != numFeatures) {
                    throw new InvalidParameterException("Inconsistent feature length at index " + k);
                }
                for (int feat = 0; feat < numFeatures; feat++) {
                    f.get(feat).add(new Scalar(feats[feat]));
                }
                y.add(ex.label());
            }

            output.add(new Batch(f, y));
        }

        return output;
    }
}

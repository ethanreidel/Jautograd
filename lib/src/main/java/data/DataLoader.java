package data;

import java.security.InvalidParameterException;
import java.util.*;

public final class DataLoader {
    private final int batchSize;
    private final boolean shuffle;

    public DataLoader(int batchSize, boolean shuffle) {
        if (batchSize <= 0) throw new InvalidParameterException("batchSize must be > 0");
        this.batchSize = batchSize;
        this.shuffle = shuffle;
    }

    public List<Batch> loadData(CsvDataset ds) {
        if (ds == null) {
            throw new InvalidParameterException("Dataset is not ready");
        }
        List<Example> all = new ArrayList<>();
        for (Example e : ds) {
            all.add(e);
        }
        List<Batch> output = new ArrayList<>();
        for (int i = 0; i < all.size(); i+= batchSize) {
            int j = Math.min(i + batchSize, all.size());
            var f = new ArrayList<double[]>(j - i);
            var y = new ArrayList<Integer>(j - i);
            for (int k = i; k < j; k++) {
                f.add(all.get(k).features());
                y.add(all.get(k).label());
            }
            output.add(new Batch(f, y));
            }
        return output;
    }
}

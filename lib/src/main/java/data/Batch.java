package data;

import java.util.*;

public record Batch(ArrayList<double[]> features, ArrayList<Integer> label) {
    public Batch {}
 
    @Override
    public String toString() {
        return "features: " + Arrays.deepToString(features.toArray(new double[features.size()][])) + ". label: " + Arrays.toString(label.stream().mapToInt(Integer::intValue).toArray());
    }
}

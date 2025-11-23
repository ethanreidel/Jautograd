package data;

import java.util.*;
import scalar.Scalar;

public record Batch(List<List<Scalar>> features, List<Integer> label) {
    public Batch {}

    public List<Scalar> getBatch(int i) {
        return features.get(i);
    }
 
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("features: [");
        for (List<Scalar> row : features) {
            sb.append("[");
            for (int i = 0; i < row.size(); i++) {
                sb.append(String.format("%.3f", row.get(i).data()));
                if (i < row.size() - 1) sb.append(", ");
            }
            sb.append("]");
        }
        sb.append("], label: ").append(label);
        return sb.toString();
    }
}

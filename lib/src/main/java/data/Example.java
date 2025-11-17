package data;

public record Example(double[] features, int label) {
    public Example {
        features = features.clone();
    }
    public double[] features() {
        return features;
    }
    public int label() {
        return label;
    }
}

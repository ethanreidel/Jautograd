package nn;

import java.util.List;
import java.io.*;

import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.Constructor;

public class ModelConfig {
    private int numFeatures;
    private List<Integer> layerSizes;
    private double learningRate;
    private int epochs;
    private int batchSize;
    private String datasetPath;

    private transient Optimizer optimizer;
    private transient MLP model;

    public ModelConfig() {}

    private ModelConfig(Builder builder) {
        this.numFeatures = builder.numFeatures;
        this.layerSizes = builder.layerSizes;
        this.learningRate = builder.learningRate;
        this.epochs = builder.epochs;
        this.batchSize = builder.batchSize;
        this.datasetPath = builder.datasetPath;
        this.optimizer = builder.optimizer;
        this.model = builder.model;
    }

    public int getNumFeatures() { return numFeatures; }
    public List<Integer> getLayerSizes() { return layerSizes; }
    public double getLearningRate() { return learningRate; }
    public int getEpochs() { return epochs; }
    public int getBatchSize() { return batchSize; }
    public String getDatasetPath() { return datasetPath; }
    public Optimizer getOptimizer() { return optimizer; }
    public MLP getModel() { return model; }

    public void setNumFeatures(int numFeatures) { this.numFeatures = numFeatures; }
    public void setLayerSizes(List<Integer> layerSizes) { this.layerSizes = layerSizes; }
    public void setLearningRate(double learningRate) { this.learningRate = learningRate; }
    public void setEpochs(int epochs) { this.epochs = epochs; }
    public void setBatchSize(int batchSize) { this.batchSize = batchSize; }
    public void setDatasetPath(String datasetPath) { this.datasetPath = datasetPath; }

    public static class Builder {
        private int numFeatures;
        private List<Integer> layerSizes;
        private double learningRate;
        private int epochs;
        private int batchSize;
        private String datasetPath;
        private Optimizer optimizer;
        private MLP model;

        public Builder numFeatures(int numFeatures) { this.numFeatures = numFeatures; return this; }
        public Builder layerSizes(List<Integer> layerSizes) { this.layerSizes = layerSizes; return this; }
        public Builder learningRate(double learningRate) { this.learningRate = learningRate; return this; }
        public Builder epochs(int epochs) { this.epochs = epochs; return this; }
        public Builder batchSize(int batchSize) { this.batchSize = batchSize; return this; }
        public Builder datasetPath(String datasetPath) { this.datasetPath = datasetPath; return this; }
        public Builder optimizer(Optimizer optimizer) { this.optimizer = optimizer; return this; }
        public Builder model(MLP model) { this.model = model; return this; }
        public ModelConfig build() { return new ModelConfig(this); }
    }

    public void saveConfigYaml(String filePath) {
        Yaml yaml = new Yaml();
        try (FileWriter writer = new FileWriter(filePath)) {
            yaml.dump(this, writer);
        } catch (IOException e) {
            throw new RuntimeException("Failed to save ModelConfig YAML: " + e.getMessage(), e);
        }
    }

    public static ModelConfig readConfigYaml(String filePath) {
        Yaml yaml = new Yaml(new Constructor(ModelConfig.class));
        try (FileReader reader = new FileReader(filePath)) {
            ModelConfig config = yaml.load(reader);
            config.optimizer = new Optimizer(config.learningRate);
            config.model = new MLP(config.numFeatures, config.layerSizes);
            return config;
        } catch (IOException e) {
            throw new RuntimeException("Failed to read ModelConfig YAML: " + e.getMessage(), e);
        }
    }
}
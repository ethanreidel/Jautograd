package nn;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.Constructor;

/**
 * Configuration for model, optimizer, and training setup.
 *
 * <p>Serializable to/from YAML via SnakeYAML.
 */
public class ModelConfig {

    private int numFeatures;
    private List<Integer> layerSizes;
    private double learningRate;
    private int epochs;
    private int batchSize;
    private String datasetPath;
    private LearningRateScheduler learningRateScheduler;

    private OptimizerType optimizerType = OptimizerType.SGD;

    // Non-serialized runtime fields.
    private transient Optimizer optimizer;
    private transient MLP model;

    // Required by SnakeYAML.
    public ModelConfig() {
    }

    private ModelConfig(Builder builder) {
        this.numFeatures = builder.numFeatures;
        this.layerSizes = builder.layerSizes;
        this.learningRate = builder.learningRate;
        this.epochs = builder.epochs;
        this.batchSize = builder.batchSize;
        this.datasetPath = builder.datasetPath;
        this.optimizerType = builder.optimizerType;
        this.learningRateScheduler = builder.learningRateScheduler;
        this.optimizer = builder.optimizer;
        this.model = builder.model;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    public List<Integer> getLayerSizes() {
        return layerSizes;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public int getEpochs() {
        return epochs;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public String getDatasetPath() {
        return datasetPath;
    }

    public OptimizerType getOptimizerType() {
        return optimizerType;
    }

    public LearningRateScheduler getLearningRateScheduler() {
        return learningRateScheduler;
    }

    public MLP getModel() {
        return model;
    }

    public void setOptimizerType(OptimizerType optimizerType) {
        this.optimizerType = optimizerType;
    }

    public void setNumFeatures(int numFeatures) {
        this.numFeatures = numFeatures;
    }

    public void setLayerSizes(List<Integer> layerSizes) {
        this.layerSizes = layerSizes;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void setDatasetPath(String datasetPath) {
        this.datasetPath = datasetPath;
    }

    public void setLearningRateScheduler(LearningRateScheduler learningRateScheduler) {
        this.learningRateScheduler = learningRateScheduler;
    }

    /**
     * Builder for {@link ModelConfig}.
     */
    public static class Builder {

        private int numFeatures;
        private List<Integer> layerSizes;
        private double learningRate;
        private int epochs;
        private int batchSize;
        private String datasetPath;
        private OptimizerType optimizerType = OptimizerType.SGD;
        private LearningRateScheduler learningRateScheduler;
        private Optimizer optimizer;
        private MLP model;

        public Builder numFeatures(int numFeatures) {
            this.numFeatures = numFeatures;
            return this;
        }

        public Builder layerSizes(List<Integer> layerSizes) {
            this.layerSizes = layerSizes;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder datasetPath(String datasetPath) {
            this.datasetPath = datasetPath;
            return this;
        }

        public Builder model(MLP model) {
            this.model = model;
            return this;
        }

        public Builder optimizerType(OptimizerType optimizerType) {
            this.optimizerType = optimizerType;
            return this;
        }

        public Builder optimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public Builder learningRateScheduler(LearningRateScheduler scheduler) {
            this.learningRateScheduler = scheduler;
            return this;
        }

        public ModelConfig build() {
            validate();
            return new ModelConfig(this);
        }

        private void validate() {
            if (numFeatures <= 0) {
                throw new IllegalArgumentException("numFeatures must be > 0");
            }
            if (layerSizes == null || layerSizes.isEmpty()) {
                throw new IllegalArgumentException("layerSizes must not be empty");
            }
            if (learningRate <= 0.0) {
                throw new IllegalArgumentException("learningRate must be > 0");
            }
            if (epochs < 0) {
                throw new IllegalArgumentException("epochs must be >= 0");
            }
        }
    }

    /**
     * Creates a new optimizer instance based on the configured optimizer type and learning rate.
     *
     * @return newly created optimizer
     */
    public Optimizer createOptimizer() {
        if (optimizer != null) {
            return optimizer;
        }
        if (optimizerType == null) {
            optimizerType = OptimizerType.SGD;
        }

        return switch (optimizerType) {
            case ADAM -> new AdamOptimizer(learningRate);
            case RMSPROP -> new RMSPropOptimizer(learningRate);
            case MOMENTUM -> new MomentumOptimizer(learningRate);
            case SGD -> new SGDOptimizer(learningRate);
        };
    }

    /**
     * Returns the existing optimizer if present, otherwise creates and caches it.
     */
    public Optimizer getOrCreateOptimizer() {
        if (optimizer == null) {
            optimizer = createOptimizer();
        }
        return optimizer;
    }

    /**
     * Returns the existing model if present, otherwise creates and caches it.
     */
    public MLP getOrCreateModel() {
        if (model == null) {
            model = new MLP(numFeatures, layerSizes);
        }
        return model;
    }

    /**
     * Serializes this configuration to YAML.
     *
     * @param filePath path to the YAML file
     */
    public void saveConfigYaml(String filePath) {
        Yaml yaml = new Yaml();
        try (FileWriter writer = new FileWriter(filePath)) {
            yaml.dump(this, writer);
        } catch (IOException e) {
            throw new RuntimeException("Failed to save ModelConfig YAML: " + e.getMessage(), e);
        }
    }

    /**
     * Reads configuration from YAML and reconstructs the model and optimizer.
     *
     * @param filePath path to the YAML file
     * @return restored {@link ModelConfig}
     */
    public static ModelConfig readConfigYaml(String filePath) {
        Yaml yaml = new Yaml(new Constructor(ModelConfig.class));
        try (FileReader reader = new FileReader(filePath)) {
            ModelConfig config = yaml.load(reader);
            if (config == null) {
                throw new RuntimeException("YAML did not contain a ModelConfig.");
            }
            config.model = new MLP(config.numFeatures, config.layerSizes);
            config.optimizer = config.createOptimizer();
            return config;
        } catch (IOException e) {
            throw new RuntimeException("Failed to read ModelConfig YAML: " + e.getMessage(), e);
        }
    }

    @Override
    public String toString() {
        return "ModelConfig{"
                + "numFeatures=" + numFeatures
                + ", layerSizes=" + layerSizes
                + ", learningRate=" + learningRate
                + ", epochs=" + epochs
                + ", batchSize=" + batchSize
                + ", datasetPath='" + datasetPath + '\''
                + ", optimizerType=" + optimizerType
                + ", learningRateScheduler=" + learningRateScheduler
                + '}';
    }
}

package integration_tests;

import nn.ModelConfig;
import nn.MLP;
import nn.OptimizerType;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class DefineConfigIntegrationIT {

    @Test
    @DisplayName("Define ModelConfig, save to YAML, and load it back")
    void defineSaveLoad() throws Exception {
        int numFeatures = 784;
        List<Integer> layerSizes = List.of(128, 10);
        double learningRate = 1e-3;
        int epochs = 3;
        int batchSize = 64;
        String datasetPath = "mnist-train.csv";

        ModelConfig original = new ModelConfig.Builder()
                .numFeatures(numFeatures)
                .layerSizes(layerSizes)
                .learningRate(learningRate)
                .epochs(epochs)
                .batchSize(batchSize)
                .datasetPath(datasetPath)
                .optimizerType(OptimizerType.SGD)
                .model(new MLP(numFeatures, layerSizes))
                .build();

        Path tmp = Files.createTempFile("model_config_define_save_load", ".yaml");
        original.saveConfigYaml(tmp.toString());

        ModelConfig loaded = ModelConfig.readConfigYaml(tmp.toString());

        assertEquals(original.getNumFeatures(), loaded.getNumFeatures());
        assertEquals(original.getLayerSizes(), loaded.getLayerSizes());
        assertEquals(original.getLearningRate(), loaded.getLearningRate(), 1e-12);
        assertEquals(original.getEpochs(), loaded.getEpochs());
        assertEquals(original.getBatchSize(), loaded.getBatchSize());
        assertEquals(original.getDatasetPath(), loaded.getDatasetPath());
        assertEquals(original.getOptimizerType(), loaded.getOptimizerType());

        assertNotNull(loaded.getOrCreateModel(), "Loaded config should recreate an MLP");
        assertNotNull(loaded.getOrCreateOptimizer(), "Loaded config should recreate an Optimizer");
    }
}

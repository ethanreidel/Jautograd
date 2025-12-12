package unit_tests;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;
import nn.ModelConfig;
import nn.OptimizerType;
import java.util.Arrays;

class ModelConfigTests {

    @Test
    @DisplayName("ModelConfig builder sets fields")
    void builderSetsFields() {
        ModelConfig cfg = new ModelConfig.Builder()
            .numFeatures(4)
            .layerSizes(Arrays.asList(4, 2))
            .learningRate(0.001)
            .epochs(2)
            .batchSize(8)
            .datasetPath("xor.csv")
            .optimizerType(OptimizerType.SGD)
            .build();
        assertEquals(4, cfg.getNumFeatures());
        assertEquals(Arrays.asList(4, 2), cfg.getLayerSizes());
        assertEquals(0.001, cfg.getLearningRate(), 1e-9);
        assertEquals(2, cfg.getEpochs());
        assertEquals(8, cfg.getBatchSize());
        assertEquals("xor.csv", cfg.getDatasetPath());
        assertEquals(OptimizerType.SGD, cfg.getOptimizerType());
    }

    @Test
    @DisplayName("Empty layers throws")
    void emptyLayersThrows() {
        assertThrows(IllegalArgumentException.class, () -> new ModelConfig.Builder()
            .numFeatures(4)
            .layerSizes(Arrays.asList())
            .learningRate(0.001)
            .epochs(2)
            .batchSize(8)
            .datasetPath("xor.csv")
            .optimizerType(OptimizerType.SGD)
            .build());
    }
}
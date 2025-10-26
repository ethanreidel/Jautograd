

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.qr;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.mockito.Mockito.*;



public class ModelConfigTests {
    @Test
    @DisplayName("ModelConfig test define")
    void defineModelConfig() {
        ModelConfig cfg = ModelConfig.define(
            "MLP", //modeltype
            new int[]{64, 128, 10},
            new String[]{"relu", "softmax"},
            "adam",
            "toy",
            new Hyperparameters(0.001, 32, 10) //lr, batchsize, epochs
        );
        assertEquals("MLP", cfg.getModelType());
        assertArrayEquals(new int[]{64,128,10}, cfg.getLayers());
        assertArrayEquals(new String[]{"relu","softmax"}, cfg.getActivations());
        assertEquals("adam", cfg.getOptimizer());
        assertEquals("toy", cfg.getDataset());
        assertEquals(0.001, cfg.getHyperparams().getLearningRate());
    }

    @Test
    @DisplayName("Empty layers")
    void defineEmptyLayersthrows() {
        assertThrows(InvalidArgumentException.class, () -> ModelConfig.define(
            "MLP", new int[]{}, new String[]{}, "adam", "mnist", new Hyperparams(1e-3, 32, 10)
        ));
    }

    @Test
    @DisplayName("save writes JSON to disk")
    void saveModelConfig(@TempDir Path tmp) throws Exception {
        var cfg = ModelConfig.define(
            "MLP",
            new int[]{4,2},
            new String[]{"relu"},
            "adam",
            "toy",
            new Hyperparams(0.001, 8, 2)
        );
        Path path = tmp.resolve("model-config.json");
        cfg.save(path);
        assertTrue(Files.exists(path));
        String json = Files.readString(path);
        assertTrue(json.contains("\"modelType\":\"MLP\""));
    }

    
}

package integration_tests;


import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class DefineConfigIntegrationIT {

    @Test
    void defineSaveLoad() {
        var configs = new ConfigRepository.InMemory();
        var cfgSvc = new ModelConfigService(configs);

        var cfgId = cfgSvc.defineAndSave(
            "MLP", new int[]{784, 128, 10}, new String[]{"relu", "softmax"},
            "sgd", "xor",
            new Hyperparams(1e-3, 64, 3)
        );

        var loaded = cfgSvc.load(cfgId);
        assertEquals("MLP", loaded.getModelType());
        assertArrayEquals(new int[]{784,128,10}, loaded.getLayers());
        assertEquals("xor", loaded.getDataset());
    }
}

package unit_tests;

import data.CsvDataset;
import nn.MLP;
import nn.ModelConfig;
import nn.OptimizerType;
import nn.Trainer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ModelTests {

    @Test
    @DisplayName("train: XOR end-to-end reaches reasonable accuracy")
    void trainSuccess() throws Exception {
        // 1) Create a tiny XOR dataset on disk
        Path tmp = Files.createTempFile("xor_train_success", ".csv");
        String csv = String.join("\n",
                "x1,x2,label",
                "0,0,0",
                "0,1,1",
                "1,0,1",
                "1,1,0"
        );
        Files.writeString(tmp, csv);

        // 2) Load dataset with your CsvDataset (label column = "label")
        CsvDataset ds = new CsvDataset(tmp, "label");

        int numFeatures = 2;
        List<Integer> layerSizes = List.of(4, 1);   // small 2-4-1 network
        double learningRate = 0.1;
        int epochs = 300;
        int batchSize = 1;

        // 3) Build ModelConfig
        ModelConfig cfg = new ModelConfig.Builder()
                .numFeatures(numFeatures)
                .layerSizes(layerSizes)
                .learningRate(learningRate)
                .epochs(epochs)
                .batchSize(batchSize)
                .datasetPath(tmp.toString())
                .optimizerType(OptimizerType.SGD)
                .model(new MLP(numFeatures, layerSizes))
                .build();

        // 4) Train
        Trainer trainer = new Trainer(cfg, ds);
        trainer.train();

        double finalAcc = trainer.getLastAccuracy();
        double finalLoss = trainer.getLastLoss();

        assertFalse(Double.isNaN(finalAcc), "Final accuracy should be a number");
        assertFalse(Double.isNaN(finalLoss), "Final loss should be a number");
        assertTrue(finalAcc >= 0.75,
                "Expected XOR accuracy >= 0.75 but got " + finalAcc);
    }

    @Test
    @DisplayName("Trainer: null dataset throws IllegalArgumentException")
    void trainNullDataset() {
        int numFeatures = 1;
        List<Integer> layerSizes = List.of(1);
        ModelConfig cfg = new ModelConfig.Builder()
                .numFeatures(numFeatures)
                .layerSizes(layerSizes)
                .learningRate(0.1)
                .epochs(1)
                .batchSize(1)
                .datasetPath("dummy.csv")
                .optimizerType(OptimizerType.SGD)
                .model(new MLP(numFeatures, layerSizes))
                .build();

        assertThrows(IllegalArgumentException.class,
                () -> new Trainer(cfg, null),
                "Expected Trainer constructor to throw on null dataset");
    }

    @Test
    @DisplayName("Trainer: zero epochs should not crash (no training steps)")
    void trainZeroEpochsNoCrash() throws Exception {
        // Small dummy dataset
        Path tmp = Files.createTempFile("xor_zero_epochs", ".csv");
        String csv = String.join("\n",
                "x1,x2,label",
                "0,0,0",
                "0,1,1"
        );
        Files.writeString(tmp, csv);

        CsvDataset ds = new CsvDataset(tmp, "label");

        int numFeatures = 2;
        List<Integer> layerSizes = List.of(2, 1);

        ModelConfig cfg = new ModelConfig.Builder()
                .numFeatures(numFeatures)
                .layerSizes(layerSizes)
                .learningRate(0.1)
                .epochs(0)                // <= focus of this test
                .batchSize(1)
                .datasetPath(tmp.toString())
                .optimizerType(OptimizerType.SGD)
                .model(new MLP(numFeatures, layerSizes))
                .build();

        Trainer trainer = new Trainer(cfg, ds);
        // Should not throw, even if 0 epochs
        trainer.train();

        // With 0 epochs, lastLoss/lastAccuracy may remain NaN; just ensure no crash
        assertTrue(true, "Training with 0 epochs completed without exception");
    }
}

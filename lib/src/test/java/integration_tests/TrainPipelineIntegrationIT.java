package integration_tests;

import data.CsvDataset;
import nn.ModelConfig;
import nn.MLP;
import nn.OptimizerType;
import nn.Trainer;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class TrainPipelineIntegrationIT {

    @Test
    @DisplayName("End-to-end training on tiny XOR dataset")
    void trainEnd2End() throws Exception {
        Path tmp = Files.createTempFile("xor_dataset", ".csv");
        String csv = String.join("\n",
                "x1,x2,label",
                "0,0,0",
                "0,1,1",
                "1,0,1",
                "1,1,0"
        );
        Files.writeString(tmp, csv);

        CsvDataset ds = new CsvDataset(tmp, "label");

        int numFeatures = 2;
        List<Integer> layerSizes = List.of(4, 1);

        double learningRate = 0.1;
        int epochs = 300;
        int batchSize = 1;

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

        Trainer trainer = new Trainer(cfg, ds);
        trainer.train();

        double finalAcc = trainer.getLastAccuracy();
        double finalLoss = trainer.getLastLoss();

        assertFalse(Double.isNaN(finalAcc), "Final accuracy should be a real number");
        assertFalse(Double.isNaN(finalLoss), "Final loss should be a real number");

        assertTrue(finalAcc >= 0.75,
                "Expected accuracy >= 0.75 on XOR, but was " + finalAcc);
    }
}

package integration_tests;

import data.CsvDataset;
import data.DataLoader;
import data.Batch;
import nn.ConfusionMatrix;
import nn.ModelConfig;
import nn.MLP;
import nn.OptimizerType;
import nn.Trainer;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class EvaluateModelIntegrationIT {

    @Test
    @DisplayName("Train model and evaluate with ConfusionMatrix")
    void evaluate_confusion_matrix_stored() throws Exception {
        Path tmp = Files.createTempFile("xor_dataset_eval", ".csv");
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
        List<Integer> layerSizes = List.of(16, 16, 1);
        double learningRate = 0.01;
        int epochs = 100;
        int batchSize = 2;

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

        ConfusionMatrix cm = new ConfusionMatrix();

        DataLoader loader = new DataLoader(batchSize, false);
        List<Batch> batches = loader.loadData(ds);

        final double threshold = 0.5;
        final boolean outputsAreLogits = true;

        int total = 0;
        int correct = 0;

        for (Batch b : batches) {
            List<Integer> labels = b.label();
            List<List<scalar.Scalar>> features = b.features();
            int featureCount = features.size();
            int size = features.get(0).size();

            for (int i = 0; i < size; i++) {
                List<scalar.Scalar> sample = new ArrayList<>(featureCount);
                for (int f = 0; f < featureCount; f++) {
                    sample.add(features.get(f).get(i));
                }

                MLP model = cfg.getOrCreateModel();
                var output = model.forward(sample).get(0);

                double val = output.data();
                double p = outputsAreLogits ? (1.0 / (1.0 + Math.exp(-val))) : val;
                int pred = (p >= threshold) ? 1 : 0;

                int label = labels.get(i);
                if (pred == label) correct++;
                total++;

                cm.update(label, pred);
            }
        }

        double accuracy = (total == 0) ? 0.0 : correct / (double) total;

        assertTrue(accuracy >= 0.0 && accuracy <= 1.0, "Accuracy should be in [0,1]");
        assertNotNull(cm, "ConfusionMatrix should not be null");

        assertTrue(accuracy >= 0.75,
                "Expected accuracy >= 0.75 on XOR evaluation, but was " + accuracy);

        int sum = cm.getTP() + cm.getFP() + cm.getFN() + cm.getTN();
        assertEquals(total, sum, "Sum of confusion matrix entries should equal number of samples");
    }
}

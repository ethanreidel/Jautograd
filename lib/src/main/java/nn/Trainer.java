package nn;


import data.*;
import nn.*;
import scalar.*;
import java.util.*;


public class Trainer {
    private final Optimizer optim;
    private final MLP model;
    private final int epochs;
    private final DataLoader loader;
    private final CsvDataset ds;
    private final List<Batch> data;
    
    public Trainer(Optimizer optim, MLP model, int epochs, DataLoader loader, CsvDataset ds) {
        this.optim = optim;
        this.model = model;
        this.epochs = epochs;
        this.loader = loader;
        this.ds = ds;
        this.data = loader.loadData(ds);
    }

    public void train() {
        double threshold = 0.5;
        final boolean outputsAreLogits = false;
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0f;
            int count = 0;
            int numCorrect = 0;

            for (Batch b : data) {
                List<Integer> labels = b.label();
                List<List<Scalar>> features = b.features();

                for (int i = 0; i < features.size(); i++) {
                    List<Scalar> sample = features.get(i);
                    List<Scalar> output = model.forward(sample);
                    Scalar out = output.get(0);

                    int label = labels.get(i);
                    double val = out.data();
                    double p = outputsAreLogits ? (1.0 / (1.0 + Math.exp(-val))) : val;
                    int pred = (p >= threshold) ? 1 : 0;
                    if (pred == label) numCorrect++;

                    Scalar loss = out.sub(new Scalar(label)).pow(2.0);
                    model.zeroGrad();
                    loss.backward();
                    optim.step(model.parameters());

                    epochLoss += (float) loss.data();
                    count++;
                }
            }
            System.out.println("accuracy: " + (numCorrect / (double)count));
            float avgLoss = count == 0 ? 0f : epochLoss / count;
            System.out.printf("Epoch %d, Loss: %.6f\n", epoch+1, avgLoss);
        }
    }       


}

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
    private List<Batch> data;

    private double lastLoss = Double.NaN;
    private double lastAccuracy = Double.NaN;
    private int demoDelayMs = 25;

    private final List<TrainingObserver> observers = new ArrayList<>();

    public Trainer(Optimizer optim, MLP model, int epochs, DataLoader loader, CsvDataset ds) {
        this.optim = optim;
        this.model = model;
        this.epochs = epochs;
        this.loader = loader;
        this.ds = ds;
        this.data = loader.loadData(ds);
    }

    public Trainer(ModelConfig config, CsvDataset ds) {
        if (ds == null) throw new IllegalArgumentException("Dataset is null");

        Optimizer opt = config.getOrCreateOptimizer();
        if (opt == null) opt = new SGDOptimizer(config.getLearningRate());

        MLP mdl = config.getOrCreateModel();

        this.optim = opt;
        this.model = mdl;
        this.epochs = config.getEpochs();
        this.ds = ds;

        int bs = config.getBatchSize() > 0 ? config.getBatchSize() : 32;
        this.loader = new DataLoader(bs, false);
        this.data = loader.loadData(ds);
    }

    public void addObserver(TrainingObserver observer) {
        if (observer != null) {
            observers.add(observer);
        }
    }

    public void removeObserver(TrainingObserver observer) {
        observers.remove(observer);
    }

    private void notifyStep(int globalStep) {
        for (TrainingObserver o : observers) {
            o.onStep(globalStep, lastLoss, lastAccuracy);
        }
    }

    private void notifyEpoch(int epoch,
                             ConfusionMatrix cm,
                             double avgLoss,
                             double accuracy) {
        for (TrainingObserver o : observers) {
            o.onEpoch(epoch, cm, avgLoss, accuracy);
        }
    }

    public double getLastLoss()     { return lastLoss; }
    public double getLastAccuracy() { return lastAccuracy; }

    public void setDemoDelayMs(int ms) {
        this.demoDelayMs = Math.max(0, ms);
    }

    private Scalar softplus(Scalar z) {
        Scalar abs = z.abs();
        return Scalar.max(z, new Scalar(0.0))
                     .add(abs.neg().exp().add(new Scalar(1.0)).log());
    }

    private Scalar bceWithLogits(Scalar logit, int y) {
        return softplus(logit).sub(logit.mul(new Scalar(y)));
    }

    public void train() {
        final double  threshold         = 0.5;
        final boolean outputsAreLogits  = true;

        ConfusionMatrix confmat = new ConfusionMatrix();
        int globalStep = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            data = loader.loadData(ds);
            confmat.reset();
            float epochLoss = 0f;
            int count = 0;
            int numCorrect = 0;

            for (Batch b : data) {
                List<Integer> labels = b.label();
                List<List<Scalar>> features = b.features();
                int numFeatures = features.size();
                int batchSize = features.get(0).size();

                for (int i = 0; i < batchSize; i++) {
                    List<Scalar> sample = new ArrayList<>(numFeatures);
                    for (int f = 0; f < numFeatures; f++) {
                        sample.add(features.get(f).get(i));
                    }

                    List<Scalar> output = model.forward(sample);
                    Scalar out = output.get(0);

                    int label = labels.get(i);
                    double val = out.data();
                    double p = outputsAreLogits ? (1.0 / (1.0 + Math.exp(-val))) : val;
                    int pred = (p >= threshold) ? 1 : 0;

                    if (pred == label) numCorrect++;
                    confmat.update(label, pred);

                    Scalar loss = bceWithLogits(out, label);
                    model.zeroGrad();
                    loss.backward();
                    optim.step(model.parameters());

                    epochLoss += (float) loss.data();
                    count++;

                    this.lastLoss = loss.data();
                    this.lastAccuracy = numCorrect / (double) count;

                    globalStep++;
                    notifyStep(globalStep);

                    if (demoDelayMs > 0) {
                        try {
                            Thread.sleep(demoDelayMs);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                    }
                }
            }

            double acc = (count == 0) ? 0.0 : numCorrect / (double) count;
            float avgLoss = (count == 0) ? 0f : epochLoss / count;
            System.out.printf("Epoch %d, Acc: %.4f, Loss: %.6f%n", epoch + 1, acc, avgLoss);
            notifyEpoch(epoch + 1, confmat, avgLoss, acc);
        }
    }
}

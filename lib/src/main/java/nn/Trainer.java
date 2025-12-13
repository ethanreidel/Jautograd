package nn;

import data.Batch;
import data.CsvDataset;
import data.DataLoader;
import java.util.ArrayList;
import java.util.List;
import scalar.Scalar;

/**
 * Training loop for an {@link MLP} on a {@link CsvDataset}.
 */
public class Trainer {

    private static final double DEFAULT_THRESHOLD = 0.5;
    private static final boolean OUTPUTS_ARE_LOGITS = true;
    private static final int DEFAULT_BATCH_SIZE = 32;

    private final Optimizer optimizer;
    private final MLP model;
    private final int epochs;
    private final DataLoader dataLoader;
    private final CsvDataset dataset;
    private final LearningRateScheduler learningRateScheduler;

    private List<Batch> batches;

    private double lastLoss = Double.NaN;
    private double lastAccuracy = Double.NaN;
    private int demoDelayMs = 25;

    private final List<TrainingObserver> observers = new ArrayList<>();

    public Trainer(
            Optimizer optimizer,
            MLP model,
            int epochs,
            DataLoader dataLoader,
            CsvDataset dataset,
            LearningRateScheduler scheduler) {

        if (optimizer == null) {
            throw new IllegalArgumentException("optimizer must not be null");
        }
        if (model == null) {
            throw new IllegalArgumentException("model must not be null");
        }
        if (dataLoader == null) {
            throw new IllegalArgumentException("dataLoader must not be null");
        }
        if (dataset == null) {
            throw new IllegalArgumentException("dataset must not be null");
        }
        if (epochs < 0) {
            throw new IllegalArgumentException("epochs must be >= 0");
        }

        this.optimizer = optimizer;
        this.model = model;
        this.epochs = epochs;
        this.dataLoader = dataLoader;
        this.dataset = dataset;
        this.learningRateScheduler = scheduler;
        this.batches = dataLoader.loadData(dataset);
    }

    public Trainer(ModelConfig config, CsvDataset dataset) {
        if (dataset == null) {
            throw new IllegalArgumentException("dataset must not be null");
        }
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }

        Optimizer optimizer = config.getOrCreateOptimizer();
        if (optimizer == null) {
            optimizer = new SGDOptimizer(config.getLearningRate());
        }

        MLP model = config.getOrCreateModel();

        this.optimizer = optimizer;
        this.model = model;
        this.epochs = config.getEpochs();
        this.dataset = dataset;

        int batchSize =
                config.getBatchSize() > 0 ? config.getBatchSize() : DEFAULT_BATCH_SIZE;
        this.dataLoader = new DataLoader(batchSize, false);
        this.batches = dataLoader.loadData(dataset);
        this.learningRateScheduler = config.getLearningRateScheduler();
    }

    public void addObserver(TrainingObserver observer) {
        if (observer != null) {
            observers.add(observer);
        }
    }

    public void removeObserver(TrainingObserver observer) {
        observers.remove(observer);
    }

    public double getLastLoss() {
        return lastLoss;
    }

    public double getLastAccuracy() {
        return lastAccuracy;
    }

    public void setDemoDelayMs(int ms) {
        this.demoDelayMs = Math.max(0, ms);
    }

    public void train() {
        ConfusionMatrix confusionMatrix = new ConfusionMatrix();
        int globalStep = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            prepareEpoch(confusionMatrix);
            EpochStats stats = runEpoch(epoch, confusionMatrix, globalStep);
            globalStep = stats.globalStep();

            System.out.printf(
                    "Epoch %d, Acc: %.4f, Loss: %.2f%n",
                    epoch + 1,
                    stats.accuracy(),
                    stats.averageLoss());

            notifyEpoch(epoch + 1, confusionMatrix, stats.averageLoss(), stats.accuracy());
        }
    }

    private void prepareEpoch(ConfusionMatrix confusionMatrix) {
        batches = dataLoader.loadData(dataset);
        confusionMatrix.reset();
    }

    private EpochStats runEpoch(int epoch, ConfusionMatrix confusionMatrix, int startGlobalStep) {
        double epochLoss = 0.0;
        int count = 0;
        int numCorrect = 0;
        int globalStep = startGlobalStep;

        updateLearningRateIfNeeded(epoch);

        for (Batch batch : batches) {
            List<Integer> labels = batch.label();
            List<List<Scalar>> features = batch.features();

            int numFeatures = features.size();
            int batchSize = features.get(0).size();

            for (int i = 0; i < batchSize; i++) {
                List<Scalar> sample = buildSample(features, numFeatures, i);

                Scalar logit = forwardSample(sample);
                int label = labels.get(i);

                int prediction = predictLabel(logit, label);
                if (prediction == label) {
                    numCorrect++;
                }
                confusionMatrix.update(label, prediction);

                Scalar loss = computeLossAndBackpropagate(logit, label);
                optimizer.step(model.parameters());

                epochLoss += loss.data();
                count++;

                lastLoss = loss.data();
                lastAccuracy = numCorrect / (double) count;

                globalStep++;
                notifyStep(globalStep);

                maybeSleepForDemo();
            }
        }

        double accuracy = (count == 0) ? 0.0 : numCorrect / (double) count;
        double averageLoss = (count == 0) ? 0.0 : epochLoss / count;
        return new EpochStats(globalStep, accuracy, averageLoss);
    }

    private void updateLearningRateIfNeeded(int epoch) {
        if (learningRateScheduler != null && optimizer instanceof AbstractOptimizer ao) {
            double newLearningRate = learningRateScheduler.getLearningRate(epoch);
            ao.setLearningRate(newLearningRate);
        }
    }

    private List<Scalar> buildSample(
            List<List<Scalar>> features, int numFeatures, int indexInBatch) {

        List<Scalar> sample = new ArrayList<>(numFeatures);
        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            sample.add(features.get(featureIndex).get(indexInBatch));
        }
        return sample;
    }

    private Scalar forwardSample(List<Scalar> sample) {
        List<Scalar> outputs = model.forward(sample);
        return outputs.get(0);
    }

    private int predictLabel(Scalar logitOrProb, int label) {
        double value = logitOrProb.data();
        double probability = OUTPUTS_ARE_LOGITS
                ? 1.0 / (1.0 + Math.exp(-value))
                : value;
        return (probability >= DEFAULT_THRESHOLD) ? 1 : 0;
    }

    private Scalar computeLossAndBackpropagate(Scalar logit, int label) {
        Scalar loss = binaryCrossEntropyWithLogits(logit, label);
        model.zeroGrad();
        loss.backward();
        return loss;
    }

    private void notifyStep(int globalStep) {
        for (TrainingObserver observer : observers) {
            observer.onStep(globalStep, lastLoss, lastAccuracy);
        }
    }

    private void notifyEpoch(
            int epoch,
            ConfusionMatrix confusionMatrix,
            double avgLoss,
            double accuracy) {

        for (TrainingObserver observer : observers) {
            observer.onEpoch(epoch, confusionMatrix, avgLoss, accuracy);
        }
    }

    private void maybeSleepForDemo() {
        if (demoDelayMs <= 0) {
            return;
        }

        try {
            Thread.sleep(demoDelayMs);
        } catch (InterruptedException interruptedException) {
            Thread.currentThread().interrupt();
        }
    }

    private Scalar softplus(Scalar z) {
        Scalar abs = z.abs();
        // max(z, 0) + log(1 + exp(-|z|)) (numerically stable softplus)
        return Scalar.max(z, new Scalar(0.0))
                .add(abs.neg().exp().add(new Scalar(1.0)).log());
    }

    private Scalar binaryCrossEntropyWithLogits(Scalar logit, int label) {
        return softplus(logit).sub(logit.mul(new Scalar(label)));
    }

    /**
     * Simple record-like holder for epoch statistics.
     */
    private static final class EpochStats {
        private final int globalStep;
        private final double accuracy;
        private final double averageLoss;

        EpochStats(int globalStep, double accuracy, double averageLoss) {
            this.globalStep = globalStep;
            this.accuracy = accuracy;
            this.averageLoss = averageLoss;
        }

        int globalStep() {
            return globalStep;
        }

        double accuracy() {
            return accuracy;
        }

        double averageLoss() {
            return averageLoss;
        }
    }
}

package ui;

import java.util.concurrent.ConcurrentLinkedQueue;
import javafx.animation.AnimationTimer;
import javafx.application.Platform;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.layout.VBox;
import nn.ConfusionMatrix;
import nn.TrainingObserver;

/**
 * JavaFX component that draws training loss and accuracy over time.
 *
 * <p>It implements {@link TrainingObserver}, so the Trainer can push
 * step-by-step metrics, even from a background thread. Updates are
 * buffered in a queue and rendered on the JavaFX Application Thread
 * via an {@link AnimationTimer}.
 */
public class DrawMetricGraph extends VBox implements TrainingObserver {

    private static final int MAX_POINTS = 2000;
    private static final double SPACING = 8.0;

    private final LineChart<Number, Number> lossChart;
    private final LineChart<Number, Number> accuracyChart;

    private final XYChart.Series<Number, Number> lossSeries =
            new XYChart.Series<>();
    private final XYChart.Series<Number, Number> accuracySeries =
            new XYChart.Series<>();

    // Thread-safe queue for incoming points (from Trainer)
    private final ConcurrentLinkedQueue<Point> inbox =
            new ConcurrentLinkedQueue<>();

    private final AnimationTimer animationTimer;

    /**
     * Small value object representing a single training step's metrics.
     */
    private static final class Point {
        final int step;
        final double loss;
        final double accuracy;

        Point(int step, double loss, double accuracy) {
            this.step = step;
            this.loss = loss;
            this.accuracy = accuracy;
        }
    }

    public DrawMetricGraph() {
        super(SPACING);

        this.lossChart = createLossChart();
        this.accuracyChart = createAccuracyChart();

        getChildren().addAll(lossChart, accuracyChart);

        this.animationTimer = new AnimationTimer() {
            @Override
            public void handle(long now) {
                flushQueue();
            }
        };
        this.animationTimer.start();
    }

    @Override
    public void onStep(int globalStep, double loss, double accuracy) {
        push(globalStep, loss, accuracy);
    }

    @Override
    public void onEpoch(
            int epoch,
            ConfusionMatrix confusionMatrix,
            double avgLoss,
            double accuracy) {
    }

    /**
     * Enqueue a new point to be drawn. Can be safely called from any thread.
     */
    public void push(int step, double loss, double accuracy) {
        inbox.offer(new Point(step, loss, accuracy));
    }

    /**
     * Clears both series from the charts.
     */
    public void clear() {
        Runnable clearTask = () -> {
            lossSeries.getData().clear();
            accuracySeries.getData().clear();
        };

        if (Platform.isFxApplicationThread()) {
            clearTask.run();
        } else {
            Platform.runLater(clearTask);
        }
    }

    private LineChart<Number, Number> createLossChart() {
        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis();

        xAxis.setLabel("Step");
        yAxis.setLabel("Loss");
        xAxis.setForceZeroInRange(false);
        yAxis.setForceZeroInRange(false);

        LineChart<Number, Number> chart =
                new LineChart<>(xAxis, yAxis);
        chart.setTitle("Training Loss");
        chart.setCreateSymbols(false);
        chart.setAnimated(false);

        lossSeries.setName("Loss");
        chart.getData().add(lossSeries);

        return chart;
    }

    private LineChart<Number, Number> createAccuracyChart() {
        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis(0.0, 1.0, 0.1);

        xAxis.setLabel("Step");
        yAxis.setLabel("Accuracy");
        xAxis.setForceZeroInRange(false);
        yAxis.setForceZeroInRange(false);

        LineChart<Number, Number> chart =
                new LineChart<>(xAxis, yAxis);
        chart.setTitle("Training Accuracy");
        chart.setCreateSymbols(false);
        chart.setAnimated(false);

        accuracySeries.setName("Accuracy");
        chart.getData().add(accuracySeries);

        return chart;
    }

    /**
     * Flushes pending points from the inbox queue and adds them to the charts.
     *
     * <p>This is called on every frame by the {@link AnimationTimer} on the
     * JavaFX Application Thread.
     */
    private void flushQueue() {
        Point point;
        while ((point = inbox.poll()) != null) {
            lossSeries.getData().add(
                    new XYChart.Data<>(point.step, point.loss));
            accuracySeries.getData().add(
                    new XYChart.Data<>(point.step, point.accuracy));

            trim(lossSeries);
            trim(accuracySeries);
        }
    }

    /**
     * Ensures the series doesn't grow beyond {@link #MAX_POINTS}.
     */
    private void trim(XYChart.Series<Number, Number> series) {
        int size = series.getData().size();
        if (size > MAX_POINTS) {
            series.getData().remove(0, size - MAX_POINTS);
        }
    }
}

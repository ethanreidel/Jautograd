package ui;

import javafx.animation.AnimationTimer;
import javafx.application.Platform;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.layout.VBox;

import java.util.concurrent.ConcurrentLinkedQueue;

public class DrawMetricGraph extends VBox {

    private final LineChart<Number, Number> lossChart;
    private final LineChart<Number, Number> accChart;
    private final XYChart.Series<Number, Number> lossSeries = new XYChart.Series<>();
    private final XYChart.Series<Number, Number> accSeries  = new XYChart.Series<>();

    private final ConcurrentLinkedQueue<Point> inbox = new ConcurrentLinkedQueue<>();
    private static final int MAX_POINTS = 2000;

    private static final class Point {
        final int step; final double loss; final double acc;
        Point(int step, double loss, double acc) { this.step = step; this.loss = loss; this.acc = acc; }
    }

    public DrawMetricGraph() {
        super(8.0);

        NumberAxis x1 = new NumberAxis();
        NumberAxis y1 = new NumberAxis();
        x1.setLabel("Step");
        y1.setLabel("Loss");
        x1.setForceZeroInRange(false);
        y1.setForceZeroInRange(false);
        lossChart = new LineChart<>(x1, y1);
        lossChart.setTitle("Training Loss");
        lossChart.setCreateSymbols(false);
        lossChart.setAnimated(false);
        lossSeries.setName("Loss");
        lossChart.getData().add(lossSeries);

        NumberAxis x2 = new NumberAxis();
        NumberAxis y2 = new NumberAxis(0, 1, 0.1);
        x2.setLabel("Step");
        y2.setLabel("Accuracy");
        x2.setForceZeroInRange(false);
        y2.setForceZeroInRange(false);
        accChart = new LineChart<>(x2, y2);
        accChart.setTitle("Training Accuracy");
        accChart.setCreateSymbols(false);
        accSeries.setName("Accuracy");
        accChart.getData().add(accSeries);

        getChildren().addAll(lossChart, accChart);

        new AnimationTimer() {
            @Override public void handle(long now) {
                boolean updated = false;
                Point p;
                while ((p = inbox.poll()) != null) {
                    lossSeries.getData().add(new XYChart.Data<>(p.step, p.loss));
                    accSeries.getData().add(new XYChart.Data<>(p.step, p.acc));
                    trim(lossSeries);
                    trim(accSeries);
                    updated = true;
                }
                if (updated) {
                    // charts auto-refresh when data changes
                }
            }
        }.start();
    }

    public void push(int step, double loss, double accuracy) {
        inbox.offer(new Point(step, loss, accuracy));
    }

    public void clear() {
        Runnable r = () -> {
            lossSeries.getData().clear();
            accSeries.getData().clear();
        };
        if (Platform.isFxApplicationThread()) r.run(); else Platform.runLater(r);
    }

    private void trim(XYChart.Series<Number, Number> s) {
        if (s.getData().size() > MAX_POINTS) {
            s.getData().remove(0, s.getData().size() - MAX_POINTS);
        }
    }
}
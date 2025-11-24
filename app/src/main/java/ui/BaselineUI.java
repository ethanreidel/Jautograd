package ui;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.List;
import java.util.ArrayList;
import java.util.stream.Collectors;



import data.CsvDataset;
import nn.*;

public class BaselineUI extends Application {
    private final BooleanProperty configLoaded = new SimpleBooleanProperty(false);
    private ModelConfig loaded = null;
    private File lastDir = null;
    private CsvDataset dataset = null;

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Model Trainer");

        VBox root = new VBox(10);
        root.setStyle("-fx-padding: 20;");

        Button loadConfigButton = new Button("Load Config YAML");

        Label title = new Label("Model Training Interface");
        title.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");

        TextField datasetPath = new TextField();
        datasetPath.setPromptText("Dataset path (CSV)");

        TextField inputSize = new TextField();
        inputSize.setPromptText("Feature count (e.g. 2)");

        TextField labelColumn = new TextField();
        labelColumn.setPromptText("Dataset label");

        TextField batchSize = new TextField();
        batchSize.setPromptText("Batch size");

        TextField layers = new TextField();
        layers.setPromptText("Layers (comma separated, e.g. 16,16,1)");

        TextField learningRate = new TextField();
        learningRate.setPromptText("Learning rate (e.g. 0.01)");

        TextField epochs = new TextField();
        epochs.setPromptText("Epochs (e.g. 100)");

        Button trainButton = new Button("Train Model");

        DrawMetricGraph graphs = new DrawMetricGraph();

        TextArea cmArea = new TextArea();
        cmArea.setEditable(false);
        cmArea.setPrefRowCount(6);
        cmArea.setPromptText("Confusion matrix and epoch metrics will appear here…");


        loadConfigButton.setOnAction(e -> {
            if (configLoaded.get() && loaded != null) {
                new Alert(Alert.AlertType.INFORMATION, "Config already loaded.").showAndWait();
                return;
            }

            TextInputDialog pathDialog = new TextInputDialog();
            pathDialog.setTitle("Load Config");
            pathDialog.setHeaderText("Enter the path to your config YAML\n(or leave blank to browse)");
            pathDialog.setContentText("Path:");
            Optional<String> pathOpt = pathDialog.showAndWait();

            String chosenPath = null;
            if (pathOpt.isPresent()) {
                String raw = pathOpt.get().trim();
                if (!raw.isEmpty()) {
                    try {
                        Path p = Paths.get(raw).toAbsolutePath().normalize();
                        if (!Files.exists(p)) {
                            new Alert(Alert.AlertType.ERROR, "File does not exist:\n" + p).showAndWait();
                            return;
                        }
                        if (!Files.isRegularFile(p)) {
                            new Alert(Alert.AlertType.ERROR, "Not a regular file:\n" + p).showAndWait();
                            return;
                        }
                        chosenPath = p.toString();
                    } catch (InvalidPathException ex) {
                        new Alert(Alert.AlertType.ERROR, "Invalid path:\n" + raw + "\n\n" + ex.getMessage()).showAndWait();
                        return;
                    }
                }
            }

            if (chosenPath == null) {
                FileChooser fc = new FileChooser();
                fc.setTitle("Select Config YAML");
                fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("YAML Files", "*.yaml", "*.yml"));
                if (lastDir != null && lastDir.isDirectory()) {
                    fc.setInitialDirectory(lastDir);
                }
                File f = fc.showOpenDialog(primaryStage);
                if (f == null) return;
                chosenPath = f.getAbsolutePath();
                lastDir = f.getParentFile();
            }

            try {
                loaded = ModelConfig.readConfigYaml(chosenPath);
                configLoaded.set(true);

                if (loaded.getDatasetPath() != null) datasetPath.setText(loaded.getDatasetPath());
                if (loaded.getNumFeatures() > 0) inputSize.setText(String.valueOf(loaded.getNumFeatures()));
                if (loaded.getLayerSizes() != null && !loaded.getLayerSizes().isEmpty()) {
                    layers.setText(loaded.getLayerSizes().stream().map(Object::toString).collect(Collectors.joining(",")));
                }
                if (loaded.getLearningRate() > 0) learningRate.setText(String.valueOf(loaded.getLearningRate()));
                if (loaded.getEpochs() > 0) epochs.setText(String.valueOf(loaded.getEpochs()));
                if (loaded.getBatchSize() > 0) batchSize.setText(String.valueOf(loaded.getBatchSize()));

                new Alert(Alert.AlertType.INFORMATION, "Config loaded from:\n" + chosenPath).showAndWait();
            } catch (Exception ex) {
                new Alert(Alert.AlertType.ERROR, "Failed to load config:\n" + ex.getMessage()).showAndWait();
            }
        });

        trainButton.setOnAction(e2 -> {
            try {
                ModelConfig cfg;
                if (loaded == null) {
                    String ds = requireNonEmpty(datasetPath.getText(), "Dataset path");
                    int inSize = parseInt(requireNonEmpty(inputSize.getText(), "Input size"), "Input size");
                    List<Integer> layerSizesList = parseLayerList(requireNonEmpty(layers.getText(), "Layers"));
                    double lr = parseDouble(requireNonEmpty(learningRate.getText(), "Learning rate"), "Learning rate");
                    int eps = parseInt(requireNonEmpty(epochs.getText(), "Epochs"), "Epochs");
                    Integer bs = isBlank(batchSize.getText()) ? null : parseInt(batchSize.getText(), "Batch size");

                    String tgt = requireNonEmpty(labelColumn.getText(), "Target/label column");
                    dataset = new CsvDataset(Paths.get(ds), tgt);
                    try { lastDir = Paths.get(ds).toAbsolutePath().getParent().toFile(); } catch (Exception ignore) {}

                    Optimizer optim = new Optimizer(lr);
                    MLP model = new MLP(inSize, layerSizesList);

                    ModelConfig.Builder b = new ModelConfig.Builder()
                            .numFeatures(inSize)
                            .layerSizes(layerSizesList)
                            .learningRate(lr)
                            .epochs(eps)
                            .datasetPath(ds)
                            .optimizer(optim)
                            .model(model);
                    if (bs != null) b.batchSize(bs);
                    cfg = b.build();
                } else {
                    cfg = loaded;

                    if (!isBlank(datasetPath.getText())) cfg.setDatasetPath(datasetPath.getText().trim());
                    if (!isBlank(inputSize.getText())) cfg.setNumFeatures(parseInt(inputSize.getText().trim(), "Input size"));
                    if (!isBlank(layers.getText())) cfg.setLayerSizes(parseLayerList(layers.getText().trim()));
                    if (!isBlank(learningRate.getText())) cfg.setLearningRate(parseDouble(learningRate.getText().trim(), "Learning rate"));
                    if (!isBlank(epochs.getText())) cfg.setEpochs(parseInt(epochs.getText().trim(), "Epochs"));
                    if (!isBlank(batchSize.getText())) cfg.setBatchSize(parseInt(batchSize.getText().trim(), "Batch size"));

                    cfg = new ModelConfig.Builder()
                            .numFeatures(cfg.getNumFeatures())
                            .layerSizes(cfg.getLayerSizes())
                            .learningRate(cfg.getLearningRate())
                            .epochs(cfg.getEpochs())
                            .batchSize(cfg.getBatchSize())
                            .datasetPath(cfg.getDatasetPath())
                            .optimizer(new Optimizer(cfg.getLearningRate()))
                            .model(new MLP(cfg.getNumFeatures(), cfg.getLayerSizes()))
                            .build();

                    String dsPath = requireNonEmpty(cfg.getDatasetPath(), "Dataset path (from YAML/fields)");
                    String tgt = requireNonEmpty(labelColumn.getText(), "Target/label column");
                    dataset = new CsvDataset(Paths.get(dsPath), tgt);
                    try { lastDir = Paths.get(dsPath).toAbsolutePath().getParent().toFile(); } catch (Exception ignore) {}



                    loaded = cfg;
                }

                trainButton.setDisable(true);
                final ModelConfig runCfg = cfg;
                final CsvDataset runDataset = dataset;

                new Thread(() -> {
                    try {
                        Trainer trainer = new Trainer(runCfg, runDataset);

                        trainer.setStepListener(step -> {
                            double loss = trainer.getLastLoss();
                            double acc  = trainer.getLastAccuracy();
                            graphs.push(step, loss, acc);
                        });

                        trainer.setEpochListener((epoch, cm) -> {
                            double acc = cm.accuracy();
                            double prec = cm.precision();
                            double rec = cm.recall();
                            double f1 = cm.f1();
                            String txt = String.format(
                                    "Epoch %d%n" +
                                            "TP=%d FP=%d FN=%d TN=%d  |  Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f%n%n",
                                    epoch, cm.getTP(), cm.getFP(), cm.getFN(), cm.getTN(),
                                    acc, prec, rec, f1
                            );
                            Platform.runLater(() -> cmArea.appendText(txt));
                        });

                        trainer.train();

                        Platform.runLater(() ->
                                new Alert(Alert.AlertType.INFORMATION, "Training finished.").showAndWait()
                        );
                    } catch (Exception ex) {
                        Platform.runLater(() ->
                                new Alert(Alert.AlertType.ERROR, "Training error:\n" + ex.getMessage()).showAndWait()
                        );
                    } finally {
                        Platform.runLater(() -> trainButton.setDisable(false));
                    }
                }, "training-thread").start();

            } catch (IllegalArgumentException ex) {
                new Alert(Alert.AlertType.ERROR, ex.getMessage()).showAndWait();
            } catch (Exception ex) {
                new Alert(Alert.AlertType.ERROR, "Unexpected error:\n" + ex.getMessage()).showAndWait();
            }
        });

        root.getChildren().addAll(
                title,
                loadConfigButton, labelColumn,
                datasetPath, batchSize, inputSize, layers, learningRate, epochs,
                trainButton,
                graphs,
                new Label("Epoch summaries (confusion matrix):"),
                cmArea
        );


        Scene scene = new Scene(root, 480, 420);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private static String requireNonEmpty(String s, String fieldName) {
        if (s == null || s.trim().isEmpty()) {
            throw new IllegalArgumentException(fieldName + " is required.");
        }
        return s.trim();
    }

    private static boolean isBlank(String s) {
        return s == null || s.trim().isEmpty();
    }

    private static int parseInt(String s, String fieldName) {
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException nfe) {
            throw new IllegalArgumentException(fieldName + " must be an integer.");
        }
    }

    private static double parseDouble(String s, String fieldName) {
        try {
            return Double.parseDouble(s.trim());
        } catch (NumberFormatException nfe) {
            throw new IllegalArgumentException(fieldName + " must be a number.");
        }
    }

    private static List<Integer> parseLayerList(String csv) {
        try {
            String[] parts = csv.split(",");
            List<Integer> list = new ArrayList<>(parts.length);
            for (String p : parts) {
                String t = p.trim();
                if (!t.isEmpty()) list.add(Integer.parseInt(t));
            }
            if (list.isEmpty()) throw new IllegalArgumentException("Layers must not be empty.");
            return list;
        } catch (NumberFormatException nfe) {
            throw new IllegalArgumentException("Layers must be a comma-separated list of integers (e.g., 16,16,1).");
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}

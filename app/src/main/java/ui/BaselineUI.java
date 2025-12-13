package ui;

import data.Batch;
import data.CsvDataset;
import data.DataLoader;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.TextInputDialog;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import nn.ConfusionMatrix;
import nn.ExponentialDecayScheduler;
import nn.MLP;
import nn.ModelConfig;
import nn.OptimizerType;
import nn.Trainer;
import nn.TrainingObserver;
import nn.LearningRateScheduler;
import scalar.Scalar;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class BaselineUI extends Application {

    private static final int DEFAULT_WIDTH = 480;
    private static final int DEFAULT_HEIGHT = 500;

    private final BooleanProperty configLoaded = new SimpleBooleanProperty(false);

    private ModelConfig modelConfig;
    private File lastDirectory;
    private CsvDataset dataset;

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Model Trainer");

        VBox root = new VBox(10);
        root.setStyle("-fx-padding: 20;");

        Label titleLabel = createTitleLabel();
        Button loadConfigButton = new Button("Load Config YAML");

        TextField datasetPathField = createTextField("Dataset path (CSV)");
        TextField batchSizeField = createTextField("Batch size");
        TextField inputSizeField = createTextField("Feature count (e.g. 2)");
        TextField layersField = createTextField("Layers (comma separated, e.g. 16,16,1)");
        TextField learningRateField = createTextField("Learning rate (e.g. 0.01)");
        TextField epochsField = createTextField("Epochs (e.g. 100)");
        TextField labelColumnField = createTextField("Dataset label");
        TextField learningRateDecayField = createTextField("LR decay (exp, optional, e.g. 0.99)");

        ComboBox<OptimizerType> optimizerChoiceBox = createOptimizerChoiceBox();

        Button trainButton = new Button("Train Model");
        DrawMetricGraph metricGraphs = new DrawMetricGraph();

        TextArea confusionMatrixArea = createConfusionMatrixArea();

        Label predictionHeaderLabel = new Label("Prediction");
        Label predictionInputLabel = new Label("Enter input features to predict:");
        TextField predictionInputField =
                createTextField("Comma-separated features (e.g. 5.1,3.5)");
        Button predictionButton = new Button("Predict");
        Label predictionOutputLabel = new Label("Prediction: (run training first)");

        configureLoadConfigButton(
                primaryStage,
                loadConfigButton,
                datasetPathField,
                inputSizeField,
                layersField,
                learningRateField,
                epochsField,
                batchSizeField,
                optimizerChoiceBox
        );

        configureTrainButton(
                trainButton,
                metricGraphs,
                confusionMatrixArea,
                datasetPathField,
                batchSizeField,
                inputSizeField,
                layersField,
                learningRateField,
                epochsField,
                labelColumnField,
                learningRateDecayField,
                optimizerChoiceBox
        );

        configurePredictButton(
                predictionButton,
                predictionInputField,
                predictionOutputLabel
        );

        root.getChildren().addAll(
                titleLabel,
                loadConfigButton,
                labelColumnField,
                datasetPathField,
                batchSizeField,
                inputSizeField,
                layersField,
                learningRateField,
                epochsField,
                optimizerChoiceBox,
                learningRateDecayField,
                trainButton,
                metricGraphs,
                new Label("Epoch summaries (confusion matrix):"),
                confusionMatrixArea,
                predictionHeaderLabel,
                predictionInputLabel,
                predictionInputField,
                predictionButton,
                predictionOutputLabel
        );

        Scene scene = new Scene(root, DEFAULT_WIDTH, DEFAULT_HEIGHT);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private Label createTitleLabel() {
        Label title = new Label("Model Training Interface");
        title.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        return title;
    }

    private TextField createTextField(String promptText) {
        TextField textField = new TextField();
        textField.setPromptText(promptText);
        return textField;
    }

    private ComboBox<OptimizerType> createOptimizerChoiceBox() {
        ComboBox<OptimizerType> comboBox = new ComboBox<>();
        comboBox.getItems().setAll(OptimizerType.values());
        comboBox.setPromptText("Optimizer");
        comboBox.setValue(OptimizerType.SGD);
        return comboBox;
    }

    private TextArea createConfusionMatrixArea() {
        TextArea textArea = new TextArea();
        textArea.setEditable(false);
        textArea.setPrefRowCount(6);
        textArea.setPromptText(
                "Confusion matrix and epoch metrics will appear here…"
        );
        return textArea;
    }

    private void configureLoadConfigButton(
            Stage primaryStage,
            Button loadConfigButton,
            TextField datasetPathField,
            TextField inputSizeField,
            TextField layersField,
            TextField learningRateField,
            TextField epochsField,
            TextField batchSizeField,
            ComboBox<OptimizerType> optimizerChoiceBox
    ) {
        loadConfigButton.setOnAction(event -> {
            if (configLoaded.get() && modelConfig != null) {
                showInfoAlert("Config already loaded.");
                return;
            }

            String chosenPath = askForConfigPath(primaryStage);
            if (chosenPath == null) {
                return;
            }

            try {
                modelConfig = ModelConfig.readConfigYaml(chosenPath);
                configLoaded.set(true);

                populateFieldsFromConfig(
                        modelConfig,
                        datasetPathField,
                        inputSizeField,
                        layersField,
                        learningRateField,
                        epochsField,
                        batchSizeField,
                        optimizerChoiceBox
                );

                showInfoAlert("Config loaded from:\n" + chosenPath);
            } catch (Exception ex) {
                showErrorAlert("Failed to load config:\n" + ex.getMessage());
            }
        });
    }

    private String askForConfigPath(Stage primaryStage) {
        TextInputDialog pathDialog = new TextInputDialog();
        pathDialog.setTitle("Load Config");
        pathDialog.setHeaderText(
                "Enter the path to your config YAML\n(or leave blank to browse)"
        );
        pathDialog.setContentText("Path:");
        Optional<String> pathOptional = pathDialog.showAndWait();

        String chosenPath = null;

        if (pathOptional.isPresent()) {
            String rawPath = pathOptional.get().trim();
            if (!rawPath.isEmpty()) {
                chosenPath = validatePathOrShowError(rawPath);
                if (chosenPath == null) {
                    return null;
                }
            }
        }

        if (chosenPath == null) {
            File selectedFile =
                    showConfigFileChooser(primaryStage);
            if (selectedFile == null) {
                return null;
            }
            chosenPath = selectedFile.getAbsolutePath();
            lastDirectory = selectedFile.getParentFile();
        }

        return chosenPath;
    }

    private String validatePathOrShowError(String rawPath) {
        try {
            Path path = Paths.get(rawPath).toAbsolutePath().normalize();
            if (!Files.exists(path)) {
                showErrorAlert("File does not exist:\n" + path);
                return null;
            }
            if (!Files.isRegularFile(path)) {
                showErrorAlert("Not a regular file:\n" + path);
                return null;
            }
            return path.toString();
        } catch (InvalidPathException ex) {
            showErrorAlert(
                    "Invalid path:\n" + rawPath + "\n\n" + ex.getMessage()
            );
            return null;
        }
    }

    private File showConfigFileChooser(Stage primaryStage) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select Config YAML");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("YAML Files", "*.yaml", "*.yml")
        );
        if (lastDirectory != null && lastDirectory.isDirectory()) {
            fileChooser.setInitialDirectory(lastDirectory);
        }
        return fileChooser.showOpenDialog(primaryStage);
    }

    private void populateFieldsFromConfig(
            ModelConfig config,
            TextField datasetPathField,
            TextField inputSizeField,
            TextField layersField,
            TextField learningRateField,
            TextField epochsField,
            TextField batchSizeField,
            ComboBox<OptimizerType> optimizerChoiceBox
    ) {
        if (config.getDatasetPath() != null) {
            datasetPathField.setText(config.getDatasetPath());
        }
        if (config.getNumFeatures() > 0) {
            inputSizeField.setText(String.valueOf(config.getNumFeatures()));
        }
        if (config.getLayerSizes() != null && !config.getLayerSizes().isEmpty()) {
            String layerText = config.getLayerSizes()
                    .stream()
                    .map(Object::toString)
                    .collect(Collectors.joining(","));
            layersField.setText(layerText);
        }
        if (config.getLearningRate() > 0) {
            learningRateField.setText(String.valueOf(config.getLearningRate()));
        }
        if (config.getEpochs() > 0) {
            epochsField.setText(String.valueOf(config.getEpochs()));
        }
        if (config.getBatchSize() > 0) {
            batchSizeField.setText(String.valueOf(config.getBatchSize()));
        }

        if (config.getOptimizerType() != null) {
            optimizerChoiceBox.setValue(config.getOptimizerType());
        }
    }

    private void configureTrainButton(
            Button trainButton,
            DrawMetricGraph metricGraphs,
            TextArea confusionMatrixArea,
            TextField datasetPathField,
            TextField batchSizeField,
            TextField inputSizeField,
            TextField layersField,
            TextField learningRateField,
            TextField epochsField,
            TextField labelColumnField,
            TextField learningRateDecayField,
            ComboBox<OptimizerType> optimizerChoiceBox
    ) {
        trainButton.setOnAction(event -> {
            try {
                ModelConfig configToRun =
                        buildOrUpdateModelConfig(
                                datasetPathField,
                                batchSizeField,
                                inputSizeField,
                                layersField,
                                learningRateField,
                                epochsField,
                                labelColumnField,
                                learningRateDecayField,
                                optimizerChoiceBox
                        );

                trainButton.setDisable(true);
                metricGraphs.clear();
                confusionMatrixArea.clear();

                final ModelConfig runConfig = configToRun;
                final CsvDataset runDataset = dataset;

                startTrainingThread(
                        trainButton,
                        metricGraphs,
                        confusionMatrixArea,
                        runConfig,
                        runDataset
                );

            } catch (IllegalArgumentException ex) {
                showErrorAlert(ex.getMessage());
            } catch (Exception ex) {
                showErrorAlert("Unexpected error:\n" + ex.getMessage());
            }
        });
    }

    private ModelConfig buildOrUpdateModelConfig(
            TextField datasetPathField,
            TextField batchSizeField,
            TextField inputSizeField,
            TextField layersField,
            TextField learningRateField,
            TextField epochsField,
            TextField labelColumnField,
            TextField learningRateDecayField,
            ComboBox<OptimizerType> optimizerChoiceBox
    ) {
        if (modelConfig == null) {
            return buildNewConfig(
                    datasetPathField,
                    batchSizeField,
                    inputSizeField,
                    layersField,
                    learningRateField,
                    epochsField,
                    labelColumnField,
                    learningRateDecayField,
                    optimizerChoiceBox
            );
        }

        return updateExistingConfig(
                batchSizeField,
                datasetPathField,
                inputSizeField,
                layersField,
                learningRateField,
                epochsField,
                labelColumnField,
                learningRateDecayField,
                optimizerChoiceBox
        );
    }

    private ModelConfig buildNewConfig(
            TextField datasetPathField,
            TextField batchSizeField,
            TextField inputSizeField,
            TextField layersField,
            TextField learningRateField,
            TextField epochsField,
            TextField labelColumnField,
            TextField learningRateDecayField,
            ComboBox<OptimizerType> optimizerChoiceBox
    ) {
        String datasetPath =
                requireNonEmpty(datasetPathField.getText(), "Dataset path");
        int inputSize =
                parseInt(requireNonEmpty(inputSizeField.getText(), "Input size"), "Input size");
        List<Integer> layerSizes =
                parseLayerList(requireNonEmpty(layersField.getText(), "Layers"));
        double learningRate =
                parseDouble(
                        requireNonEmpty(learningRateField.getText(), "Learning rate"),
                        "Learning rate"
                );
        int epochs =
                parseInt(requireNonEmpty(epochsField.getText(), "Epochs"), "Epochs");
        Integer batchSize = isBlank(batchSizeField.getText())
                ? null
                : parseInt(batchSizeField.getText(), "Batch size");

        String labelColumn =
                requireNonEmpty(labelColumnField.getText(), "Target/label column");
        dataset = new CsvDataset(Paths.get(datasetPath), labelColumn);
        updateLastDirectory(datasetPath);

        OptimizerType optimizerType = optimizerChoiceBox.getValue();
        if (optimizerType == null) {
            optimizerType = OptimizerType.SGD;
        }

        LearningRateScheduler scheduler =
                buildSchedulerIfPresent(learningRateDecayField, learningRate);

        MLP model = new MLP(inputSize, layerSizes);

        ModelConfig.Builder builder = new ModelConfig.Builder()
                .numFeatures(inputSize)
                .layerSizes(layerSizes)
                .learningRate(learningRate)
                .epochs(epochs)
                .datasetPath(datasetPath)
                .optimizerType(optimizerType)
                .model(model);

        if (batchSize != null) {
            builder.batchSize(batchSize);
        }
        if (scheduler != null) {
            builder.learningRateScheduler(scheduler);
        }

        modelConfig = builder.build();
        return modelConfig;
    }

    private ModelConfig updateExistingConfig(
            TextField batchSizeField,
            TextField datasetPathField,
            TextField inputSizeField,
            TextField layersField,
            TextField learningRateField,
            TextField epochsField,
            TextField labelColumnField,
            TextField learningRateDecayField,
            ComboBox<OptimizerType> optimizerChoiceBox
    ) {
        ModelConfig cfg = modelConfig;

        if (!isBlank(datasetPathField.getText())) {
            cfg.setDatasetPath(datasetPathField.getText().trim());
        }
        if (!isBlank(inputSizeField.getText())) {
            cfg.setNumFeatures(
                    parseInt(inputSizeField.getText().trim(), "Input size")
            );
        }
        if (!isBlank(layersField.getText())) {
            cfg.setLayerSizes(parseLayerList(layersField.getText().trim()));
        }
        if (!isBlank(learningRateField.getText())) {
            cfg.setLearningRate(
                    parseDouble(
                            learningRateField.getText().trim(),
                            "Learning rate"
                    )
            );
        }
        if (!isBlank(epochsField.getText())) {
            cfg.setEpochs(parseInt(epochsField.getText().trim(), "Epochs"));
        }
        if (!isBlank(batchSizeField.getText())) {
            cfg.setBatchSize(
                    parseInt(batchSizeField.getText().trim(), "Batch size")
            );
        }

        OptimizerType optimizerType = optimizerChoiceBox.getValue();
        if (optimizerType == null) {
            optimizerType = cfg.getOptimizerType();
            if (optimizerType == null) {
                optimizerType = OptimizerType.SGD;
            }
        }

        double learningRate = cfg.getLearningRate();
        LearningRateScheduler scheduler =
                buildSchedulerIfPresent(learningRateDecayField, learningRate);

        ModelConfig updatedCfg = new ModelConfig.Builder()
                .numFeatures(cfg.getNumFeatures())
                .layerSizes(cfg.getLayerSizes())
                .learningRate(learningRate)
                .epochs(cfg.getEpochs())
                .batchSize(cfg.getBatchSize())
                .datasetPath(cfg.getDatasetPath())
                .optimizerType(optimizerType)
                .model(new MLP(cfg.getNumFeatures(), cfg.getLayerSizes()))
                .learningRateScheduler(scheduler)
                .build();

        String datasetPath = requireNonEmpty(
                updatedCfg.getDatasetPath(),
                "Dataset path (from YAML/fields)"
        );
        String labelColumn =
                requireNonEmpty(labelColumnField.getText(), "Target/label column");
        dataset = new CsvDataset(Paths.get(datasetPath), labelColumn);
        updateLastDirectory(datasetPath);

        modelConfig = updatedCfg;
        return updatedCfg;
    }

    private void updateLastDirectory(String datasetPath) {
        try {
            lastDirectory = Paths.get(datasetPath)
                    .toAbsolutePath()
                    .getParent()
                    .toFile();
        } catch (Exception ignore) {
            // ignore
        }
    }

    private LearningRateScheduler buildSchedulerIfPresent(
            TextField learningRateDecayField,
            double learningRate
    ) {
        if (isBlank(learningRateDecayField.getText())) {
            return null;
        }
        double decay =
                parseDouble(learningRateDecayField.getText(), "LR decay");
        return new ExponentialDecayScheduler(learningRate, decay);
    }

    private void startTrainingThread(
            Button trainButton,
            DrawMetricGraph metricGraphs,
            TextArea confusionMatrixArea,
            ModelConfig runConfig,
            CsvDataset runDataset
    ) {
        Thread trainingThread = new Thread(() -> {
            try {
                Trainer trainer = new Trainer(runConfig, runDataset);
                trainer.addObserver(metricGraphs);
                trainer.addObserver(createTextAreaObserver(confusionMatrixArea));
                trainer.train();

                Platform.runLater(() ->
                        showInfoAlert("Training finished.")
                );
            } catch (Exception ex) {
                Platform.runLater(() ->
                        showErrorAlert("Training error:\n" + ex.getMessage())
                );
            } finally {
                Platform.runLater(() -> trainButton.setDisable(false));
            }
        }, "training-thread");

        trainingThread.start();
    }

    private TrainingObserver createTextAreaObserver(TextArea confusionMatrixArea) {
        return new TrainingObserver() {
            @Override
            public void onEpoch(
                    int epoch,
                    ConfusionMatrix confusionMatrix,
                    double avgLoss,
                    double accuracy
            ) {
                double precision = confusionMatrix.precision();
                double recall = confusionMatrix.recall();
                double f1 = confusionMatrix.f1();

                String text = String.format(
                        "Epoch %d%n" +
                                "TP=%d FP=%d FN=%d TN=%d  |  " +
                                "Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  Loss=%.6f%n%n",
                        epoch,
                        confusionMatrix.getTP(),
                        confusionMatrix.getFP(),
                        confusionMatrix.getFN(),
                        confusionMatrix.getTN(),
                        accuracy,
                        precision,
                        recall,
                        f1,
                        avgLoss
                );

                Platform.runLater(() -> confusionMatrixArea.appendText(text));
            }
        };
    }

    private void configurePredictButton(
            Button predictionButton,
            TextField predictionInputField,
            Label predictionOutputLabel
    ) {
        predictionButton.setOnAction(event -> {
            try {
                if (modelConfig == null) {
                    showErrorAlert("Train a model first!");
                    return;
                }

                String text = predictionInputField.getText().trim();
                if (text.isEmpty()) {
                    showErrorAlert("Please enter feature values.");
                    return;
                }

                String[] parts = text.split(",");
                if (parts.length != modelConfig.getNumFeatures()) {
                    showErrorAlert(
                            "Expected " + modelConfig.getNumFeatures()
                                    + " features but got " + parts.length
                    );
                    return;
                }

                List<Scalar> features = parseFeatureList(parts);

                MLP model = modelConfig.getOrCreateModel();
                List<Scalar> outputs = model.forward(features);

                double logit = outputs.get(0).data();
                double probability = 1.0 / (1.0 + Math.exp(-logit));
                int prediction = probability >= 0.5 ? 1 : 0;

                predictionOutputLabel.setText(
                        String.format(
                                "Prediction: %d  |  Probability: %.4f",
                                prediction,
                                probability
                        )
                );

            } catch (Exception ex) {
                showErrorAlert("Prediction error:\n" + ex.getMessage());
            }
        });
    }

    private List<Scalar> parseFeatureList(String[] parts) {
        List<Scalar> features = new ArrayList<>(parts.length);
        for (String part : parts) {
            features.add(new Scalar(Double.parseDouble(part.trim())));
        }
        return features;
    }

    private static String requireNonEmpty(String value, String fieldName) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException(fieldName + " is required.");
        }
        return value.trim();
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    private static int parseInt(String value, String fieldName) {
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(fieldName + " must be an integer.");
        }
    }

    private static double parseDouble(String value, String fieldName) {
        try {
            return Double.parseDouble(value.trim());
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(fieldName + " must be a number.");
        }
    }

    private static List<Integer> parseLayerList(String csv) {
        try {
            String[] parts = csv.split(",");
            List<Integer> list = new ArrayList<>(parts.length);
            for (String part : parts) {
                String trimmed = part.trim();
                if (!trimmed.isEmpty()) {
                    list.add(Integer.parseInt(trimmed));
                }
            }
            if (list.isEmpty()) {
                throw new IllegalArgumentException("Layers must not be empty.");
            }
            return list;
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(
                    "Layers must be a comma-separated list of integers (e.g., 16,16,1)."
            );
        }
    }

    private void showErrorAlert(String message) {
        new Alert(Alert.AlertType.ERROR, message).showAndWait();
    }

    private void showInfoAlert(String message) {
        new Alert(Alert.AlertType.INFORMATION, message).showAndWait();
    }

    public static void main(String[] args) {
        launch(args);
    }
}

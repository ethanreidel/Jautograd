package unit_tests;

import data.CsvDataset;
import data.DataLoader;
import data.Batch;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DatasetTests {

    @Test
    @DisplayName("create: valid CSV can be loaded into batches")
    void create_validFieldsRetained() throws Exception {
        Path tmp = Files.createTempFile("dataset_valid", ".csv");
        String csv = String.join("\n",
                "x1,x2,label",
                "0,0,0",
                "0,1,1",
                "1,0,1",
                "1,1,0"
        );
        Files.writeString(tmp, csv);

        CsvDataset ds = new CsvDataset(tmp, "label");

        DataLoader loader = new DataLoader(2, false);
        List<Batch> batches = loader.loadData(ds);

        assertNotNull(batches);
        assertFalse(batches.isEmpty(), "Expected at least one batch");
    }

    @Test
    @DisplayName("validate: missing label column throws")
    void validate_missingLabelColumnThrows() throws Exception {
        Path tmp = Files.createTempFile("dataset_missing_label", ".csv");
        String csv = String.join("\n",
                "x1,x2,y",
                "0,0,0",
                "0,1,1"
        );
        Files.writeString(tmp, csv);

        assertThrows(RuntimeException.class, () -> {
            new CsvDataset(tmp, "label");
        });
    }
}
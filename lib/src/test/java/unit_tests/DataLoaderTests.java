package unit_tests;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;
import data.CsvDataset;
import data.DataLoader;
import java.nio.file.Path;
import java.nio.file.Paths;


class DataLoaderTests {

    @Test @DisplayName("loadData: returns non-empty batches with batchSize respected")
    void loadDataBatchesOK() {
        var loader = new DataLoader(2, true);
        Path path = Paths.get("/home/ethan-reidel/Coding/PomonaWork/Jautograd/lib/src/main/resources/data/xor.csv");
        CsvDataset ds = new CsvDataset(path, "label");
        var batches = loader.loadData(ds);
        assertFalse(batches.isEmpty());
        assertTrue(batches.stream().allMatch(b -> b.features().size() <= 2));
    }

    @Test @DisplayName("loadData: invalid batch size throws")
    void loadDataBadBatchSize() {
        assertThrows(IllegalArgumentException.class, () -> new DataLoader(0, false));
        assertThrows(IllegalArgumentException.class, () -> new DataLoader(-2, true));
    }

    @Test @DisplayName("loadData: unready dataset throws")
    void loadDataUnready() {
        var loader = new DataLoader(4, true);
        Path path = Paths.get("/home/ethan-reidel/Coding/PomonaWork/Jautograd/lib/src/main/resources/data/xor.csv");
        assertThrows(java.io.UncheckedIOException.class, () -> new CsvDataset(Paths.get(""), "label"));
    }
}

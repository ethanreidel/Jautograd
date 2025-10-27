package unit_tests;
import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import org.junit.jupiter.api.*;

class DataLoaderTests {

    @Test @DisplayName("loadData: returns non-empty batches with batchSize respected")
    void loadDataBatchesOK() {
        var loader = new DataLoader(4, true);
        var ds = InMemoryDatasets.xor(); ds.validate(); ds.store();
        var batches = loader.loadData(ds);
        assertFalse(batches.isEmpty());
        assertTrue(batches.stream().allMatch(b -> b.features().size() <= 4 && b.labels().size() == b.features().size()));
    }

    @Test @DisplayName("loadData: invalid batch size throws")
    void loadDataBadBatchSize() {
        assertThrows(InvalidArgumentException.class, () -> new DataLoader(0, false));
        assertThrows(InvalidArgumentException.class, () -> new DataLoader(-2, true));
    }

    @Test @DisplayName("loadData: unready dataset throws")
    void loadDataUnready() {
        var loader = new DataLoader(4, true);
        var ds = InMemoryDatasets.xor();
        assertThrows(ValidationException.class, () -> loader.loadData(ds));
    }
}

import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.*;

class DatasetTests {

    @Test 
    @DisplayName("create: valid fields retained")
    void createValidOK() {
        var ds = Dataset.create("xor", "memory", "label", List.of(Path.of("train.csv")));
        assertEquals("xor", ds.getName());
        assertEquals("memory", ds.getType());
        assertEquals("label", ds.getLabelColumn());
        assertEquals(1, ds.getFiles().size());
    }

    @Test 
    @DisplayName("create: empty files throws")
    void createNoFiles() {
        assertThrows(InvalidArgumentException.class, () ->
            Dataset.create("xor", "memory", "label", List.of()));
    }

    @Test 
    @DisplayName("validate: missing label column throws")
    void validateMissingLabel() {
        var ds = Dataset.create("xor", "csv", "missing", List.of(Path.of("train.csv")));
        assertThrows(ValidationException.class, ds::validate);
    }

    @Test 
    @DisplayName("validate: non-existent file throws")
    void validateBadFile() {
        var ds = Dataset.create("xor", "csv", "label", List.of(Path.of("does_not_exist.csv")));
        assertThrows(ValidationException.class, ds::validate);
    }

    @Test 
    @DisplayName("store: marks dataset as ready and assigns id")
    void storePersists() {
        var ds = Dataset.create("xor", "memory", "label", List.of(Path.of("train.csv")));
        assertDoesNotThrow(ds::store);
        assertTrue(ds.isReady());
        assertNotNull(ds.getId());
    }
}

import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.*;

class DatasetTests {

    @Test 
    @DisplayName("create: valid fields retained")
    void create_Valid_ok() {
        var ds = Dataset.create("xor", "memory", "label", List.of(Path.of("train.csv")));
        assertEquals("xor", ds.getName());
        assertEquals("memory", ds.getType());
        assertEquals("label", ds.getLabelColumn());
        assertEquals(1, ds.getFiles().size());
    }

    @Test 
    @DisplayName("create: empty files throws")
    void create_NoFiles_throws() {
        assertThrows(InvalidArgumentException.class, () ->
            Dataset.create("xor", "memory", "label", List.of()));
    }

    @Test 
    @DisplayName("validate: missing label column throws")
    void validate_MissingLabel_throws() {
        var ds = Dataset.create("xor", "csv", "missing", List.of(Path.of("train.csv")));
        assertThrows(ValidationException.class, ds::validate);
    }

    @Test 
    @DisplayName("validate: non-existent file throws")
    void validate_BadFile_throws() {
        var ds = Dataset.create("xor", "csv", "label", List.of(Path.of("does_not_exist.csv")));
        assertThrows(ValidationException.class, ds::validate);
    }

    @Test 
    @DisplayName("store: marks dataset as ready and assigns id")
    void store_Persists_ok() {
        var ds = Dataset.create("xor", "memory", "label", List.of(Path.of("train.csv")));
        assertDoesNotThrow(ds::store);
        assertTrue(ds.isReady());
        assertNotNull(ds.getId());
    }
}

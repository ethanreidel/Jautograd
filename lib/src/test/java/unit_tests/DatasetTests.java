package unit_tests;
import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.*;
import data.*;
import java.io.UncheckedIOException;

class DatasetTests {

    @Test 
    @DisplayName("create: valid fields retained")
    void createValidOK() {
        Path csv = Paths.get("/home/ethan-reidel/Coding/PomonaWork/Jautograd/lib/src/main/resources/data/xor.csv");
        CsvDataset ds = new CsvDataset(csv, "label");
        assertEquals("label", ds.labelColumn());
        assertEquals(ds.header().indexOf("label"), ds.labelIndex());
        assertEquals(ds.header().size(), ds.header().size());
    }

    @Test 
    @DisplayName("create: empty files throws")
    void createNoFiles() {
        Path emptyCsv = Paths.get("/home/ethan-reidel/Coding/PomonaWork/Jautograd/lib/src/main/resources/data/empty.csv");
        assertThrows(IllegalArgumentException.class, () -> new CsvDataset(emptyCsv, "label"));
    }

    @Test 
    @DisplayName("validate: missing label column throws")
    void validateMissingLabel() {
        Path csv = Paths.get("/home/ethan-reidel/Coding/PomonaWork/Jautograd/lib/src/main/resources/data/xor.csv");
        assertThrows(IllegalArgumentException.class, () -> new CsvDataset(csv, "missing"));
    }

    @Test 
    @DisplayName("validate: non-existent file throws")
    void validateBadFile() {
        Path badCsv = Paths.get("/home/ethan-reidel/Coding/PomonaWork/Jautograd/lib/src/main/resources/data/does_not_exist.csv");
        assertThrows(UncheckedIOException.class, () -> new CsvDataset(badCsv, "label"));
    }

}

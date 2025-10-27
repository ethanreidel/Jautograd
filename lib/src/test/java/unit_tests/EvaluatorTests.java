package unit_tests;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;

class EvaluatorTests {

    @Test @DisplayName("run: returns metrics with 0<=acc<=1")
    void runValidSplit() {
        var evaluator = new Evaluator();
        var artifact = new ModelArtifact("mlp-xor", 1024);
        var result = evaluator.run(artifact, 0.25);
        assertNotNull(result);
        assertTrue(result.getAccuracy() >= 0.0 && result.getAccuracy() <= 1.0);
    }

    @Test @DisplayName("run: invalid split throws")
    void runInvalidSplit() {
        var evaluator = new Evaluator();
        var artifact = new ModelArtifact("mlp-xor", 1024);
        assertThrows(InvalidArgumentException.class, () -> evaluator.run(artifact, -0.1));
        assertThrows(InvalidArgumentException.class, () -> evaluator.run(artifact, 1.5));
    }

    @Test @DisplayName("run: empty artifact throws")
    void runEmptyArtifact() {
        var evaluator = new Evaluator();
        var artifact = new ModelArtifact("empty", 0);
        assertThrows(ValidationException.class, () -> evaluator.run(artifact, 0.2));
    }
}

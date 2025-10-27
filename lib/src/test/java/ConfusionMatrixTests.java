import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;

class ConfusionMatrixTests {

    @Test @DisplayName("accumulate: increments correct cell")
    void accumulateOK() {
        var cm = new ConfusionMatrix(2);
        cm.accumulate(1, 0);
        assertEquals(1, cm.get(1, 0));
        assertEquals(0, cm.get(0, 0));
    }

    @Test @DisplayName("accumulate: out-of-range labels throw")
    void accumulateOutofRange() {
        var cm = new ConfusionMatrix(2);
        assertThrows(InvalidArgumentException.class, () -> cm.accumulate(-1, 0));
        assertThrows(InvalidArgumentException.class, () -> cm.accumulate(0, 2));
    }
}

package unit_tests;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;

class MetricsTests {

    @Test @DisplayName("logEpoch: stores epoch, loss, accuracy")
    void logEpochStores() {
        var m = new Metrics();
        m.logEpoch(1, 0.9, 0.5);
        m.logEpoch(2, 0.7, 0.6);
        assertEquals(2, m.getHistory().size());
        assertEquals(2, m.getHistory().get(1).epoch());
        assertEquals(0.7, m.getHistory().get(1).loss(), 1e-9);
        assertEquals(0.6, m.getHistory().get(1).accuracy(), 1e-9);
    }

    @Test @DisplayName("logEpoch: invalid values throw")
    void logEpochInvalid() {
        var m = new Metrics();
        assertThrows(InvalidArgumentException.class, () -> m.logEpoch(0, 0.5, 0.5)); // epoch must start at 1?
        assertThrows(InvalidArgumentException.class, () -> m.logEpoch(1, Double.NaN, 0.5));
        assertThrows(InvalidArgumentException.class, () -> m.logEpoch(1, 0.5, -0.1));
        assertThrows(InvalidArgumentException.class, () -> m.logEpoch(1, 0.5, 1.1));
    }
}

package unit_tests;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;
import nn.ConfusionMatrix;

class ConfusionMatrixTests {

    @Test
    @DisplayName("accumulate: increments correct cell")
    void accumulateIncrementsCorrectCell() {
        ConfusionMatrix cm = new ConfusionMatrix();
        cm.update(1, 0); // actual=1, predicted=0 (false negative)
        int[][] arr = cm.toArray();
        assertEquals(1, arr[1][0]); // [actual=1][pred=0] is false negative
        assertEquals(0, arr[0][0]); // [actual=0][pred=0] is true negative
        assertEquals(0, arr[0][1]); // [actual=0][pred=1] is false positive
        assertEquals(0, arr[1][1]); // [actual=1][pred=1] is true positive
    }

    @Test
    @DisplayName("accumulate: out-of-range labels throw")
    void accumulateOutOfRangeThrows() {
        ConfusionMatrix cm = new ConfusionMatrix();
        assertThrows(IllegalArgumentException.class, () -> cm.update(-1, 0));
        assertThrows(IllegalArgumentException.class, () -> cm.update(0, 2));
        assertThrows(IllegalArgumentException.class, () -> cm.update(2, 1));
    }

    @Test
    @DisplayName("accumulate: batch update works")
    void accumulateBatchWorks() {
        ConfusionMatrix cm = new ConfusionMatrix();
        int[] actual = {0, 1, 1, 0};
        int[] predicted = {0, 1, 0, 1};
        cm.update(actual, predicted);
        int[][] arr = cm.toArray();
        assertEquals(1, arr[0][0]);
        assertEquals(1, arr[0][1]);
        assertEquals(1, arr[1][0]);
        assertEquals(1, arr[1][1]);
    }
}
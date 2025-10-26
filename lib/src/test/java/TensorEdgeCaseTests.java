

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

class TensorEdgeCaseTests {

    private static final double EPS = 1e-6;

    private static void assertTensorEquals(INDArray expected, INDArray actual) {
        assertTrue(expected.equalsWithEps(actual, EPS),
            () -> "Expected:\n" + expected + "\nActual:\n" + actual);
        assertArrayEquals(expected.shape(), actual.shape());
    }
    @Test
    @DisplayName("reshape throws on element count mismatch")
    void reshape_ElementCountMismatch_throws() {
        Tensor t = Tensor.zeros(2, 3);
        assertThrows(TensorShapeException.class, () -> t.reshape(4, 2));
    }

    @Test
    @DisplayName("zeros throws on negative dimension")
    void zeros_NegativeDim_throws() {
        assertThrows(TensorShapeException.class, () -> Tensor.zeros(3, -1));
    }

    @Test
    @DisplayName("of(null) throws type mismatch")
    void of_Null_throws() {
        assertThrows(TensorTypeMismatchException.class, () -> Tensor.of((double[][]) null));
    }

    @Test
    @DisplayName("add supports row-wise broadcasting")
    void add_RowBroadcast_ok() {
        Tensor a = Tensor.of(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Tensor b = Tensor.of(new double[]{10, 20, 30});
        Tensor c = a.add(b);

        INDArray expected = Nd4j.create(new double[][]{
            {11, 22, 33},
            {14, 25, 36}
        });
        assertTensorEquals(expected, c.data());
    }

    @Test
    @DisplayName("add throws on incompatible broadcast shapes")
    void add_IncompatibleBroadcast_throws() {
        Tensor a = Tensor.zeros(2, 3);
        Tensor b = Tensor.zeros(2); // cannot broadcast to (2,3)
        assertThrows(TensorBroadcastException.class, () -> a.add(b));
    }

    @Test
    @DisplayName("mul scalar multiply is supported")
    void mul_Scalar_ok() {
        Tensor a = Tensor.of(new double[][]{{1, 2}, {3, 4}});
        Tensor c = a.mul(Tensor.of(new double[]{2})); // (1,) scalar
        INDArray expected = Nd4j.create(new double[][]{{2, 4}, {6, 8}});
        assertTensorEquals(expected, c.data());
    }
    @Test
    @DisplayName("div throws on zero divisor when safe-div not requested")
    void div_ByZero_throws() {
        Tensor a = Tensor.of(new double[][]{{1, 2}});
        Tensor z = Tensor.zeros(1, 2);
        assertThrows(TensorDomainException.class, () -> a.div(z));
    }
    @ParameterizedTest(name = "relu({0}) == {1}")
    @CsvSource({
        "-1.0, 0.0",
        "0.0, 0.0",
        "0.5, 0.5",
        "10.0, 10.0"
    })
    @DisplayName("relu clamps negatives and preserves positives")
    void relu_Clamps(double x, double expected) {
        Tensor t = Tensor.of(new double[][]{{x}});
        Tensor y = t.relu();
        INDArray e = Nd4j.create(new double[][]{{expected}});
        assertTensorEquals(e, y.data());
    }

    // ---------- MatMul ----------

    @Test
    @DisplayName("matmul dimension mismatch throws")
    void matmul_DimMismatch_throws() {
        Tensor a = Tensor.zeros(2, 3);
        Tensor b = Tensor.zeros(3, 5);
        Tensor c = Tensor.zeros(4, 2); // wrong inner dim if used with a
        assertThrows(TensorShapeException.class, () -> a.matmul(c));
        // Sanity: valid multiply works
        assertDoesNotThrow(() -> a.matmul(b));
    }
    @Test
    @DisplayName("sum(axis) throws on invalid axis")
    void sum_InvalidAxis_throws() {
        Tensor x = Tensor.zeros(2, 3);
        assertThrows(TensorAxisOutOfBoundsException.class, () -> x.sum(2, false));
        assertThrows(TensorAxisOutOfBoundsException.class, () -> x.sum(-3, true));
    }
    @Test
    @DisplayName("mean keeps dims when requested")
    void mean_KeepDims_ok() {
        Tensor x = Tensor.of(new double[][]{
            {1, 3},
            {5, 7}
        });
        Tensor m = x.mean(1, true);
        INDArray expected = Nd4j.create(new double[][]{{2.0}, {6.0}});
        assertTensorEquals(expected, m.data());
        assertArrayEquals(new int[]{2,1}, m.shape());
    }
    @Test
    @DisplayName("operations throw on non-finite inputs when strict mode is on")
    void ops_NonFinite_throws() {
        Tensor a = Tensor.of(new double[][]{{1, Double.POSITIVE_INFINITY}});
        Tensor b = Tensor.of(new double[][]{{2, 3}});
        assertThrows(TensorDomainException.class, () -> a.add(b));
    }
}
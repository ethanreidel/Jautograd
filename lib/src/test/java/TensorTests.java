

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class TensorTests {

    private static final double EPS = 1e-6;
    private static void assertTensorEquals(INDArray expected, INDArray actual) {
        assertTrue(expected.equalsWithEps(actual, EPS),
            () -> "Expected:\n" + expected + "\nActual:\n" + actual);
        assertArrayEquals(expected.shape(), actual.shape());
    }

    //creation ops
    @Test 
    @DisplayName("zeros creates correct shape")
    void testZeros() {
        Tensor t = Tensor.zeros(3, 3);
        INDArray expected = Nd4j.zeros(3, 3);
        assertTensorEquals(t.data(), expected);
    }
    @Test void testRandn() {
        Tensor t = Tensor.randn(3, 3);
        assertArrayEquals(new int[]{3, 4}, t.shape());
    }

    //shape ops
    @Test void testReshape() {
        Tensor t = Tensor.zeros(6).reshape(2, 3);
        assertArrayEquals(new int[]{2, 3}, t.shape());
    }
    @Test void testTranspose() {
        Tensor t = Tensor.of(new double[][] {
            {1, 2},
            {3, 4}
        });
        Tensor tr = t.transpose();

        INDArray expected = Nd4j.create(new double[][] {
            {1, 3},
            {2, 4}
        });
        assertTensorEquals(tr.data, expected);
    }

    //elementwise ops
    @Test void testAdd() {
        Tensor a = Tensor.of(new double[][]{{1, 2}, {3, 4}});
        Tensor b = Tensor.of(new double[][]{{10, 20}, {30, 40}});
        Tensor c = a.add(b);

        INDArray expected = Nd4j.create(new double[][]{{11, 22}, {33, 44}});
        assertTensorEquals(c.data(), expected);
    }
    @Test void testMul() {
        Tensor a = Tensor.of(new double[][]{{1, 2}, {3, 4}});
        Tensor b = Tensor.of(new double[][]{{-1, 0.5}, {2, -3}});
        Tensor c = a.mul(b);

        INDArray expected = Nd4j.create(new double[][]{{-1, 1}, {6, -12}});
        assertTensorEquals(c.data(), expected);
    }
    @Test void testSub() {
        Tensor a = Tensor.of(new double[][]{{1, 2}, {3, 4}});
        Tensor b = Tensor.of(new double[][]{{-1, 0.5}, {2, -3}});
        Tensor c = a.sub(b);

        INDArray expected = Nd4j.create(new double[][]{{2, 1.5}, {1, 7}});
        assertTensorEquals(c.data(), expected);
    }
    @Test void testDiv() {
        Tensor a = Tensor.of(new double[][]{{1, 2}, {3, 4}});
        Tensor b = Tensor.of(new double[][]{{-1, 0.5}, {2, -3}});
        Tensor c = a.div(b); //a/b = A * B^-1

        INDArray expected = Nd4j.create(new double[][]{{-3.5, -1.25}, {-8.5, -2.75}});
        assertTensorEquals(c.data(), expected);
    }
    @Test
    @DisplayName("relu check")
    void testRelu() {
        Tensor a = Tensor.of(new double[][]{
            {1, 2},
            {-1, -2}
        });
        a.relu();
        INDArray expected = Nd4j.create(new double[][]{
            {1, 2},
            {0, 0}
        });
        assertTensorEquals(a.data, expected);
    }


    @Test
    @DisplayName("matrix multiplication works")
    void testMatmul() {
        Tensor a = Tensor.of(new double[][]{
            {1, 2, 3},   // 2x3
            {4, 5, 6}
        });
        Tensor b = Tensor.of(new double[][]{
            {7, 8},      // 3x2
            {9, 10},
            {11, 12}
        });
        Tensor c = a.matmul(b);

        INDArray expected = Nd4j.create(new double[][]{
            { 58,  64},
            {139, 154}
        });
        assertTensorEquals(c.data(), expected);
    }

    @Test
    @DisplayName("sum over axis with/without keepDims")
    void testSumAxis() {
        Tensor x = Tensor.of(new double[][]{
            {1, 2, 3},
            {4, 5, 6}
        });

        Tensor sumAxis0 = x.sum(0, false);
        INDArray expected0 = Nd4j.create(new double[]{5, 7, 9});
        assertINDNearlyEquals(expected0, sumAxis0.data());
        assertArrayEquals(new int[]{3}, sumAxis0.shape());

        Tensor sumAxis1Keep = x.sum(1, true); // -> (2,1)
        INDArray expected1k = Nd4j.create(new double[][]{{6}, {15}});
        assertINDNearlyEquals(expected1k, sumAxis1Keep.data());
        assertArrayEquals(new int[]{2,1}, sumAxis1Keep.shape());
    }

    @Test
    @DisplayName("mean over all elements and axis")
    void testMean() {
        Tensor x = Tensor.of(new double[][]{{1, 2}, {3, 7}});
        Tensor mAll = x.mean();
        INDArray expAll = Nd4j.scalar((1 + 2 + 3 + 7) / 4.0);
        assertTensorEquals(mAll.data(), expAll);

        Tensor mAxis0 = x.mean(0, false);
        INDArray exp0 = Nd4j.create(new double[]{(1 + 3) / 2.0, (2 + 7) / 2.0});
        assertTensorEquals(mAxis0, new int[]{2});
    }

}

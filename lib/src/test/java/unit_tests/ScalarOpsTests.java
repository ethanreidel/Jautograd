package unit_tests;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class ScalarOpsTests {

    private static final double EPS = 1e-9;

    private static void assertClose(double expected, double actual) {
        assertEquals(expected, actual, EPS, "expected=" + expected + " actual=" + actual);
    }

    @Test @DisplayName("add: forward and backward")
    void addForwardBackward() {
        Scalar a = Scalar.of(2.0);
        Scalar b = Scalar.of(3.0);
        Scalar c = a.add(b);              // c = 5
        c.backward();                     // dc/da = 1, dc/db = 1

        assertClose(5.0, c.data());
        assertClose(1.0, a.grad());
        assertClose(1.0, b.grad());
    }

    @Test @DisplayName("sub: forward and backward")
    void subForwardBackward() {
        Scalar a = Scalar.of(5.0);
        Scalar b = Scalar.of(2.0);
        Scalar c = a.sub(b);              // c = 3
        c.backward();                     // dc/da = 1, dc/db = -1

        assertClose(3.0, c.data());
        assertClose(1.0, a.grad());
        assertClose(-1.0, b.grad());
    }

    @Test @DisplayName("mul: forward and backward")
    void mulForwardBackward() {
        Scalar a = Scalar.of(2.0);
        Scalar b = Scalar.of(3.0);
        Scalar c = a.mul(b);              // c = 6
        c.backward();                     // dc/da = b, dc/db = a

        assertClose(6.0, c.data());
        assertClose(3.0, a.grad());
        assertClose(2.0, b.grad());
    }

    @Test @DisplayName("div: forward and backward")
    void divForwardBackward() {
        Scalar a = Scalar.of(6.0);
        Scalar b = Scalar.of(3.0);
        Scalar c = a.div(b);              // c = 2
        c.backward();                     // dc/da=1/b=1/3, dc/db=-a/b^2=-6/9=-2/3

        assertClose(2.0, c.data());
        assertClose(1.0/3.0, a.grad());
        assertClose(-2.0/3.0, b.grad());
    }

    @Test @DisplayName("pow(const): forward and backward dy/da = p * a^(p-1)")
    void powConstExpForwardBackward() {
        Scalar a = Scalar.of(2.0);
        Scalar y = a.pow(3.0);            // 8
        y.backward();
        assertClose(8.0, y.data());
        assertClose(3.0 * Math.pow(2.0, 2.0), a.grad()); // 12
    }

    @Test @DisplayName("pow(var): forward and backward dy/da = b a^(b-1), dy/db = a^b ln(a)")
    void powVarExpForwardBackward() {
        Scalar a = Scalar.of(2.0);
        Scalar b = Scalar.of(3.0);
        Scalar y = a.pow(b);              // 8
        y.backward();
        assertClose(8.0, y.data());
        assertClose(3.0 * Math.pow(2.0, 2.0), a.grad()); // 12
        assertClose(Math.pow(2.0, 3.0) * Math.log(2.0), b.grad()); // 8 ln 2
    }

    @Test @DisplayName("exp: forward and backward dy/da = exp(a)")
    void expForwardBackward() {
        Scalar a = Scalar.of(1.2);
        Scalar y = a.exp();
        y.backward();
        assertClose(Math.exp(1.2), y.data());
        assertClose(Math.exp(1.2), a.grad());
    }

    @Test @DisplayName("log: forward and backward dy/da = 1/a")
    void logForwardBackward() {
        Scalar a = Scalar.of(2.5);
        Scalar y = a.log();
        y.backward();
        assertClose(Math.log(2.5), y.data());
        assertClose(1.0/2.5, a.grad());
    }

    @Test @DisplayName("tanh: forward in (-1,1) and backward dy/da = 1 - tanh(a)^2")
    void tanhForwardBackward() {
        Scalar a = Scalar.of(0.7);
        Scalar y = a.tanh();
        y.backward();

        double t = Math.tanh(0.7);
        assertClose(t, y.data());
        assertClose(1.0 - t*t, a.grad());
    }

    @Test @DisplayName("relu: forward clamp and backward {0 if a<=0, 1 if a>0}")
    void reluForwardBackward_Positive() {
        Scalar a = Scalar.of(1.5);
        Scalar y = a.relu();
        y.backward();
        assertClose(1.5, y.data());
        assertClose(1.0, a.grad());
    }

    @Test @DisplayName("relu at zero: gradient defined as 0")
    void reluForwardBackward_AtZero() {
        Scalar a = Scalar.of(0.0);
        Scalar y = a.relu();
        y.backward();
        assertClose(0.0, y.data());
        assertClose(0.0, a.grad());
    }

    @Test @DisplayName("relu negative: gradient 0")
    void reluForwardBackward_Negative() {
        Scalar a = Scalar.of(-2.0);
        Scalar y = a.relu();
        y.backward();
        assertClose(0.0, y.data());
        assertClose(0.0, a.grad());
    }

    @Test @DisplayName("gradients accumulate over multiple paths")
    void gradientAccumulation() {
        Scalar a = Scalar.of(2.0);
        Scalar b = Scalar.of(3.0);
        Scalar x = a.mul(b);              // a*b
        Scalar y = a.mul(b);              // a*b
        Scalar z = x.add(y);              // 2ab
        z.backward();                     // dz/da = 2b = 6, dz/db = 2a = 4

        assertClose(2.0 * 2.0 * 3.0, z.data()); // 12
        assertClose(6.0, a.grad());
        assertClose(4.0, b.grad());
    }

    @Test @DisplayName("zeroGrad clears old gradients before reuse")
    void zeroGrad_Clears() {
        Scalar a = Scalar.of(2.0);
        Scalar b = Scalar.of(5.0);

        // First loss: (a*b)
        Scalar l1 = a.mul(b);
        l1.backward();
        assertClose(b.data(), a.grad());      // dl1/da = b = 5
        assertClose(a.data(), b.grad());      // dl1/db = a = 2

        a.zeroGrad();
        b.zeroGrad();

        Scalar l2 = a.add(b);                 // a + b
        l2.backward();
        assertClose(1.0, a.grad());
        assertClose(1.0, b.grad());
    }
}

package scalar;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A scalar value that supports automatic differentiation via a computation graph.
 *
 * <p>Each {@code Scalar} node stores:
 * <ul>
 *   <li>its numeric value ({@link #data})</li>
 *   <li>its gradient with respect to some loss ({@link #grad})</li>
 *   <li>its children in the computation graph</li>
 *   <li>an operation label (for debugging)</li>
 *   <li>a {@code _backward} function to propagate gradients to its children</li>
 * </ul>
 */
public final class Scalar {
    public double data;

    private double grad;
    private final List<Scalar> children;
    private String op;
    private Runnable _backward;

    public Scalar(double data) {
        this.data = data;
        this.grad = 0.0;
        this.children = new ArrayList<>();
        this.op = "leaf";
        this._backward = () -> {};
    }

    public double data() {
        return data;
    }

    public double grad() {
        return grad;
    }

    public Scalar add(Scalar other) {
        Scalar result = new Scalar(this.data + other.data);
        result.op = "+";
        result.children.add(this);
        result.children.add(other);

        result._backward = () -> {
            this.grad += result.grad;
            other.grad += result.grad;
        };

        return result;
    }

    public Scalar sub(Scalar other) {
        Scalar result = new Scalar(this.data - other.data);
        result.op = "-";
        result.children.add(this);
        result.children.add(other);

        result._backward = () -> {
            this.grad += result.grad;
            other.grad -= result.grad;
        };

        return result;
    }

    public Scalar mul(Scalar other) {
        Scalar result = new Scalar(this.data * other.data);
        result.op = "*";
        result.children.add(this);
        result.children.add(other);

        result._backward = () -> {
            this.grad += other.data * result.grad;
            other.grad += this.data * result.grad;
        };

        return result;
    }

    public Scalar div(Scalar other) {
        Scalar result = new Scalar(this.data / other.data);
        result.op = "/";
        result.children.add(this);
        result.children.add(other);

        result._backward = () -> {
            this.grad += (1.0 / other.data) * result.grad;
            other.grad += (-this.data / (other.data * other.data)) * result.grad;
        };

        return result;
    }

    public Scalar pow(double exponent) {
        Scalar result = new Scalar(Math.pow(this.data, exponent));
        result.op = "**";
        result.children.add(this);

        result._backward = () -> {
            this.grad += exponent * Math.pow(this.data, exponent - 1.0) * result.grad;
        };

        return result;
    }

    public Scalar exp() {
        Scalar result = new Scalar(Math.exp(this.data));
        result.op = "exp";
        result.children.add(this);

        result._backward = () -> {
            this.grad += result.data * result.grad;
        };

        return result;
    }

    public Scalar log() {
        Scalar result = new Scalar(Math.log(this.data));
        result.op = "log";
        result.children.add(this);

        result._backward = () -> {
            this.grad += (1.0 / this.data) * result.grad;
        };

        return result;
    }

    public Scalar tanh() {
        Scalar result = new Scalar(Math.tanh(this.data));
        result.op = "tanh";
        result.children.add(this);

        result._backward = () -> {
            double exp2x = Math.exp(2.0 * this.data);
            double numerator = 4.0 * exp2x;
            double denominator = Math.pow(exp2x + 1.0, 2.0);
            double derivative = numerator / denominator;
            this.grad += derivative * result.grad;
        };

        return result;
    }

    public Scalar relu() {
        Scalar result = new Scalar(Math.max(0.0, this.data));
        result.op = "relu";
        result.children.add(this);

        result._backward = () -> {
            double localGrad = (this.data > 0.0) ? 1.0 : 0.0;
            this.grad += localGrad * result.grad;
        };

        return result;
    }

    public Scalar abs() {
        Scalar result = new Scalar(Math.abs(this.data));
        result.op = "abs";
        result.children.add(this);

        result._backward = () -> {
            double sign;
            if (this.data > 0.0) {
                sign = 1.0;
            } else if (this.data < 0.0) {
                sign = -1.0;
            } else {
                sign = 0.0;
            }
            this.grad += result.grad * sign;
        };

        return result;
    }

    public static Scalar max(Scalar a, Scalar b) {
        Scalar result = new Scalar(Math.max(a.data, b.data));
        result.op = "max";
        result.children.add(a);
        result.children.add(b);

        result._backward = () -> {
            double gradA = (a.data >= b.data) ? 1.0 : 0.0;
            double gradB = (b.data > a.data) ? 1.0 : 0.0;
            a.grad += result.grad * gradA;
            b.grad += result.grad * gradB;
        };

        return result;
    }

    public Scalar neg() {
        Scalar result = new Scalar(-this.data);
        result.op = "neg";
        result.children.add(this);

        result._backward = () -> {
            this.grad += -1.0 * result.grad;
        };

        return result;
    }

    // -------------------------------------------------------------------------
    // Autograd engine: building topological order and backpropagation
    // -------------------------------------------------------------------------

    /**
     * Builds a topological ordering of the computation graph rooted at {@code node}.
     *
     */
    public void buildTopo(List<Scalar> topo, Set<Scalar> visited, Scalar node) {
        if (!visited.contains(node)) {
            visited.add(node);
            for (Scalar child : node.children) {
                buildTopo(topo, visited, child);
            }
            topo.add(node);
        }
    }

    /**
     * Runs backpropagation starting from this node as the "loss" node.
     *
     * <p>It:
     * <ol>
     *   <li>builds a topological ordering of the graph</li>
     *   <li>sets this.grad = 1 (dL/dL)</li>
     *   <li>traverses the graph in reverse topological order, calling each node's
     *       {@code _backward} function</li>
     * </ol>
     */
    public void backward() {
        List<Scalar> topo = new ArrayList<>();
        Set<Scalar> visited = new HashSet<>();

        buildTopo(topo, visited, this);

        // dL/dL = 1
        this.grad = 1.0;

        for (int i = topo.size() - 1; i >= 0; i--) {
            topo.get(i)._backward.run();
        }
    }

    /**
     * Resets this node's gradient to zero.
     *
     * <p>Note: in this project we usually call {@code zeroGrad()} on parameters
     * (weights/biases). The computation graph nodes are recreated for each
     * forward pass, so we don't need to recursively clear the whole graph.
     */
    public void zeroGrad() {
        this.grad = 0.0;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Scalar {\n");
        builder.append("  data: ").append(data).append(",\n");
        builder.append("  grad: ").append(grad).append(",\n");
        builder.append("  op: ").append(op == null ? "null" : op).append(",\n");
        builder.append("  children: [");
        for (int i = 0; i < children.size(); i++) {
            builder.append(children.get(i).data());
            if (i < children.size() - 1) {
                builder.append(", ");
            }
        }
        builder.append("]\n}");
        return builder.toString();
    }
}

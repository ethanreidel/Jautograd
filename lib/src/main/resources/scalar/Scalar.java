package scalar;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;

public class Scalar {
    
    private double data;
    private double grad;
    private List<Scalar> children;
    private String op;
    private Runnable _backward;

    public Scalar(double data) {
        this.data = data;
        this.grad = 0;
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


    //operations
    public Scalar add(Scalar other) {
        Scalar res = new Scalar(this.data + other.data);
        res.op = "+";
        res.children.add(this);
        res.children.add(other);
        res._backward = () -> {
            this.grad += res.grad;
            other.grad += res.grad;
        };
        return res;
    }
    public Scalar mul(Scalar other) {
        Scalar res = new Scalar(this.data * other.data);
        res.op = "*";
        res.children.add(this);
        res.children.add(other);
        res._backward = () -> {
            this.grad += other.data * res.grad;
            other.grad += this.data * res.grad;
        };
        return res;
    }
    public Scalar sub(Scalar other) {
        Scalar res = new Scalar(this.data - other.data);
        res.op = "-";
        res.children.add(this);
        res.children.add(other);
        res._backward = () -> { 
            this.grad += res.grad;
            other.grad -= res.grad;
        };
        return res;
    }
    public Scalar div(Scalar other) {
        Scalar res = new Scalar(this.data / other.data);
        res.op = "/";
        res.children.add(this);
        res.children.add(other);
        res._backward = () -> {
            this.grad += (1/other.data) * res.grad;
            other.grad += (-this.data / (other.data * other.data)) * res.grad;
        };
        return res;
    }
  
    public void buildTopo(List<Scalar> topo, Set<Scalar> visited, Scalar node) {
        if (!visited.contains(node)) {
            visited.add(node);
            for (Scalar child : node.children) {
                buildTopo(topo, visited, child);
            }
            topo.add(node);
        }
    }

    public void backward() {
        List<Scalar> topo = new ArrayList<>();
        Set<Scalar> visited = new HashSet<>();

        buildTopo(topo, visited, this);

        this.grad = 1; //dL/dL = 1
        for (int i = topo.size()-1; i >= 0; i--) {
            topo.get(i)._backward.run();
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Scalar {\n");
        sb.append("  data: ").append(data).append(",\n");
        sb.append("  grad: ").append(grad).append(",\n");
        sb.append("  op: ").append(op == null ? "null" : op).append(",\n");
        sb.append("  children: [");
        for (int i = 0; i < children.size(); i++) {
            sb.append(children.get(i).data());
            if (i < children.size() - 1) sb.append(", ");
        }
        sb.append("]\n}");
        return sb.toString();
    }

}

package nn;


import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import scalar.Scalar;

public class Neuron extends Module {
    private final List<Scalar> w;
    private final Scalar b;
    private final boolean nonlin;
    private static final Random rand = new Random();

    public Neuron(int inputSize, boolean nonlinearity) {
        this.w = new ArrayList<>();
        for (int i = 0; i < inputSize; i++) {
            this.w.add(new Scalar(rand.nextDouble() * 2 - 1));
        }
        this.b = new Scalar(0.0);
        this.nonlin = nonlinearity;
    }

    public Scalar forward(List<Scalar> x) {
        Scalar act = b;
        for (int i = 0; i < w.size(); i++) {
            act = act.add(w.get(i).mul(x.get(i)));
        }
        return nonlin ? act.relu() : act;
    }
    public List<Scalar> parameters() {
        List<Scalar> params = new ArrayList<>(w);
        params.add(b);
        return params;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(nonlin ? "ReLU" : "Linear");
        sb.append("Neuron(").append(w.size());
        sb.append(")");
        
        //.append(") weights=[");
        // for (int i = 0; i < w.size(); i++) {
        //     sb.append(String.format("%.3f", w.get(i).data()));
        //     if (i < w.size() - 1) sb.append(", ");
        // }
        sb.append("]");
        return sb.toString();
    }
}

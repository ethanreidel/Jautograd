package nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

import nn.MLP;
import scalar.Scalar;

public class test {
    public static void main(String[] args) {

        List<Integer> layers = new ArrayList<>(java.util.Arrays.asList(2, 2, 1));

        MLP model = new MLP(2, layers);
        System.out.println(model);

        Scalar x = new Scalar(10.0);
        Scalar y = new Scalar(5.0);
        //Scalar z = new Scalar(5.0);


        List<Scalar> input = new ArrayList<>(Arrays.asList(x, y));
        var output = model.forward(input);
        System.out.println(output);
    }
}

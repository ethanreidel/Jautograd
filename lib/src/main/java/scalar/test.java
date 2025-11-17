package scalar;

import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;

// import javafx.application.Application;
// import javafx.scene.Scene;
// import javafx.stage.Stage;
// import com.brunomnsilva.smartgraph.graph.Graph;
// import com.brunomnsilva.smartgraph.graph.DigraphEdgeList;
// import com.brunomnsilva.smartgraph.graphview.SmartGraphPanel;
// import com.brunomnsilva.smartgraph.graphview.SmartPlacementStrategy;


public class test {

    public static void main(String[] args) {
        Scalar a = new Scalar(1.2);
        Scalar y = a.exp();
        y.backward();
        
        // Scalar a = new Scalar(3.0);
        // Scalar b = new Scalar(4.0);
    
        // Scalar c = a.add(b); //c=7.0

        // Scalar d = new Scalar(5.0);
        // Scalar e = c.sub(d); //e=35
        // //Scalar f = e.tanh();
        // e.backward();

        List<Scalar> order = new ArrayList<>();
        y.buildTopo(order, new HashSet<>(), y);

        // topological order
        for (Scalar s : order) {
            System.out.println(s);
        }



    }
    

}

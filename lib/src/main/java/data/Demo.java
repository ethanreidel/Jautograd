package data;

// import data.CsvDataset;
// import data.Example;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import nn.*;
import scalar.*;

public class Demo {
    public static void main(String[] args) {
        Path csv = Paths.get("/home/ethan-reidel/Coding/PomonaWork/Jautograd/lib/src/main/resources/data/xor.csv");
        CsvDataset ds = new CsvDataset(csv, "label");

        DataLoader dl = new DataLoader(2, false);

        List<Batch> data = dl.loadData(ds);
        
 
        List<Integer> layers = new ArrayList<>(java.util.Arrays.asList(2, 2, 1));

        // x \--> x ---> output
        // x /--> x /


        //List<List<Scalar>> output = new ArrayList<>();

        
        MLP model = new MLP(2, Arrays.asList(16, 16, 1));
        Optimizer optim = new Optimizer(.001);


        

        Trainer trainer = new Trainer(optim, model, 100, dl, ds);
        trainer.train();

        


        









        // for (Example ex : ds) {
        //     double[] x = ex.features();
        //     int y = ex.label();
        //     System.out.printf("x=%s  y=%d%n", Arrays.toString(x), y);
        // }
    }
}

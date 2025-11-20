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
        
 
        List<Integer> layers = new ArrayList<>(java.util.Arrays.asList(2, 1));

        MLP model = new MLP(2, layers); 
        System.out.println(model);
        for (Batch b : data) {
            for (int i = 0; i < b.size(); i++) {
                
            }
        //     ArrayList<double[]> features = b.getBatch();

            model.forward(b.getBatch());

        //     System.out.println(b);
            
        }












        // for (Example ex : ds) {
        //     double[] x = ex.features();
        //     int y = ex.label();
        //     System.out.printf("x=%s  y=%d%n", Arrays.toString(x), y);
        // }
    }
}

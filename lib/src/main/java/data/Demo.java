package data;

// import data.CsvDataset;
// import data.Example;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Demo {
    public static void main(String[] args) {
        Path csv = Paths.get("/home/ethan-reidel/Coding/PomonaWork/Jautograd/lib/src/main/resources/data/xor.csv");
        CsvDataset ds = new CsvDataset(csv, "label");

        DataLoader dl = new DataLoader(2, false);

        List<Batch> data = dl.loadData(ds);


        for (Batch b : data) {
            System.out.println(b);
        }

        // for (Example ex : ds) {
        //     double[] x = ex.features();
        //     int y = ex.label();
        //     System.out.printf("x=%s  y=%d%n", Arrays.toString(x), y);
        // }
    }
}

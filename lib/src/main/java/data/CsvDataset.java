package data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public final class CsvDataset implements Iterable<Example> {
    private final Path file;
    private final String labelColumn;
    private final int labelIndex;
    private final int totalColumns;
    private final List<String> header;
    
    public CsvDataset(Path file, String labelColumn) {
        if (file == null) {
            throw new IllegalArgumentException("file required.");
        }
        if (labelColumn == null || labelColumn.isBlank()) {
            throw new IllegalArgumentException("labelColumn required");
        }
        
        this.file = file;
        this.labelColumn = labelColumn;

        //get label index and total columns
        try (BufferedReader reader = Files.newBufferedReader(file)) {
            String line = reader.readLine();
            if (line == null) {
                throw new IllegalArgumentException("empty CSV");
            }
            this.header = List.of(line.split(",", -1));
            int idx = this.header.indexOf(labelColumn);
            if (idx < 0) throw new IllegalArgumentException(
                "missing label column '" + labelColumn + "' in header: " + this.header);
            this.labelIndex = idx;
            this.totalColumns = this.header.size();
        } catch (IOException e) {
            throw new UncheckedIOException("failed reading header", e);
        }
    }

    public List<String> header() { return header; }
    public int labelIndex() { return labelIndex; }
    public Path path() { return file; }
    public String labelColumn() { return labelColumn; }


    @Override
    public Iterator<Example> iterator() {
        return new Iterator<>() {
            final BufferedReader br = open();
            String next = advance();

            private BufferedReader open() {
                try {
                    BufferedReader r = Files.newBufferedReader(file);
                    r.readLine();
                    return r;
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }

            private String advance() {
                try {
                    String line;
                    while ((line = (br != null ? br.readLine() : null)) != null) {
                        if (line.trim().isEmpty()) continue;
                        if (line.startsWith("#")) continue;
                        return line;
                    }
                    if (br != null) br.close();
                    return null;
                } catch (IOException e) {
                    try { if (br != null) br.close(); } catch (IOException ignored) {}
                    throw new UncheckedIOException(e);
                }
            }

            @Override public boolean hasNext() { return next != null; }

            @Override public Example next() {
                if (next == null) throw new NoSuchElementException();
                String line = next;
                next = advance();

                String[] parts = line.split(",", -1);
                
                if (parts.length != totalColumns) {
                    throw new IllegalArgumentException("column count mismatch at row: " + line);
                }

                int label;
                try {
                    label = Integer.parseInt(parts[labelIndex].trim());
                } catch (Exception e) {
                    throw new IllegalArgumentException("bad label value: " + parts[labelIndex]);
                }

                double[] feats = new double[totalColumns - 1];
                int k = 0;
                for (int i = 0; i < totalColumns; i++) {
                    if (i == labelIndex) continue;
                    String s = parts[i].trim();
                    try {
                        feats[k++] = s.isEmpty() ? Double.NaN : Double.parseDouble(s);
                    } catch (NumberFormatException nfe) {
                        throw new IllegalArgumentException("bad feature value '" + s + "' at column " + i);
                    }
                }
                return new Example(feats, label);
            }
        };
    }
}

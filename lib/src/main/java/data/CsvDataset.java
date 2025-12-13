package data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * CSV-backed dataset that yields {@link Example} instances.
 *
 * <p>Skips empty lines and lines starting with '#'.
 */
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

        try (BufferedReader reader = Files.newBufferedReader(file)) {
            String firstLine = reader.readLine();
            if (firstLine == null) {
                throw new IllegalArgumentException("empty CSV");
            }

            this.header = List.of(firstLine.split(",", -1));
            int index = header.indexOf(labelColumn);

            if (index < 0) {
                throw new IllegalArgumentException(
                        "missing label column '" + labelColumn + "' in header: " + header
                );
            }

            this.labelIndex = index;
            this.totalColumns = header.size();
        } catch (IOException e) {
            throw new UncheckedIOException("failed reading header", e);
        }
    }

    public List<String> header() {
        return header;
    }

    public int labelIndex() {
        return labelIndex;
    }

    public Path path() {
        return file;
    }

    public String labelColumn() {
        return labelColumn;
    }

    @Override
    public Iterator<Example> iterator() {
        return new CsvIterator();
    }

    /**
     * Iterator over CSV rows that turns them into {@link Example} objects.
     */
    private final class CsvIterator implements Iterator<Example> {

        private final BufferedReader reader;
        private String nextLine;

        CsvIterator() {
            this.reader = openReaderSkippingHeader();
            this.nextLine = advance();
        }

        private BufferedReader openReaderSkippingHeader() {
            try {
                BufferedReader bufferedReader = Files.newBufferedReader(file);
                bufferedReader.readLine();
                return bufferedReader;
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        private String advance() {
            try {
                String line;
                while ((line = (reader != null ? reader.readLine() : null)) != null) {
                    String trimmed = line.trim();
                    if (trimmed.isEmpty()) {
                        continue;
                    }
                    if (trimmed.startsWith("#")) {
                        continue;
                    }
                    return line;
                }
                closeReaderQuietly();
                return null;
            } catch (IOException e) {
                closeReaderQuietly();
                throw new UncheckedIOException(e);
            }
        }

        private void closeReaderQuietly() {
            try {
                if (reader != null) {
                    reader.close();
                }
            } catch (IOException ignored) {
                // ignore
            }
        }

        @Override
        public boolean hasNext() {
            return nextLine != null;
        }

        @Override
        public Example next() {
            if (nextLine == null) {
                throw new NoSuchElementException();
            }

            String line = nextLine;
            nextLine = advance();

            String[] parts = line.split(",", -1);

            if (parts.length != totalColumns) {
                throw new IllegalArgumentException(
                        "column count mismatch at row: " + line
                                + " (expected " + totalColumns + " columns, got " + parts.length + ")"
                );
            }

            int label = parseLabel(parts[labelIndex]);
            double[] features = parseFeatures(parts);

            return new Example(features, label);
        }

        private int parseLabel(String labelString) {
            try {
                return Integer.parseInt(labelString.trim());
            } catch (Exception e) {
                throw new IllegalArgumentException("bad label value: " + labelString);
            }
        }

        private double[] parseFeatures(String[] parts) {
            double[] features = new double[totalColumns - 1];
            int k = 0;

            for (int i = 0; i < totalColumns; i++) {
                if (i == labelIndex) {
                    continue;
                }
                String value = parts[i].trim();
                try {
                    features[k++] = value.isEmpty()
                            ? Double.NaN
                            : Double.parseDouble(value);
                } catch (NumberFormatException exception) {
                    throw new IllegalArgumentException(
                            "bad feature value '" + value + "' at column " + i
                    );
                }
            }

            return features;
        }
    }
}

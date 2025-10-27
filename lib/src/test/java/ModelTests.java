package com.example.scalarml;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;

class ModelTests {

    @Test @DisplayName("train: returns artifact with positive size")
    void train_Success_ok() {
        var model = new Model(new int[]{2, 4, 1}, new String[]{"tanh","identity"});
        var ds = InMemoryDatasets.xor(); // provide a tiny in-memory dataset helper
        ds.validate(); ds.store();

        var art = model.train(ds, new Hyperparams(0.1, 4, 200));
        assertNotNull(art);
        assertTrue(art.getSizeBytes() > 0);
    }

    @Test @DisplayName("train: null dataset throws")
    void train_NullDataset_throws() {
        var model = new Model(new int[]{1,1}, new String[]{"identity"});
        assertThrows(InvalidArgumentException.class, () -> model.train(null, new Hyperparams(0.1,1,1)));
    }

    @Test @DisplayName("train: unvalidated dataset throws")
    void train_UnvalidatedDataset_throws() {
        var model = new Model(new int[]{1,1}, new String[]{"identity"});
        var ds = InMemoryDatasets.xor(); // not validated / stored
        assertThrows(ValidationException.class, () -> model.train(ds, new Hyperparams(0.1,4,5)));
    }
}

package unit_tests;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;
import nn.MLP;

class ModelTests {

    @Test @DisplayName("train: returns artifact with positive size")
    void trainSuccess() {
        var model = new MLP(new int[]{2, 4, 1}, new String[]{"tanh","identity"});
        var ds = InMemoryDatasets.xor();
        ds.validate(); ds.store();

        var art = model.train(ds, new Hyperparams(0.1, 4, 200));
        assertNotNull(art);
        assertTrue(art.getSizeBytes() > 0);
    }

    @Test @DisplayName("train: null dataset throws")
    void trainNullDataset() {
        var model = new Model(new int[]{1,1}, new String[]{"identity"});
        assertThrows(InvalidArgumentException.class, () -> model.train(null, new Hyperparams(0.1,1,1)));
    }

    @Test @DisplayName("train: unvalidated dataset throws")
    void trainUnvalidatedDataset() {
        var model = new Model(new int[]{1,1}, new String[]{"identity"});
        var ds = InMemoryDatasets.xor(); // not validated / stored
        assertThrows(ValidationException.class, () -> model.train(ds, new Hyperparams(0.1,4,5)));
    }
}

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.*;

class PredictionTests {

    @Test @DisplayName("send: returns accepted request with id")
    void sendAccepted() {
        var model = new Model(new int[]{2,4,1}, new String[]{"tanh","identity"});
        var ds = InMemoryDatasets.xor(); ds.validate(); ds.store();

        var req = new PredictionRequest();
        var handle = req.send(model, ds);
        assertNotNull(handle.getRequestId());
        assertEquals(RequestStatus.ACCEPTED, handle.getStatus());
    }

    @Test @DisplayName("send: unready dataset throws")
    void sendUnreadyDataset() {
        var model = new Model(new int[]{1,1}, new String[]{"identity"});
        var ds = InMemoryDatasets.xor();
        var req = new PredictionRequest();
        assertThrows(ValidationException.class, () -> req.send(model, ds));
    }

    @Test @DisplayName("fetch: returns predictions when ready")
    void fetchReady() {
        var result = new PredictionResult("req-123");
        result._debugMarkReady(new double[]{0.1, 0.9});
        var preds = result.fetch();
        assertEquals(2, preds.length);
    }

    @Test @DisplayName("fetch: unknown request throws NotFound")
    void fetchNotFound() {
        var result = new PredictionResult("unknown");
        assertThrows(NotFoundException.class, result::fetch);
    }

    @Test @DisplayName("fetch: transient network error bubbles up")
    void fetchNetworkError() {
        var result = new PredictionResult("req-err");
        result._debugMarkNetworkError(); // test hook to simulate transient failure
        assertThrows(NetworkException.class, result::fetch);
    }
}

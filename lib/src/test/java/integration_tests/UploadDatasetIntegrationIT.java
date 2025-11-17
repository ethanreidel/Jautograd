// package integration_tests;
// import static org.junit.jupiter.api.Assertions.*;
// import static org.mockito.Mockito.*;

// import java.nio.file.Path;
// import java.util.List;

// import org.junit.jupiter.api.DisplayName;
// import org.junit.jupiter.api.Test;

// public class UploadDatasetIntegrationIT {

//     @Test
//     void uploadValidateStoreDiscover() {
//         var datasets = new DatasetRepository.InMemory();
//         var storage = new StorageService.InMemory();
//         var dsSvc = new DatasetService(datasets, storage);

//         var ds = dsSvc.create("mnist", "csv", "label",
//                 List.of(Path.of("fixtures/xor/train.csv"), Path.of("fixtures/xor/test.csv")));
//         dsSvc.validate(ds.getId());
//         dsSvc.store(ds.getId());

//         var available = dsSvc.listReady();
//         assertTrue(available.get(0).isReady());
//     }
// }
// package integration_tests;

// import static org.junit.jupiter.api.Assertions.*;

// import org.junit.jupiter.api.DisplayName;
// import org.junit.jupiter.api.Test;

// public class EvaluateModelIntegrationIT {

//     @Test
//     void evaluate_confusion_matrix_stored() {
//         var artifacts = new ArtifactRepository.InMemory();
//         var evalRepo  = new EvaluationRepository.InMemory();

//         var artifact = artifacts.save(new ModelArtifact("mlp-xor", 4096));
//         var evaluator = new Evaluator(new InferenceEngine.withPredictions(Fixtures.mnistLabels(), Fixtures.mnistPreds()));

//         var result = evaluator.run(artifact, 0.2);
//         evalRepo.save(result);

//         assertTrue(result.getAccuracy() >= 0 && result.getAccuracy() <= 1);
//         assertNotNull(result.getConfusionMatrix());
//         assertEquals(result.getId(), evalRepo.get(result.getId()).getId());
//     }
// }
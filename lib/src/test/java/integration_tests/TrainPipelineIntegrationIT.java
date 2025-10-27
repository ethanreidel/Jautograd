package integration_tests;



import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class TrainPipelineIntegrationIT {

    @Test
    void trainEnd2End() {
        var datasets = new DatasetRepository.InMemory();
        var artifacts = new ArtifactRepository.InMemory();
        var jobs = new JobRepository.InMemory();
        var metricsRepo = new MetricsRepository.InMemory();

        var ds = Dataset.create("xor", "csv", "label", Fixtures.mnistFiles());
        ds.markReady();
        datasets.save(ds);

        var cfg = ModelConfig.define(
            "MLP", new int[]{784, 128, 10}, new String[]{"relu","softmax"},
            "sgd", ds.getName(), new Hyperparams(1e-3, 64, 2)
        );

        var clock = Clock.fixed(Instant.parse("2025-01-01T00:00:00Z"), ZoneOffset.UTC);
        var trainer = new Trainer(artifacts, jobs, metricsRepo, clock);
        var loader  = new DataLoader(64, true); //shuffle param

        var job = trainer.train(new TrainingJob(cfg, ds, loader));

        assertEquals(JobState.COMPLETED, job.getState());
        assertNotNull(job.getArtifactId());
        assertTrue(artifacts.find(job.getArtifactId()).getSizeBytes() > 0);

        var history = metricsRepo.forJob(job.getId());
        assertEquals(2, history.size());
        assertEquals(0.9, history.get(0).loss(), 1e-6);
        assertEquals(0.6, history.get(0).accuracy(), 1e-6);
        assertEquals(0.6, history.get(1).loss(), 1e-6);
        assertEquals(0.8, history.get(1).accuracy(), 1e-6);
    }
}

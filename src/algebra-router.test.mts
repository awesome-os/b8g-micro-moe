import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { after, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { HEAD_LABELS, TRAIN_SAMPLES, VAL_SAMPLES } from './dataset.mts';
import { loadAlgebraRouter } from './infer.mts';
import { trainAndSaveRouter } from './train-router.mts';

const THIS_DIR = dirname(fileURLToPath(import.meta.url));
const CLI_PATH = resolve(THIS_DIR, 'cli.mts');
const DIST_MODEL_PATH = resolve(THIS_DIR, 'dist', 'algebra-router.b8p');
const HEAD_NAMES = ['ontology', 'element', 'locality', 'runtime', 'intent'] as const;

describe('b8g-algebra-router', () => {
  const tempDir = mkdtempSync(join(tmpdir(), 'b8g-algebra-router-test-'));
  const tempModelPath = join(tempDir, 'algebra-router.b8p');

  after(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('keeps TRAIN_SAMPLES and VAL_SAMPLES strictly disjoint', () => {
    assert.ok(TRAIN_SAMPLES.length >= 180, `Expected >= 180 train samples, got ${TRAIN_SAMPLES.length}`);
    assert.ok(VAL_SAMPLES.length >= 48, `Expected >= 48 val samples, got ${VAL_SAMPLES.length}`);

    const trainTexts = new Set(TRAIN_SAMPLES.map((s) => s.text.trim().toLowerCase()));
    for (const valSample of VAL_SAMPLES) {
      assert.equal(
        trainTexts.has(valSample.text.trim().toLowerCase()),
        false,
        `Validation sample leaked into TRAIN_SAMPLES: "${valSample.text}"`,
      );
    }
  });

  it('trainAndSaveRouter() produces < 256 KB .b8p with >= 0.88 overallValAccuracy on unseen VAL_SAMPLES', () => {
    const metrics = trainAndSaveRouter(tempModelPath);

    assert.equal(metrics.modelPath, tempModelPath);
    assert.equal(existsSync(tempModelPath), true);

    const diskSize = statSync(tempModelPath).size;
    assert.equal(metrics.sizeBytes, diskSize);
    assert.ok(metrics.sizeBytes > 0, 'Model file must be non-empty');
    assert.ok(
      metrics.sizeBytes < 256 * 1024,
      `Expected .b8p < 256 KB, got ${metrics.sizeBytes} bytes`,
    );
    assert.ok(Number.isFinite(metrics.trainLoss) && metrics.trainLoss >= 0);
    assert.ok(
      metrics.overallValAccuracy >= 0.88,
      `Expected overallValAccuracy >= 0.88, got ${metrics.overallValAccuracy.toFixed(4)}`,
    );

    for (const head of HEAD_NAMES) {
      const acc = metrics.valAccuracy[head];
      assert.ok(
        typeof acc === 'number' && Number.isFinite(acc) && acc >= 0.80,
        `Expected valAccuracy.${head} >= 0.80, got ${String(acc)}`,
      );
    }
  });

  it('loadAlgebraRouter() classifies natural-language queries across all 5 heads in < 5 ms', () => {
    const modelPath = existsSync(DIST_MODEL_PATH) ? DIST_MODEL_PATH : tempModelPath;
    const router = loadAlgebraRouter(modelPath);

    // Single warmup pass for V8 JIT stabilization
    router.classify('warmup query');

    const cases: ReadonlyArray<{
      query: string;
      expectedIntent: string;
      expectedRuntime?: string;
    }> = [
      {
        query: 'compile C11 display library to wasm',
        expectedIntent: 'build-compile',
        expectedRuntime: 'holo-c11',
      },
      {
        query: 'open split browser shell with tab groups',
        expectedIntent: 'open-gui-shell',
      },
      {
        query: 'deploy b8g-production-rack linux server',
        expectedIntent: 'deploy-rack',
      },
      {
        query: 'export my data after 30 day lease revoked',
        expectedIntent: 'export-data',
      },
    ];

    for (const tc of cases) {
      const pred = router.classify(tc.query);

      assert.ok(
        pred.latencyMs < 5,
        `Expected latencyMs < 5 ms for "${tc.query}", got ${pred.latencyMs.toFixed(3)} ms`,
      );
      assert.equal(
        pred.escalateToLlm,
        false,
        `Expected escalateToLlm=false for "${tc.query}", got true (minConfidence=${pred.minConfidence})`,
      );
      assert.equal(
        pred.heads.intent.label,
        tc.expectedIntent,
        `Expected intent=${tc.expectedIntent} for "${tc.query}", got ${pred.heads.intent.label}`,
      );
      if (tc.expectedRuntime !== undefined) {
        assert.equal(
          pred.heads.runtime.label,
          tc.expectedRuntime,
          `Expected runtime=${tc.expectedRuntime} for "${tc.query}", got ${pred.heads.runtime.label}`,
        );
      }

      for (const head of HEAD_NAMES) {
        const h = pred.heads[head];
        const allowed = HEAD_LABELS[head] as readonly string[];
        assert.ok(
          allowed.includes(h.label),
          `Head ${head} predicted unknown label "${h.label}"`,
        );
        assert.ok(
          h.atomId !== undefined && h.atomId !== null && String(h.atomId).length > 0,
          `Head ${head} missing atomId`,
        );
        assert.ok(
          h.confidence > 0 && h.confidence <= 1,
          `Head ${head} confidence out of range: ${h.confidence}`,
        );
      }
    }
  });

  it('triggers escalateToLlm: true for unknown gibberish', () => {
    const modelPath = existsSync(DIST_MODEL_PATH) ? DIST_MODEL_PATH : tempModelPath;
    const router = loadAlgebraRouter(modelPath);
    const pred = router.classify('qqqzzz xxxvvv');

    assert.equal(
      pred.escalateToLlm,
      true,
      `Expected escalateToLlm=true for gibberish, got false (intent=${pred.heads.intent.label} @ ${pred.heads.intent.confidence})`,
    );
  });

  it('crystallizeSample() immediately adapts predictions to target labels with high confidence', () => {
    const modelPath = existsSync(DIST_MODEL_PATH) ? DIST_MODEL_PATH : tempModelPath;
    const router = loadAlgebraRouter(modelPath);
    const query = 'launch my secret dragon forge';

    router.crystallizeSample(query, { intent: 'deploy-rack', element: 'Au' });
    const afterCrystallize = router.classify(query);

    assert.equal(afterCrystallize.heads.intent.label, 'deploy-rack');
    assert.equal(afterCrystallize.heads.element.label, 'Au');
    assert.ok(
      afterCrystallize.heads.intent.confidence >= 0.80,
      `Expected high intent confidence after crystallizeSample, got ${afterCrystallize.heads.intent.confidence}`,
    );
    assert.ok(
      afterCrystallize.heads.element.confidence >= 0.80,
      `Expected high element confidence after crystallizeSample, got ${afterCrystallize.heads.element.confidence}`,
    );
  });

  it('CLI smoke test exits 0 and prints structured output', () => {
    const modelPath = existsSync(DIST_MODEL_PATH) ? DIST_MODEL_PATH : tempModelPath;
    const query = 'I want a split browser with frozen tabs';

    const textOut = execFileSync(
      process.execPath,
      [CLI_PATH, '--model', modelPath, query],
      { encoding: 'utf8', shell: false },
    );

    assert.match(textOut, /ontology/i);
    assert.match(textOut, /element/i);
    assert.match(textOut, /locality/i);
    assert.match(textOut, /runtime/i);
    assert.match(textOut, /intent/i);
    assert.match(textOut, /atomId=/);
    assert.match(textOut, /escalateToLlm:/);
    assert.match(textOut, /latencyMs:/);

    const jsonOut = execFileSync(
      process.execPath,
      [CLI_PATH, '--model', modelPath, '--json', query],
      { encoding: 'utf8', shell: false },
    );

    const parsed = JSON.parse(jsonOut) as {
      query: string;
      heads: Record<string, { label: string; atomId: string | number; confidence: number }>;
      minConfidence: number;
      escalateToLlm: boolean;
      latencyMs: number;
    };

    assert.equal(parsed.query, query);
    assert.equal(typeof parsed.escalateToLlm, 'boolean');
    assert.equal(typeof parsed.latencyMs, 'number');
    for (const head of HEAD_NAMES) {
      assert.ok(parsed.heads[head], `Missing head ${head} in JSON output`);
      assert.equal(typeof parsed.heads[head].label, 'string');
      assert.equal(typeof parsed.heads[head].confidence, 'number');
    }
  });
});

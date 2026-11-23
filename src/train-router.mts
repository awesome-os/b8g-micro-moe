import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  HEAD_LABELS,
  HEAD_NAMES,
  TRAIN_SAMPLES,
  VAL_SAMPLES,
  type HeadName,
  type LabeledSample,
} from './dataset.mts';

export const DEFAULT_VOCAB_SIZE = 1024;
export const DEFAULT_HIDDEN_SIZE = 48;
export const KNOWN_WORD_BITSET_BYTES = 2048; // 16384 bits (3-hash Bloom filter)
export const ROUTER_MAGIC = 'B8GR';
export const ROUTER_VERSION = 1;

export interface TensorDescriptor {
  name: string;
  shape: number[];
  offset: number;
  length: number;
  scale: number;
}

export interface RouterHeader {
  format: 'B8G_ROUTER_V1';
  vocabSize: number;
  hiddenSize: number;
  headOrder: HeadName[];
  heads: Record<HeadName, string[]>;
  knownWordBits: string;
  tensors: TensorDescriptor[];
  trainMetrics?: {
    epochs: number;
    trainSamples: number;
    valSamples: number;
    trainLoss: number;
    valAccuracy: Record<HeadName, number>;
    overallValAccuracy: number;
  };
}

export interface TrainResult {
  modelPath: string;
  sizeBytes: number;
  trainLoss: number;
  valAccuracy: Record<HeadName, number>;
  overallValAccuracy: number;
}

/** Deterministic 32-bit FNV-1a hash. */
export function fnv1a32(input: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/** Extract normalized word tokens from text. */
export function extractWordTokens(text: string): string[] {
  const lower = text.toLowerCase();
  const matches = lower.match(/[a-z0-9]+/g);
  if (!matches) return [];
  const out: string[] = [];
  for (let i = 0; i < matches.length; i++) {
    const tok = matches[i]!;
    if (tok.length >= 2 || tok === 'c') {
      out.push(tok);
    }
  }
  return out;
}

/**
 * Featurize text into a normalized sparse map of bucket -> weight
 * using word tokens, word bigrams, and character 3-grams hashed via FNV-1a.
 */
export function featurizeText(
  text: string,
  buckets: number = DEFAULT_VOCAB_SIZE,
): Map<number, number> {
  const vec = new Map<number, number>();
  const words = extractWordTokens(text);
  if (words.length === 0) return vec;

  const addFeature = (key: string, weight: number): void => {
    const idx = fnv1a32(key) % buckets;
    vec.set(idx, (vec.get(idx) ?? 0) + weight);
  };

  for (let i = 0; i < words.length; i++) {
    const w = words[i]!;
    addFeature(`w:${w}`, 2.0);
    if (i + 1 < words.length) {
      addFeature(`b:${w}_${words[i + 1]!}`, 1.2);
    }
    const padded = `^${w}$`;
    for (let j = 0; j <= padded.length - 3; j++) {
      addFeature(`g:${padded.slice(j, j + 3)}`, 0.45);
    }
  }

  let sqSum = 0;
  for (const val of vec.values()) {
    sqSum += val * val;
  }
  if (sqSum > 0) {
    const invNorm = 1 / Math.sqrt(sqSum);
    for (const [idx, val] of vec.entries()) {
      vec.set(idx, val * invNorm);
    }
  }
  return vec;
}

/** Deterministic PRNG (Mulberry32). */
function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function quantizeInt8(arr: Float32Array): { data: Int8Array; scale: number } {
  let maxAbs = 0;
  for (let i = 0; i < arr.length; i++) {
    const a = Math.abs(arr[i]!);
    if (a > maxAbs) maxAbs = a;
  }
  const scale = maxAbs > 1e-8 ? maxAbs / 127 : 1e-8;
  const invScale = 1 / scale;
  const data = new Int8Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    let q = Math.round(arr[i]! * invScale);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    data[i] = q;
  }
  return { data, scale };
}

function dequantizeInt8(data: Int8Array, scale: number): Float32Array {
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    out[i] = data[i]! * scale;
  }
  return out;
}

function softmaxInPlace(logits: Float32Array): void {
  let maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i]! > maxVal) maxVal = logits[i]!;
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i]! - maxVal);
    logits[i] = e;
    sum += e;
  }
  const invSum = sum > 0 ? 1 / sum : 1 / logits.length;
  for (let i = 0; i < logits.length; i++) {
    logits[i] = logits[i]! * invSum;
  }
}

interface EncodedSample {
  indices: Int32Array;
  values: Float32Array;
  targets: Int32Array;
}

function encodeSamples(
  samples: readonly LabeledSample[],
  vocabSize: number,
): EncodedSample[] {
  const labelMaps: Record<HeadName, Map<string, number>> = {
    ontology: new Map(),
    element: new Map(),
    locality: new Map(),
    runtime: new Map(),
    intent: new Map(),
  };
  for (const head of HEAD_NAMES) {
    const labels = HEAD_LABELS[head];
    for (let i = 0; i < labels.length; i++) {
      labelMaps[head].set(labels[i]!, i);
    }
  }

  return samples.map((s) => {
    const feat = featurizeText(s.text, vocabSize);
    const indices = new Int32Array(feat.size);
    const values = new Float32Array(feat.size);
    let k = 0;
    for (const [idx, val] of feat.entries()) {
      indices[k] = idx;
      values[k] = val;
      k++;
    }
    const targets = new Int32Array(HEAD_NAMES.length);
    for (let h = 0; h < HEAD_NAMES.length; h++) {
      const head = HEAD_NAMES[h]!;
      const label = s.labels[head];
      const idx = labelMaps[head].get(label);
      if (idx === undefined) {
        throw new Error(`Unknown label "${label}" for head "${head}"`);
      }
      targets[h] = idx;
    }
    return { indices, values, targets };
  });
}

function buildKnownWordBitset(samples: readonly LabeledSample[]): Uint8Array {
  const bits = new Uint8Array(KNOWN_WORD_BITSET_BYTES);
  const totalBits = KNOWN_WORD_BITSET_BYTES * 8;
  for (let i = 0; i < samples.length; i++) {
    const words = extractWordTokens(samples[i]!.text);
    for (let j = 0; j < words.length; j++) {
      const w = words[j]!;
      for (let h = 0; h < 3; h++) {
        const bit = fnv1a32(`kw${h}:${w}`) % totalBits;
        bits[bit >>> 3] = bits[bit >>> 3]! | (1 << (bit & 7));
      }
    }
  }
  return bits;
}

function assertDisjointSplit(
  train: readonly LabeledSample[],
  val: readonly LabeledSample[],
): void {
  const trainSet = new Set<string>();
  for (const s of train) {
    trainSet.add(s.text.trim().toLowerCase());
  }
  for (const s of val) {
    const norm = s.text.trim().toLowerCase();
    if (trainSet.has(norm)) {
      throw new Error(`Data leakage detected: "${s.text}" appears in both TRAIN and VAL`);
    }
  }
}

export function getDefaultModelPath(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  return resolve(here, 'dist', 'algebra-router.b8p');
}

export function trainAndSaveRouter(outPath?: string): TrainResult {
  assertDisjointSplit(TRAIN_SAMPLES, VAL_SAMPLES);

  const V = DEFAULT_VOCAB_SIZE;
  const H = DEFAULT_HIDDEN_SIZE;
  const epochs = 80;
  const lrStart = 0.28;
  const lrEnd = 0.025;
  const weightDecay = 1.5e-4;
  const rng = createRng(0xb86a16e8);

  const trainData = encodeSamples(TRAIN_SAMPLES, V);
  const valData = encodeSamples(VAL_SAMPLES, V);

  // Shared encoder weights: W1 [V x H], b1 [H]
  const W1 = new Float32Array(V * H);
  const b1 = new Float32Array(H);
  const w1Scale = Math.sqrt(2 / V);
  for (let i = 0; i < W1.length; i++) {
    W1[i] = (rng() * 2 - 1) * w1Scale;
  }

  // 5 linear softmax heads: W2[head] [numClasses x H], b2[head] [numClasses]
  const W2: Record<HeadName, Float32Array> = {} as Record<HeadName, Float32Array>;
  const b2: Record<HeadName, Float32Array> = {} as Record<HeadName, Float32Array>;
  const w2Scale = Math.sqrt(1 / H);
  for (const head of HEAD_NAMES) {
    const C = HEAD_LABELS[head].length;
    const w = new Float32Array(C * H);
    for (let i = 0; i < w.length; i++) {
      w[i] = (rng() * 2 - 1) * w2Scale;
    }
    W2[head] = w;
    b2[head] = new Float32Array(C);
  }

  // Scratch buffers
  const z1 = new Float32Array(H);
  const a1 = new Float32Array(H);
  const da1 = new Float32Array(H);
  const logitsByHead: Float32Array[] = HEAD_NAMES.map(
    (head) => new Float32Array(HEAD_LABELS[head].length),
  );
  const order = new Int32Array(trainData.length);
  for (let i = 0; i < order.length; i++) order[i] = i;

  let finalTrainLoss = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Fisher-Yates shuffle
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i]!;
      order[i] = order[j]!;
      order[j] = tmp;
    }

    const progress = epoch / Math.max(1, epochs - 1);
    const lr = lrEnd + 0.5 * (lrStart - lrEnd) * (1 + Math.cos(Math.PI * progress));
    let epochLoss = 0;

    for (let si = 0; si < order.length; si++) {
      const sampleItem = trainData[order[si]!]!;
      const idxs = sampleItem.indices;
      const vals = sampleItem.values;

      // Forward shared encoder
      for (let h = 0; h < H; h++) {
        z1[h] = b1[h]!;
      }
      for (let k = 0; k < idxs.length; k++) {
        const rowOff = idxs[k]! * H;
        const v = vals[k]!;
        for (let h = 0; h < H; h++) {
          z1[h] = z1[h]! + W1[rowOff + h]! * v;
        }
      }
      for (let h = 0; h < H; h++) {
        const zh = z1[h]!;
        a1[h] = zh > 0 ? zh : 0;
        da1[h] = 0;
      }

      // Forward & backward 5 heads
      for (let hi = 0; hi < HEAD_NAMES.length; hi++) {
        const head = HEAD_NAMES[hi]!;
        const C = HEAD_LABELS[head].length;
        const Wh = W2[head];
        const bh = b2[head];
        const probs = logitsByHead[hi]!;
        const target = sampleItem.targets[hi]!;

        for (let c = 0; c < C; c++) {
          let sum = bh[c]!;
          const cOff = c * H;
          for (let h = 0; h < H; h++) {
            sum += Wh[cOff + h]! * a1[h]!;
          }
          probs[c] = sum;
        }
        softmaxInPlace(probs);

        const pTarget = Math.max(probs[target]!, 1e-12);
        epochLoss -= Math.log(pTarget);

        for (let c = 0; c < C; c++) {
          const dlogit = probs[c]! - (c === target ? 1 : 0);
          const cOff = c * H;
          for (let h = 0; h < H; h++) {
            const wVal = Wh[cOff + h]!;
            da1[h] = da1[h]! + dlogit * wVal;
            Wh[cOff + h] = wVal - lr * (dlogit * a1[h]! + weightDecay * wVal);
          }
          bh[c] = bh[c]! - lr * dlogit;
        }
      }

      // Backward shared encoder
      for (let h = 0; h < H; h++) {
        const dz = z1[h]! > 0 ? da1[h]! : 0;
        da1[h] = dz;
        b1[h] = b1[h]! - lr * dz;
      }
      for (let k = 0; k < idxs.length; k++) {
        const rowOff = idxs[k]! * H;
        const v = vals[k]!;
        for (let h = 0; h < H; h++) {
          const dz = da1[h]!;
          if (dz !== 0) {
            W1[rowOff + h] = W1[rowOff + h]! - lr * dz * v;
          }
        }
      }
    }

    // L2 weight decay on W1 once per epoch
    const decayFactor = 1 - lr * weightDecay;
    for (let i = 0; i < W1.length; i++) {
      W1[i] = W1[i]! * decayFactor;
    }

    finalTrainLoss = epochLoss / (trainData.length * HEAD_NAMES.length);
  }

  // Quantize all tensors to INT8 and pack into .b8p
  const rawTensors: Array<{ name: string; shape: number[]; array: Float32Array }> = [
    { name: 'W1', shape: [V, H], array: W1 },
    { name: 'b1', shape: [H], array: b1 },
  ];
  for (const head of HEAD_NAMES) {
    const C = HEAD_LABELS[head].length;
    rawTensors.push({ name: `W2_${head}`, shape: [C, H], array: W2[head] });
    rawTensors.push({ name: `b2_${head}`, shape: [C], array: b2[head] });
  }

  const descriptors: TensorDescriptor[] = [];
  const quantizedChunks: Int8Array[] = [];
  const dequantizedByName = new Map<string, Float32Array>();
  let byteOffset = 0;

  for (const t of rawTensors) {
    const q = quantizeInt8(t.array);
    descriptors.push({
      name: t.name,
      shape: t.shape,
      offset: byteOffset,
      length: q.data.byteLength,
      scale: q.scale,
    });
    quantizedChunks.push(q.data);
    dequantizedByName.set(t.name, dequantizeInt8(q.data, q.scale));
    byteOffset += q.data.byteLength;
  }

  // Evaluate honest held-out validation accuracy using the quantized/dequantized INT8 weights
  const qW1 = dequantizedByName.get('W1')!;
  const qb1 = dequantizedByName.get('b1')!;
  const correctByHead = new Int32Array(HEAD_NAMES.length);

  for (let vi = 0; vi < valData.length; vi++) {
    const item = valData[vi]!;
    for (let h = 0; h < H; h++) z1[h] = qb1[h]!;
    for (let k = 0; k < item.indices.length; k++) {
      const rowOff = item.indices[k]! * H;
      const v = item.values[k]!;
      for (let h = 0; h < H; h++) {
        z1[h] = z1[h]! + qW1[rowOff + h]! * v;
      }
    }
    for (let h = 0; h < H; h++) {
      a1[h] = z1[h]! > 0 ? z1[h]! : 0;
    }

    for (let hi = 0; hi < HEAD_NAMES.length; hi++) {
      const head = HEAD_NAMES[hi]!;
      const C = HEAD_LABELS[head].length;
      const qWh = dequantizedByName.get(`W2_${head}`)!;
      const qbh = dequantizedByName.get(`b2_${head}`)!;
      let bestC = 0;
      let bestScore = -Infinity;
      for (let c = 0; c < C; c++) {
        let sum = qbh[c]!;
        const cOff = c * H;
        for (let h = 0; h < H; h++) {
          sum += qWh[cOff + h]! * a1[h]!;
        }
        if (sum > bestScore) {
          bestScore = sum;
          bestC = c;
        }
      }
      if (bestC === item.targets[hi]!) {
        correctByHead[hi] = correctByHead[hi]! + 1;
      }
    }
  }

  const valAccuracy = {} as Record<HeadName, number>;
  let accSum = 0;
  for (let hi = 0; hi < HEAD_NAMES.length; hi++) {
    const head = HEAD_NAMES[hi]!;
    const acc = correctByHead[hi]! / valData.length;
    valAccuracy[head] = acc;
    accSum += acc;
  }
  const overallValAccuracy = accSum / HEAD_NAMES.length;

  const knownWordBits = Buffer.from(buildKnownWordBitset(TRAIN_SAMPLES)).toString('base64');
  const headsRecord = {} as Record<HeadName, string[]>;
  for (const head of HEAD_NAMES) {
    headsRecord[head] = [...HEAD_LABELS[head]];
  }

  const header: RouterHeader = {
    format: 'B8G_ROUTER_V1',
    vocabSize: V,
    hiddenSize: H,
    headOrder: [...HEAD_NAMES],
    heads: headsRecord,
    knownWordBits,
    tensors: descriptors,
    trainMetrics: {
      epochs,
      trainSamples: TRAIN_SAMPLES.length,
      valSamples: VAL_SAMPLES.length,
      trainLoss: finalTrainLoss,
      valAccuracy,
      overallValAccuracy,
    },
  };

  const headerBytes = Buffer.from(JSON.stringify(header), 'utf8');
  const totalBytes = 4 + 4 + 4 + headerBytes.byteLength + byteOffset;
  const outBuf = Buffer.allocUnsafe(totalBytes);
  outBuf.write(ROUTER_MAGIC, 0, 4, 'ascii');
  outBuf.writeUInt32LE(ROUTER_VERSION, 4);
  outBuf.writeUInt32LE(headerBytes.byteLength, 8);
  headerBytes.copy(outBuf, 12);

  let writePos = 12 + headerBytes.byteLength;
  for (const chunk of quantizedChunks) {
    const view = new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
    outBuf.set(view, writePos);
    writePos += chunk.byteLength;
  }

  const modelPath = outPath ? resolve(outPath) : getDefaultModelPath();
  mkdirSync(dirname(modelPath), { recursive: true });
  writeFileSync(modelPath, outBuf);

  return {
    modelPath,
    sizeBytes: outBuf.byteLength,
    trainLoss: finalTrainLoss,
    valAccuracy,
    overallValAccuracy,
  };
}

const isMain =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]).toLowerCase() ===
    resolve(fileURLToPath(import.meta.url)).toLowerCase();

if (isMain) {
  const result = trainAndSaveRouter(process.argv[2]);
  console.log('=== B8G_ROUTER_V1 Training & Held-Out Validation ===');
  console.log(`Model Path          : ${result.modelPath}`);
  console.log(`Binary Size         : ${result.sizeBytes} bytes (${(result.sizeBytes / 1024).toFixed(2)} KB)`);
  console.log(`Train Samples       : ${TRAIN_SAMPLES.length}`);
  console.log(`Held-Out Val Samples: ${VAL_SAMPLES.length}`);
  console.log(`Final Train Loss    : ${result.trainLoss.toFixed(6)}`);
  console.log('----------------------------------------------------');
  console.log('Head         | Held-Out Val Accuracy');
  console.log('-------------|----------------------');
  for (const head of HEAD_NAMES) {
    const pct = (result.valAccuracy[head] * 100).toFixed(2);
    console.log(`${head.padEnd(12)} | ${pct}%`);
  }
  console.log('-------------|----------------------');
  console.log(`OVERALL      | ${(result.overallValAccuracy * 100).toFixed(2)}%`);
}

import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { resolve } from 'node:path';
import { HEAD_NAMES, type HeadName } from './dataset.mts';
import {
  extractWordTokens,
  featurizeText,
  fnv1a32,
  getDefaultModelPath,
  ROUTER_MAGIC,
  ROUTER_VERSION,
  type RouterHeader,
} from './train-router.mts';

export interface HeadPrediction {
  label: string;
  atomId: number;
  confidence: number;
}

export interface RouterPrediction {
  heads: Record<HeadName, HeadPrediction>;
  minConfidence: number;
  escalateToLlm: boolean;
  latencyMs: number;
}

function softmax(logits: Float32Array): Float32Array {
  let maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i]! > maxVal) maxVal = logits[i]!;
  }
  const probs = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i]! - maxVal);
    probs[i] = e;
    sum += e;
  }
  const invSum = sum > 0 ? 1 / sum : 1 / logits.length;
  for (let i = 0; i < probs.length; i++) {
    probs[i] = probs[i]! * invSum;
  }
  return probs;
}

export class AlgebraRouter {
  private readonly vocabSize: number;
  private readonly hiddenSize: number;
  private readonly headOrder: readonly HeadName[];
  private readonly headLabels: Record<HeadName, readonly string[]>;
  private readonly knownWordBits: Uint8Array;
  private readonly W1: Float32Array;
  private readonly b1: Float32Array;
  private readonly W2: Record<HeadName, Float32Array>;
  private readonly b2: Record<HeadName, Float32Array>;

  constructor(buffer: Uint8Array) {
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    if (buffer.byteLength < 12) {
      throw new Error('Invalid .b8p router file: truncated header');
    }
    const magic = String.fromCharCode(
      buffer[0]!,
      buffer[1]!,
      buffer[2]!,
      buffer[3]!,
    );
    if (magic !== ROUTER_MAGIC) {
      throw new Error(`Invalid .b8p magic "${magic}", expected "${ROUTER_MAGIC}"`);
    }
    const version = view.getUint32(4, true);
    if (version !== ROUTER_VERSION) {
      throw new Error(`Unsupported .b8p router version ${version}`);
    }
    const headerLen = view.getUint32(8, true);
    if (12 + headerLen > buffer.byteLength) {
      throw new Error('Invalid .b8p router file: header length out of bounds');
    }

    const headerJson = Buffer.from(
      buffer.buffer,
      buffer.byteOffset + 12,
      headerLen,
    ).toString('utf8');
    const header = JSON.parse(headerJson) as RouterHeader;

    this.vocabSize = header.vocabSize;
    this.hiddenSize = header.hiddenSize;
    this.headOrder = header.headOrder;
    this.headLabels = header.heads;
    this.knownWordBits = new Uint8Array(Buffer.from(header.knownWordBits, 'base64'));

    const payloadBase = buffer.byteOffset + 12 + headerLen;
    const tensorMap = new Map<string, Float32Array>();
    for (const desc of header.tensors) {
      if (12 + headerLen + desc.offset + desc.length > buffer.byteLength) {
        throw new Error(`Tensor "${desc.name}" out of bounds in .b8p file`);
      }
      const q = new Int8Array(buffer.buffer, payloadBase + desc.offset, desc.length);
      const f = new Float32Array(desc.length);
      const scale = desc.scale;
      for (let i = 0; i < desc.length; i++) {
        f[i] = q[i]! * scale;
      }
      tensorMap.set(desc.name, f);
    }

    const w1 = tensorMap.get('W1');
    const b1 = tensorMap.get('b1');
    if (!w1 || !b1) {
      throw new Error('Missing shared encoder tensors W1/b1 in .b8p file');
    }
    this.W1 = w1;
    this.b1 = b1;

    const w2Record = {} as Record<HeadName, Float32Array>;
    const b2Record = {} as Record<HeadName, Float32Array>;
    for (const head of HEAD_NAMES) {
      const wh = tensorMap.get(`W2_${head}`);
      const bh = tensorMap.get(`b2_${head}`);
      if (!wh || !bh) {
        throw new Error(`Missing head tensors for "${head}" in .b8p file`);
      }
      w2Record[head] = wh;
      b2Record[head] = bh;
    }
    this.W2 = w2Record;
    this.b2 = b2Record;
  }

  private countKnownWords(text: string): number {
    const words = extractWordTokens(text);
    const totalBits = this.knownWordBits.length * 8;
    if (totalBits === 0) return words.length;
    let known = 0;
    for (let i = 0; i < words.length; i++) {
      const w = words[i]!;
      let present = true;
      for (let h = 0; h < 3; h++) {
        const bit = fnv1a32(`kw${h}:${w}`) % totalBits;
        if ((this.knownWordBits[bit >>> 3]! & (1 << (bit & 7))) === 0) {
          present = false;
          break;
        }
      }
      if (present) known++;
    }
    return known;
  }

  private markWordsKnown(text: string): void {
    const words = extractWordTokens(text);
    const totalBits = this.knownWordBits.length * 8;
    if (totalBits === 0) return;
    for (let i = 0; i < words.length; i++) {
      const w = words[i]!;
      for (let h = 0; h < 3; h++) {
        const bit = fnv1a32(`kw${h}:${w}`) % totalBits;
        this.knownWordBits[bit >>> 3] =
          this.knownWordBits[bit >>> 3]! | (1 << (bit & 7));
      }
    }
  }

  /**
   * Run the shared encoder ONCE and compute all 5 softmax heads.
   */
  classify(text: string): RouterPrediction {
    const t0 = performance.now();
    const features = featurizeText(text, this.vocabSize);
    const knownWords = this.countKnownWords(text);
    const H = this.hiddenSize;

    // Shared encoder forward pass (single pass for all 5 heads)
    const a1 = new Float32Array(H);
    for (let h = 0; h < H; h++) {
      a1[h] = this.b1[h]!;
    }
    for (const [idx, val] of features.entries()) {
      const rowOff = idx * H;
      for (let h = 0; h < H; h++) {
        a1[h] = a1[h]! + this.W1[rowOff + h]! * val;
      }
    }
    for (let h = 0; h < H; h++) {
      if (a1[h]! < 0) a1[h] = 0;
    }

    const heads = {} as Record<HeadName, HeadPrediction>;
    let minConfidence = 1;

    for (let hi = 0; hi < this.headOrder.length; hi++) {
      const head = this.headOrder[hi]!;
      const labels = this.headLabels[head];
      const C = labels.length;
      const Wh = this.W2[head];
      const bh = this.b2[head];
      const logits = new Float32Array(C);

      for (let c = 0; c < C; c++) {
        let sum = bh[c]!;
        const cOff = c * H;
        for (let h = 0; h < H; h++) {
          sum += Wh[cOff + h]! * a1[h]!;
        }
        logits[c] = sum;
      }

      const probs = softmax(logits);
      let bestC = 0;
      let bestP = probs[0]!;
      for (let c = 1; c < C; c++) {
        if (probs[c]! > bestP) {
          bestP = probs[c]!;
          bestC = c;
        }
      }

      if (bestP < minConfidence) minConfidence = bestP;
      heads[head] = {
        label: labels[bestC]!,
        atomId: (hi + 1) * 16 + bestC,
        confidence: bestP,
      };
    }

    const escalateToLlm =
      features.size === 0 ||
      knownWords === 0 ||
      heads.intent.confidence < 0.5;

    const latencyMs = performance.now() - t0;
    return {
      heads,
      minConfidence,
      escalateToLlm,
      latencyMs,
    };
  }

  /**
   * Perform fast online SGD steps on the specified head(s) so a newly learned
   * phrasing routes immediately with high confidence.
   */
  crystallizeSample(
    text: string,
    targets: Partial<Record<HeadName, string>>,
    steps: number = 10,
  ): void {
    const features = featurizeText(text, this.vocabSize);
    if (features.size === 0) return;

    const resolvedTargets: Array<{ head: HeadName; targetIdx: number }> = [];
    for (const head of this.headOrder) {
      const label = targets[head];
      if (label === undefined) continue;
      const idx = this.headLabels[head].indexOf(label);
      if (idx === -1) {
        throw new Error(`Unknown target label "${label}" for head "${head}"`);
      }
      resolvedTargets.push({ head, targetIdx: idx });
    }
    if (resolvedTargets.length === 0) return;

    this.markWordsKnown(text);

    const H = this.hiddenSize;
    const lr = 0.45;
    const z1 = new Float32Array(H);
    const a1 = new Float32Array(H);
    const da1 = new Float32Array(H);

    for (let step = 0; step < steps; step++) {
      // Forward shared encoder
      for (let h = 0; h < H; h++) {
        z1[h] = this.b1[h]!;
      }
      for (const [idx, val] of features.entries()) {
        const rowOff = idx * H;
        for (let h = 0; h < H; h++) {
          z1[h] = z1[h]! + this.W1[rowOff + h]! * val;
        }
      }
      for (let h = 0; h < H; h++) {
        a1[h] = z1[h]! > 0 ? z1[h]! : 0;
        da1[h] = 0;
      }

      // Update each specified head and accumulate encoder gradient
      for (let ti = 0; ti < resolvedTargets.length; ti++) {
        const { head, targetIdx } = resolvedTargets[ti]!;
        const C = this.headLabels[head].length;
        const Wh = this.W2[head];
        const bh = this.b2[head];
        const logits = new Float32Array(C);

        for (let c = 0; c < C; c++) {
          let sum = bh[c]!;
          const cOff = c * H;
          for (let h = 0; h < H; h++) {
            sum += Wh[cOff + h]! * a1[h]!;
          }
          logits[c] = sum;
        }
        const probs = softmax(logits);

        for (let c = 0; c < C; c++) {
          const dlogit = probs[c]! - (c === targetIdx ? 1 : 0);
          const cOff = c * H;
          for (let h = 0; h < H; h++) {
            const wVal = Wh[cOff + h]!;
            da1[h] = da1[h]! + dlogit * wVal;
            Wh[cOff + h] = wVal - lr * dlogit * a1[h]!;
          }
          bh[c] = bh[c]! - lr * dlogit;
        }
      }

      // Update shared encoder lightly for active features
      const encLr = lr * 0.35;
      for (let h = 0; h < H; h++) {
        const dz = z1[h]! > 0 ? da1[h]! : 0;
        da1[h] = dz;
        this.b1[h] = this.b1[h]! - encLr * dz;
      }
      for (const [idx, val] of features.entries()) {
        const rowOff = idx * H;
        for (let h = 0; h < H; h++) {
          const dz = da1[h]!;
          if (dz !== 0) {
            this.W1[rowOff + h] = this.W1[rowOff + h]! - encLr * dz * val;
          }
        }
      }
    }
  }
}

export function loadAlgebraRouter(modelPath?: string): AlgebraRouter {
  const resolved = modelPath ? resolve(modelPath) : getDefaultModelPath();
  const raw = readFileSync(resolved);
  return new AlgebraRouter(raw);
}

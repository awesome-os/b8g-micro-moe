import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { loadAlgebraRouter } from './infer.mts';
import { trainAndSaveRouter } from './train-router.mts';
import type { RouterPrediction } from './infer.mts';

const THIS_DIR = dirname(fileURLToPath(import.meta.url));
export const DEFAULT_MODEL_PATH = resolve(THIS_DIR, 'dist', 'algebra-router.b8p');

const HEAD_ORDER = ['ontology', 'element', 'locality', 'runtime', 'intent'] as const;

export interface ParsedCliArgs {
  query: string;
  modelPath: string;
  json: boolean;
  help: boolean;
}

export function parseCliArgs(argv: readonly string[]): ParsedCliArgs {
  let modelPath = DEFAULT_MODEL_PATH;
  let json = false;
  let help = false;
  const queryParts: string[] = [];

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      help = true;
    } else if (arg === '--json') {
      json = true;
    } else if (arg === '--model' && i + 1 < argv.length) {
      modelPath = resolve(argv[i + 1]);
      i += 1;
    } else if (arg.startsWith('--model=')) {
      modelPath = resolve(arg.slice('--model='.length));
    } else {
      queryParts.push(arg);
    }
  }

  return {
    query: queryParts.join(' ').trim(),
    modelPath,
    json,
    help,
  };
}

export function formatPrediction(query: string, prediction: RouterPrediction): string {
  const lines: string[] = [
    `Query: ${query}`,
    'Heads:',
  ];

  for (const headName of HEAD_ORDER) {
    const h = prediction.heads[headName];
    const pct = (h.confidence * 100).toFixed(2);
    lines.push(
      `  ${headName.padEnd(10)} label=${h.label.padEnd(18)} atomId=${String(h.atomId).padEnd(12)} confidence=${h.confidence.toFixed(4)} (${pct}%)`,
    );
  }

  lines.push(`minConfidence: ${prediction.minConfidence.toFixed(4)}`);
  lines.push(`escalateToLlm: ${String(prediction.escalateToLlm)}`);
  lines.push(`latencyMs:     ${prediction.latencyMs.toFixed(3)}`);
  return lines.join('\n');
}

function usageText(): string {
  return [
    'Usage: .\\bin\\b8g-shim.exe node sections/b8g-algebra-router/cli.mts [options] "<query>"',
    '',
    'Options:',
    '  --model <path>   Path to algebra-router.b8p (default: sections/b8g-algebra-router/dist/algebra-router.b8p)',
    '  --json           Output structured JSON',
    '  -h, --help       Show this help message',
  ].join('\n');
}

export function runCli(argv: readonly string[]): number {
  const parsed = parseCliArgs(argv);

  if (parsed.help) {
    process.stdout.write(`${usageText()}\n`);
    return 0;
  }

  if (parsed.query.length === 0) {
    process.stderr.write(`Error: query string is required.\n\n${usageText()}\n`);
    return 1;
  }

  if (!existsSync(parsed.modelPath)) {
    trainAndSaveRouter(parsed.modelPath);
  }

  const router = loadAlgebraRouter(parsed.modelPath);
  const prediction = router.classify(parsed.query);

  if (parsed.json) {
    process.stdout.write(`${JSON.stringify({ query: parsed.query, ...prediction }, null, 2)}\n`);
  } else {
    process.stdout.write(`${formatPrediction(parsed.query, prediction)}\n`);
  }

  return 0;
}

const isDirectEntry =
  typeof process.argv[1] === 'string' &&
  pathToFileURL(resolve(process.argv[1])).href === import.meta.url;

if (isDirectEntry) {
  process.exitCode = runCli(process.argv.slice(2));
}

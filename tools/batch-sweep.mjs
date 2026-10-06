#!/usr/bin/env node
// SOG / SOG-LE parameter sweep (ffsplat batch_sweep style): encode one PLY
// with every variant of a manifest, decode each back to PLY and record the
// encoded size. See src/cli/batch-sweep.ts for the output layout.
//
// Prerequisite: npm run build (runs from dist/cli.mjs, with worker threads).
// Usage:
//   node tools/batch-sweep.mjs --manifest tools/sweeps/sog-le.yaml \
//       --input scene/point_cloud.ply --output out/sweep [--overwrite] [--no-decode]
//       [--gpu <n|cpu>] [--max-workers <n>] [--sh-iterations <n>] [--webp-effort <0-9>] [--quiet]
import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';

import { parse } from 'yaml';

import { batchSweep } from '../dist/cli.mjs';

const { values: v } = parseArgs({
    options: {
        manifest: { type: 'string' },
        input: { type: 'string' },
        output: { type: 'string' },
        overwrite: { type: 'boolean', default: false },
        'no-decode': { type: 'boolean', default: false },
        gpu: { type: 'string', default: '-1' },
        'gpu-backend': { type: 'string' },
        'max-workers': { type: 'string' },
        'sh-iterations': { type: 'string', default: '10' },
        'webp-effort': { type: 'string' },
        quiet: { type: 'boolean', default: false }
    }
});

if (!v.manifest || !v.input || !v.output) {
    console.error('usage: node tools/batch-sweep.mjs --manifest <yaml|json> --input <scene.ply> --output <dir> [options]');
    process.exit(2);
}

const toInt = s => (s === undefined ? undefined : Number.parseInt(s, 10));

try {
    await batchSweep(parse(readFileSync(v.manifest, 'utf-8')), {
        input: v.input,
        output: v.output,
        overwrite: v.overwrite,
        noDecode: v['no-decode'],
        gpu: v.gpu,
        gpuBackend: v['gpu-backend'],
        maxWorkers: toInt(v['max-workers']),
        iterations: toInt(v['sh-iterations']),
        webpEffort: toInt(v['webp-effort']),
        quiet: v.quiet
    });
} catch (err) {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
}
// webgpu keeps the process alive after the work is done
process.exit(0);

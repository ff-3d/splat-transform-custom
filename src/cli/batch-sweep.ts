// Parameter sweeps over SOG / SOG-LE encodings in one process, modelled on
// ffsplat's batch_sweep entrypoint: a manifest of groups (output format +
// swept axes, product or zip), each variant written as
//
//   <output>/<group>/<label>/encoded/meta.json + *.webp   (the encoding)
//   <output>/<group>/<label>/decoded.ply                   (encoding -> PLY)
//   <output>/<group>/<label>/size.json                     (size breakdown)
//
// plus <output>/<group>/sweep_metrics.csv and <output>/batch_sweep_metrics.csv.
//
// Caching: the input is read once, and deterministic encode stages are memoized
// by content hash across all variants (SogEncodeCache) - SH k-means and every
// WebP encode whose texture a variant does not change (means, sh0, shN, and
// shape textures repeated between groups) run once per sweep. Variants whose
// size.json exists are skipped, so an interrupted sweep resumes.

import { mkdir, readdir, readFile as fsReadFile, rm, stat, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';

import { GraphicsDevice } from 'playcanvas';

import { createDevice, enumerateAdapters } from './node-device';
import { NodeFileSystem, NodeReadFileSystem } from './node-file-system';
import {
    createChunkDataPool,
    getInputFormat,
    readFile,
    WorkerQueue,
    writeSource,
    type ChunkSource,
    type Options
} from '../lib';

type SweepFormat = 'sog' | 'sog-le-codebook' | 'sog-le-minmax' | 'sog-le-hybrid';

type SweepAxis = { key: string; values: number[] };

type SweepGroup = {
    id: string;
    output_format: SweepFormat;
    sweep_mode?: 'product' | 'zip';
    /** Fixed parameter overrides applied to every variant of the group. */
    params?: Record<string, number>;
    axes?: SweepAxis[];
};

type SweepManifest = { groups: SweepGroup[] };

type BatchSweepOptions = {
    input: string;
    output: string;
    /** Re-run variants whose results already exist. */
    overwrite?: boolean;
    /** Skip writing decoded.ply. */
    noDecode?: boolean;
    /** GPU for SH k-means: adapter index, -1 = default adapter, 'cpu' = none. */
    gpu?: string;
    gpuBackend?: string;
    maxWorkers?: number;
    iterations?: number;
    webpEffort?: number;
    quiet?: boolean;
};

// Sweepable keys -> Options field, with the formats each applies to.
const SWEEP_KEYS: Record<string, { option: keyof Options; formats: SweepFormat[] | null }> = {
    'scales.k': { option: 'sogScalesCodebookSize', formats: ['sog'] },
    'quats.max': { option: 'sogQuatsMax', formats: ['sog'] },
    'cov.k': { option: 'sogLeCodebookSize', formats: ['sog-le-codebook', 'sog-le-hybrid'] },
    'cov.max': { option: 'sogLeMax', formats: ['sog-le-minmax', 'sog-le-hybrid'] },
    'sh.iterations': { option: 'iterations', formats: null },
    'webp.effort': { option: 'webpEffort', formats: null }
};

const FORMATS: SweepFormat[] = ['sog', 'sog-le-codebook', 'sog-le-minmax', 'sog-le-hybrid'];

// size.json groups the encoded files by the attribute they store.
const FILE_GROUPS: [string, RegExp][] = [
    ['means', /^means_/],
    ['shape', /^(quats|scales|cov_\d)\.webp$/],
    ['sh0', /^sh0\.webp$/],
    ['shN', /^shN_/],
    ['meta', /^meta\.json$/]
];

const validateManifest = (manifest: SweepManifest): void => {
    if (!manifest || !Array.isArray(manifest.groups) || manifest.groups.length === 0) {
        throw new Error('sweep manifest: expected a non-empty `groups` list');
    }
    const ids = new Set<string>();
    for (const group of manifest.groups) {
        if (!group.id || ids.has(group.id)) throw new Error(`sweep manifest: missing or duplicate group id '${group.id}'`);
        ids.add(group.id);
        if (!FORMATS.includes(group.output_format)) {
            throw new Error(`sweep manifest: group '${group.id}' output_format must be one of ${FORMATS.join(', ')}`);
        }
        const mode = group.sweep_mode ?? 'product';
        if (mode !== 'product' && mode !== 'zip') throw new Error(`sweep manifest: group '${group.id}' sweep_mode must be product or zip`);
        const keys = [...Object.keys(group.params ?? {}), ...(group.axes ?? []).map(a => a.key)];
        for (const key of keys) {
            const spec = SWEEP_KEYS[key];
            if (!spec) throw new Error(`sweep manifest: unknown key '${key}' (known: ${Object.keys(SWEEP_KEYS).join(', ')})`);
            if (spec.formats && !spec.formats.includes(group.output_format)) {
                throw new Error(`sweep manifest: key '${key}' does not apply to ${group.output_format} (group '${group.id}')`);
            }
        }
        for (const axis of group.axes ?? []) {
            if (!Array.isArray(axis.values) || axis.values.length === 0 || !axis.values.every(Number.isFinite)) {
                throw new Error(`sweep manifest: axis '${axis.key}' in group '${group.id}' needs numeric values`);
            }
        }
        if (mode === 'zip') {
            const lengths = new Set((group.axes ?? []).map(a => a.values.length));
            if (lengths.size > 1) throw new Error(`sweep manifest: zip axes in group '${group.id}' differ in length`);
        }
    }
};

const combinations = (group: SweepGroup): Record<string, number>[] => {
    const axes = group.axes ?? [];
    if (axes.length === 0) return [{}];
    if ((group.sweep_mode ?? 'product') === 'zip') {
        return axes[0].values.map((_, i) => Object.fromEntries(axes.map(a => [a.key, a.values[i]])));
    }
    let combos: Record<string, number>[] = [{}];
    for (const axis of axes) {
        combos = combos.flatMap(c => axis.values.map(v => ({ ...c, [axis.key]: v })));
    }
    return combos;
};

const variantLabel = (values: Record<string, number>): string => {
    const parts = Object.entries(values).map(([k, v]) => `${k}=${v}`);
    return parts.length ? parts.join('__') : 'default';
};

const dirSize = async (dir: string): Promise<Record<string, number>> => {
    const files: Record<string, number> = {};
    for (const name of (await readdir(dir)).sort()) {
        files[name] = (await stat(join(dir, name))).size;
    }
    return files;
};

const csvCell = (v: unknown): string => {
    const s = v === undefined || v === null ? '' : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

const writeCsv = async (path: string, rows: Record<string, unknown>[]): Promise<void> => {
    if (rows.length === 0) return;
    const header: string[] = [];
    for (const row of rows) for (const k of Object.keys(row)) if (!header.includes(k)) header.push(k);
    const lines = [header.join(','), ...rows.map(r => header.map(h => csvCell(r[h])).join(','))];
    await writeFile(path, `${lines.join('\n')}\n`);
};

type SizeRecord = {
    group_id: string;
    variant: string;
    output_format: SweepFormat;
    params: Record<string, number>;
    num_gaussians: number;
    encoded_bytes: number;
    groups: Record<string, number>;
    files: Record<string, number>;
    encode_seconds: number;
    decode_seconds: number | null;
    encoded_dir: string;
    decoded_ply: string | null;
};

const sizeRow = (r: SizeRecord, input: { bytes: number }): Record<string, unknown> => ({
    group_id: r.group_id,
    variant: r.variant,
    output_format: r.output_format,
    ...r.params,
    num_gaussians: r.num_gaussians,
    encoded_bytes: r.encoded_bytes,
    encoded_mb: (r.encoded_bytes / 1e6).toFixed(6),
    bits_per_gaussian: (r.encoded_bytes * 8 / r.num_gaussians).toFixed(4),
    compression_ratio: (input.bytes / r.encoded_bytes).toFixed(4),
    ...Object.fromEntries(Object.entries(r.groups).map(([k, v]) => [`${k}_bytes`, v])),
    encode_seconds: r.encode_seconds.toFixed(3),
    decode_seconds: r.decode_seconds === null ? '' : r.decode_seconds.toFixed(3),
    encoded_dir: r.encoded_dir,
    decoded_ply: r.decoded_ply ?? ''
});

/**
 * Run a SOG / SOG-LE parameter sweep (see the file header for the layout).
 *
 * @param manifest - Sweep groups.
 * @param options - Input, output root and encoder settings.
 * @ignore
 */
const batchSweep = async (manifest: SweepManifest, options: BatchSweepOptions): Promise<void> => {
    validateManifest(manifest);
    const log = (msg: string) => {
        if (!options.quiet) console.log(msg);
    };

    if (options.maxWorkers !== undefined) WorkerQueue.maxWorkers = options.maxWorkers;

    const input = resolve(options.input);
    const output = resolve(options.output);
    await mkdir(output, { recursive: true });

    let cachedDevice: GraphicsDevice | undefined;
    const gpu = options.gpu ?? '-1';
    const deviceCreator = gpu === 'cpu' ? undefined : async () => {
        if (!cachedDevice) {
            let adapterName: string | undefined;
            const idx = Number.parseInt(gpu, 10);
            if (idx >= 0) adapterName = (await enumerateAdapters(options.gpuBackend))[idx]?.name;
            cachedDevice = await createDevice(adapterName, options.gpuBackend);
        }
        return cachedDevice;
    };

    const readFs = new NodeReadFileSystem();
    const fs = new NodeFileSystem();
    const pool = createChunkDataPool();
    const [source] = await readFile({ filename: input, inputFormat: getInputFormat(input), fileSystem: readFs });
    const inputInfo = { bytes: (await stat(input)).size };
    const numGaussians = source.meta.numGaussians;
    log(`Loaded ${input} (${(inputInfo.bytes / 1e6).toFixed(2)} MB, ${numGaussians} gaussians)`);

    // shared across every variant of every group
    const cache = new Map<string, Promise<unknown>>();
    const baseOptions: Options = {
        iterations: options.iterations ?? 10,
        webpEffort: options.webpEffort,
        sogEncodeCache: cache
    };

    const allRows: Record<string, unknown>[] = [{
        group_id: 'input',
        variant: 'uncompressed',
        output_format: 'ply',
        num_gaussians: numGaussians,
        encoded_bytes: inputInfo.bytes,
        encoded_mb: (inputInfo.bytes / 1e6).toFixed(6),
        bits_per_gaussian: (inputInfo.bytes * 8 / numGaussians).toFixed(4),
        compression_ratio: '1.0000',
        decoded_ply: input
    }];

    try {
        for (const group of manifest.groups) {
            const combos = combinations(group);
            const groupDir = join(output, group.id);
            await mkdir(groupDir, { recursive: true });
            const groupRows: Record<string, unknown>[] = [];
            log(`\n[${group.id}] ${group.output_format}, ${combos.length} variant(s)`);

            for (let i = 0; i < combos.length; i++) {
                const values = combos[i];
                const params = { ...(group.params ?? {}), ...values };
                const label = variantLabel(values);
                const variantDir = join(groupDir, label);
                const sizePath = join(variantDir, 'size.json');

                let record: SizeRecord | null = null;
                if (!options.overwrite) {
                    try {
                        record = JSON.parse(await fsReadFile(sizePath, 'utf-8'));
                        log(`  ${i + 1}/${combos.length} ${label}: exists, skipped`);
                    } catch {
                        record = null;
                    }
                }

                if (!record) {
                    await rm(variantDir, { recursive: true, force: true });
                    const encodedDir = join(variantDir, 'encoded');
                    await mkdir(encodedDir, { recursive: true });

                    const variantOptions: Options = { ...baseOptions };
                    for (const [key, value] of Object.entries(params)) {
                        (variantOptions as Record<string, unknown>)[SWEEP_KEYS[key].option] = value;
                    }
                    if (group.output_format !== 'sog') {
                        variantOptions.sogLe = group.output_format.slice('sog-le-'.length) as Options['sogLe'];
                    }

                    const metaPath = join(encodedDir, 'meta.json');
                    const t0 = performance.now();
                    await writeSource({
                        filename: metaPath,
                        outputFormat: 'sog',
                        source,
                        pool,
                        options: variantOptions,
                        createDevice: deviceCreator
                    }, fs);
                    const encodeSeconds = (performance.now() - t0) / 1000;

                    let decodeSeconds: number | null = null;
                    let decodedPly: string | null = null;
                    if (!options.noDecode) {
                        const t1 = performance.now();
                        decodedPly = join(variantDir, 'decoded.ply');
                        const [decoded]: ChunkSource[] = await readFile({ filename: metaPath, inputFormat: 'sog', fileSystem: readFs });
                        try {
                            await writeSource({ filename: decodedPly, outputFormat: 'ply', source: decoded, pool, options: {} }, fs);
                        } finally {
                            await decoded.close();
                        }
                        decodeSeconds = (performance.now() - t1) / 1000;
                    }

                    const files = await dirSize(encodedDir);
                    const groups: Record<string, number> = {};
                    for (const [name, bytes] of Object.entries(files)) {
                        const g = FILE_GROUPS.find(([, re]) => re.test(name))?.[0] ?? 'other';
                        groups[g] = (groups[g] ?? 0) + bytes;
                    }
                    record = {
                        group_id: group.id,
                        variant: label,
                        output_format: group.output_format,
                        params,
                        num_gaussians: numGaussians,
                        encoded_bytes: Object.values(files).reduce((a, b) => a + b, 0),
                        groups,
                        files,
                        encode_seconds: encodeSeconds,
                        decode_seconds: decodeSeconds,
                        encoded_dir: encodedDir,
                        decoded_ply: decodedPly
                    };
                    await writeFile(sizePath, `${JSON.stringify(record, null, 2)}\n`);
                    const decodeNote = decodeSeconds !== null ? `, decode ${decodeSeconds.toFixed(1)}s` : '';
                    log(`  ${i + 1}/${combos.length} ${label}: ${(record.encoded_bytes / 1e6).toFixed(3)} MB ` +
                        `(shape ${((groups.shape ?? 0) / 1e6).toFixed(3)} MB), encode ${encodeSeconds.toFixed(1)}s${decodeNote}`);
                }

                groupRows.push(sizeRow(record, inputInfo));
            }

            await writeCsv(join(groupDir, 'sweep_metrics.csv'), groupRows);
            allRows.push(...groupRows);
        }
        await writeCsv(join(output, 'batch_sweep_metrics.csv'), allRows);
        log(`\nWrote ${join(output, 'batch_sweep_metrics.csv')}`);
    } finally {
        await source.close();
    }
};

export { batchSweep };
export type { BatchSweepOptions, SweepManifest };

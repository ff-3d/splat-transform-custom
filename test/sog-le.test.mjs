/**
 * Log-Euclidean SOG (.sog-le-*) tests: the LE chart math, the writer/reader
 * round trip for every quantization mode, and the standard SOG rate knobs.
 */

import { describe, it, before } from 'node:test';
import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
    Column,
    DataTable,
    Transform,
    createChunkDataPool,
    dataTableToChunkSource,
    decodeLogCovariance,
    encodeLogCovariance,
    getOutputFormat,
    materializeToDataTable,
    readFile,
    sortMortonInterleaved,
    writeSource,
    MemoryFileSystem,
    MemoryReadFileSystem,
    WebPCodec
} from '../src/lib/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');

// Deterministic pseudo-random generator (mulberry32).
const rng = (seed) => () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const covariance = (w, x, y, z, l0, l1, l2) => {
    const n = Math.hypot(w, x, y, z);
    w /= n; x /= n; y /= n; z /= n;
    const R = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ];
    const d = [l0, l1, l2].map(l => Math.exp(2 * l));
    const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[i][j] += R[i][k] * d[k] * R[j][k];
    return C;
};

const createRandomScene = (count) => {
    const rand = rng(7);
    const names = ['x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
    const cols = Object.fromEntries(names.map(n => [n, new Float32Array(count)]));
    for (let i = 0; i < count; i++) {
        cols.x[i] = rand() * 10 - 5;
        cols.y[i] = rand() * 10 - 5;
        cols.z[i] = rand() * 10 - 5;
        const base = -6 + rand() * 4;
        // half nearly round disks (two close log-scales), half generic
        cols.scale_0[i] = base;
        cols.scale_1[i] = i % 2 ? base + 0.01 * rand() : -6 + rand() * 4;
        cols.scale_2[i] = -6 + rand() * 4;
        for (const c of ['f_dc_0', 'f_dc_1', 'f_dc_2']) cols[c][i] = rand() * 2 - 1;
        cols.opacity[i] = rand() * 4 - 2;
        for (const c of ['rot_0', 'rot_1', 'rot_2', 'rot_3']) cols[c][i] = rand() * 2 - 1;
    }
    const table = new DataTable(names.map(n => new Column(n, cols[n])));
    table.transform = Transform.PLY.clone();
    return table;
};

const encodeToMemory = async (table, filename, options = {}) => {
    const pool = createChunkDataPool();
    const fs = new MemoryFileSystem();
    await writeSource({
        filename,
        outputFormat: getOutputFormat(filename, {}),
        source: dataTableToChunkSource(table, pool.chunkSize),
        pool,
        options: { iterations: 2, ...options }
    }, fs);
    const readFs = new MemoryReadFileSystem();
    for (const [name, data] of fs.results.entries()) {
        readFs.set(name.includes('/') ? name.split('/').pop() : name, data);
    }
    return readFs;
};

const decode = async (readFs, filename) => {
    const [source] = await readFile({ filename, inputFormat: 'sog', fileSystem: readFs });
    return materializeToDataTable(source, createChunkDataPool());
};

const readMeta = async (readFs, filename) => {
    if (filename === 'meta.json') return JSON.parse(new TextDecoder().decode(readFs.get('meta.json')));
    return null;
};

// Per-gaussian Frobenius error of log Σ between the input and a decoded
// (Morton-ordered) table.
const leErrors = (input, decoded) => {
    const n = input.numRows;
    const pos = new Float32Array(n * 3);
    const [x, y, z] = ['x', 'y', 'z'].map(c => input.getColumnByName(c).data);
    for (let i = 0; i < n; i++) {
        pos[i * 3] = x[i]; pos[i * 3 + 1] = y[i]; pos[i * 3 + 2] = z[i];
    }
    const order = new Uint32Array(n).map((_, i) => i);
    sortMortonInterleaved(pos, order);
    const geo = t => ['rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2'].map(c => t.getColumnByName(c).data);
    const a = geo(input);
    const b = geo(decoded);
    const ea = new Float64Array(6);
    const eb = new Float64Array(6);
    const errs = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const g = order[i];
        encodeLogCovariance(a[0][g], a[1][g], a[2][g], a[3][g], a[4][g], a[5][g], a[6][g], ea);
        encodeLogCovariance(b[0][i], b[1][i], b[2][i], b[3][i], b[4][i], b[5][i], b[6][i], eb);
        let s = 0;
        for (let c = 0; c < 6; c++) s += (ea[c] - eb[c]) ** 2;
        errs[i] = Math.sqrt(s);
    }
    return errs;
};

describe('Log-Euclidean chart', () => {
    it('round-trips the covariance exactly (incl. nearly round disks)', () => {
        const rand = rng(1);
        const v = new Float64Array(6);
        const out = new Float64Array(7);
        for (let t = 0; t < 5000; t++) {
            const q = [rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1];
            const base = -8 + rand() * 6;
            const l = t % 2 ? [base, base + 1e-5 * rand(), -8 + rand() * 6] : [base, -8 + rand() * 6, -8 + rand() * 6];
            encodeLogCovariance(...q, ...l, v);
            decodeLogCovariance(v, 0, out);
            const A = covariance(...q, ...l);
            const B = covariance(...out);
            let err = 0, norm = 0;
            for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
                err += (A[i][j] - B[i][j]) ** 2;
                norm += A[i][j] ** 2;
            }
            assert(Math.sqrt(err / norm) < 1e-9, `relative error ${Math.sqrt(err / norm)}`);
            assert(out[4] <= out[5] && out[5] <= out[6], 'scales ascending (eigh order)');
            assert(out[0] >= 0, 'canonical w >= 0');
        }
    });

    it('decodes exactly equal diagonal entries (codebook-snapped values)', () => {
        // xx = yy with xy != 0: eigenvalues -10 ± 2 (Jacobi tau = 0 case)
        const out = new Float64Array(7);
        decodeLogCovariance([-10, 2 * Math.SQRT2, 0, -10, 0, -5], 0, out);
        assert(Math.abs(out[4] - (-12 / 2)) < 1e-9, `log s0 ${out[4]}`);
        assert(Math.abs(out[5] - (-8 / 2)) < 1e-9, `log s1 ${out[5]}`);
        assert(Math.abs(out[6] - (-5 / 2)) < 1e-9, `log s2 ${out[6]}`);
    });

    it('preserves the Frobenius norm via the sqrt(2) off-diagonal scaling', () => {
        const v = new Float64Array(6);
        encodeLogCovariance(0.9, 0.3, -0.2, 0.1, -1, -2, -3, v);
        const norm2 = v.reduce((s, x) => s + x * x, 0);
        // ||log Σ||_F^2 = Σ (2 l_i)^2
        assert(Math.abs(norm2 - (4 + 16 + 36)) < 1e-9);
    });

    it('applies the scale floors (encode 1e-10, decode 1e-10)', () => {
        const v = new Float64Array(6);
        encodeLogCovariance(1, 0, 0, 0, -200, -1, -2, v);
        assert(Math.abs(v[0] - 2 * Math.log(1e-10)) < 1e-9);
        const out = new Float64Array(7);
        decodeLogCovariance([-100, 0, 0, -1, 0, -2], 0, out);
        assert(Math.abs(out[4] - Math.log(1e-10)) < 1e-9);
    });
});

describe('SOG-LE writer/reader', () => {
    let scene;

    before(() => {
        scene = createRandomScene(4096);
    });

    for (const encoding of ['codebook', 'minmax', 'hybrid']) {
        it(`round-trips .sog-le-${encoding} (bundled) with small shape error`, async () => {
            const filename = `test.sog-le-${encoding}`;
            const readFs = await encodeToMemory(scene, filename);
            const decoded = await decode(readFs, filename);
            assert.strictEqual(decoded.numRows, scene.numRows);
            const errs = leErrors(scene, decoded);
            const mean = errs.reduce((a, b) => a + b, 0) / errs.length;
            assert(mean < 0.15, `mean LE error ${mean}`);
        });
    }

    it('writes the covariances meta entry instead of quats/scales', async () => {
        const readFs = await encodeToMemory(scene, 'meta.json', { sogLe: 'hybrid', sogLeCodebookSize: 64, sogLeMax: 31 });
        const meta = await readMeta(readFs, 'meta.json');
        assert.strictEqual(meta.quats, undefined);
        assert.strictEqual(meta.scales, undefined);
        const cov = meta.covariances;
        assert.strictEqual(cov.parameterization, 'log-euclidean');
        assert.deepStrictEqual(cov.channels, ['codebook', 'minmax', 'minmax', 'codebook', 'minmax', 'codebook']);
        assert.strictEqual(cov.codebook.length, 64);
        assert.strictEqual(cov.max, 31);
        assert.strictEqual(cov.mins.length, 3);
        assert.deepStrictEqual(cov.files, ['cov_0.webp', 'cov_1.webp']);
        const decoded = await decode(readFs, 'meta.json');
        const errs = leErrors(scene, decoded);
        assert(errs.reduce((a, b) => a + b, 0) / errs.length < 0.6);
    });

    it('lower rates give larger shape error', async () => {
        const meanErr = async (opts) => {
            const readFs = await encodeToMemory(scene, 'meta.json', opts);
            const errs = leErrors(scene, await decode(readFs, 'meta.json'));
            return errs.reduce((a, b) => a + b, 0) / errs.length;
        };
        assert(await meanErr({ sogLe: 'minmax', sogLeMax: 15 }) > await meanErr({ sogLe: 'minmax', sogLeMax: 255 }));
        assert(await meanErr({ sogLe: 'codebook', sogLeCodebookSize: 16 }) > await meanErr({ sogLe: 'codebook', sogLeCodebookSize: 256 }));
    });
});

describe('SOG rate knobs', () => {
    let scene;

    before(() => {
        scene = createRandomScene(2048);
    });

    it('keeps the deployed meta layout by default', async () => {
        const meta = await readMeta(await encodeToMemory(scene, 'meta.json'), 'meta.json');
        assert.deepStrictEqual(meta.quats, { files: ['quats.webp'] });
        assert.strictEqual(meta.scales.codebook.length, 256);
        assert.strictEqual(meta.covariances, undefined);
    });

    it('decodes reduced quaternion levels and scale codebooks', async () => {
        const readFs = await encodeToMemory(scene, 'meta.json', { sogQuatsMax: 31, sogScalesCodebookSize: 32 });
        const meta = await readMeta(readFs, 'meta.json');
        assert.strictEqual(meta.quats.max, 31);
        assert.strictEqual(meta.scales.codebook.length, 32);
        const decoded = await decode(readFs, 'meta.json');
        const errs = leErrors(scene, decoded);
        const sorted = Float64Array.from(errs).sort();
        assert(sorted[sorted.length >> 1] < 1.0, `median error ${sorted[sorted.length >> 1]}`);
    });
});

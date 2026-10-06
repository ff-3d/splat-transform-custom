import { type ChunkDataPool, type ChunkSource } from '../chunk';
import { type FileSystem } from '../io/write';
import { encodeLogCovariance, LE_DIAGONAL } from '../utils';
import { runQuantize1dColumns } from '../workers';
import { type SogShapeEncoder, type WriteSogSourceOptions, writeSogSource } from './write-sog';

/**
 * How the six Log-Euclidean coordinates are quantized to 8 bits.
 *
 * - `codebook` - one shared 1D codebook over all six (the SOG scale quantizer
 *   applied to the whole LE vector; with the sqrt(2) off-diagonal scaling its
 *   squared error is the Frobenius error of log Σ).
 * - `minmax` - per-channel uniform min-max (ffsplat `simple_quantize`, as in
 *   SOG-XT-1510-logmap).
 * - `hybrid` - shared codebook on the diagonal, min-max on the off-diagonals.
 */
type SogLeEncoding = 'codebook' | 'minmax' | 'hybrid';

type SogLeOptions = {
    encoding: SogLeEncoding;
    /** Codebook entries (1-256) for `codebook` / `hybrid`. Default 256. */
    codebookSize?: number;
    /** Highest min-max level (1-255) for `minmax` / `hybrid`; levels = max + 1. Default 255. */
    max?: number;
};

const LE_COMPONENTS = ['xx', 'xy', 'xz', 'yy', 'yz', 'zz'];

/**
 * Shape encoder replacing SOG's quats + scales textures with the Log-Euclidean
 * chart (see `log-euclidean.ts`): `cov_0.webp` holds components [xx, xy, xz],
 * `cov_1.webp` [yy, yz, zz] (the ffsplat 6D -> 2x3D split).
 *
 * @param le - LE quantization options.
 * @returns The shape encoder.
 * @ignore
 */
const leShapeEncoder = (le: SogLeOptions): SogShapeEncoder => (ctx) => {
    const { encoding } = le;
    const codebookSize = le.codebookSize ?? 256;
    const max = le.max ?? 255;
    if (!Number.isInteger(codebookSize) || codebookSize < 1 || codebookSize > 256) {
        throw new Error(`sog-le: codebook size must be an integer in [1, 256], got ${codebookSize}`);
    }
    if (!Number.isInteger(max) || max < 1 || max > 255) {
        throw new Error(`sog-le: max must be an integer in [1, 255], got ${max}`);
    }

    const n = ctx.numRows;
    const [r0, r1, r2, r3, s0, s1, s2] = ctx.geometric;
    const cols = LE_COMPONENTS.map(() => new Float32Array(n));
    const tmp = new Float64Array(6);
    for (let g = 0; g < n; g++) {
        encodeLogCovariance(r0[g], r1[g], r2[g], r3[g], s0[g], s1[g], s2[g], tmp);
        for (let c = 0; c < 6; c++) cols[c][g] = tmp[c];
    }

    const kinds: ('codebook' | 'minmax')[] = LE_COMPONENTS.map((_, c) => {
        if (encoding === 'codebook') return 'codebook';
        if (encoding === 'minmax') return 'minmax';
        return LE_DIAGONAL.includes(c) ? 'codebook' : 'minmax';
    });
    const cbChannels = kinds.flatMap((k, c) => (k === 'codebook' ? [c] : []));
    const mmChannels = kinds.flatMap((k, c) => (k === 'minmax' ? [c] : []));

    const cbQuant = cbChannels.length > 0 ?
        runQuantize1dColumns(cbChannels.map(c => ({ name: LE_COMPONENTS[c], data: cols[c] })), codebookSize) :
        null;
    cbQuant?.catch(() => {});

    return async (indices) => {
        const labels: Uint8Array[] = new Array(6);

        // min-max channels: ffsplat simple_quantize (range clamped to >= 1e-8, rounded)
        const mins: number[] = [];
        const maxs: number[] = [];
        for (const c of mmChannels) {
            const col = cols[c];
            let lo = Infinity, hi = -Infinity;
            for (let g = 0; g < n; g++) {
                const v = col[g];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
            if (n === 0) {
                lo = 0; hi = 0;
            }
            const range = Math.max(hi - lo, 1e-8);
            const lab = new Uint8Array(n);
            for (let g = 0; g < n; g++) {
                lab[g] = Math.min(max, Math.max(0, Math.round((col[g] - lo) / range * max)));
            }
            labels[c] = lab;
            mins.push(lo);
            maxs.push(hi);
        }

        let codebook: number[] | null = null;
        if (cbQuant) {
            const q = await cbQuant;
            cbChannels.forEach((c, i) => {
                labels[c] = q.labels[i].data;
            });
            codebook = Array.from(q.centroids);
        }

        ctx.pending.push(
            ctx.writeLabels('cov_0.webp', [labels[0], labels[1], labels[2]], indices),
            ctx.writeLabels('cov_1.webp', [labels[3], labels[4], labels[5]], indices)
        );

        return {
            covariances: {
                parameterization: 'log-euclidean',
                encoding,
                components: LE_COMPONENTS,
                channels: kinds,
                ...(codebook ? { codebook } : {}),
                ...(mmChannels.length > 0 ? { max, mins, maxs } : {}),
                files: ['cov_0.webp', 'cov_1.webp']
            }
        };
    };
};

/**
 * SOG writer with the Log-Euclidean shape chart (`.sog-le-*`). Identical to
 * {@link writeSogSource} — same Morton order, means, sh0/opacity and shN
 * coding — except that `quats.webp` + `scales.webp` are replaced by the two
 * LE covariance textures described by `meta.json`'s `covariances` entry.
 *
 * @param source - The source to encode (its pending transform is baked to PLY space).
 * @param pool - Pool for the temporary per-chunk read buffers.
 * @param options - SOG output options plus the LE quantization options.
 * @param fs - File system to write through.
 * @ignore
 */
const writeSogLeSource = (
    source: ChunkSource,
    pool: ChunkDataPool,
    options: WriteSogSourceOptions & { le: SogLeOptions },
    fs: FileSystem
): Promise<void> => {
    const { le, ...rest } = options;
    return writeSogSource(source, pool, { ...rest, shapeEncoder: leShapeEncoder(le) }, fs);
};

export { leShapeEncoder, writeSogLeSource };
export type { SogLeEncoding, SogLeOptions };

// Log-Euclidean (LE) shape chart for 3D Gaussians: the matrix logarithm of the
// covariance, stored as its 6-value upper triangle. Mirrors ffsplat's
// `build_covariance(apply_log) + symmetric_to_triu(preserve_norm)` encode and
// `triu_to_symmetric + decompose_covariance(apply_exp)` decode
// (SOG-XT-1510-logmap), including its clamping thresholds:
//
//   encode: log Σ = R diag(2 log max(s, 1e-20)) Rᵀ
//   decode: eigh(log Σ) -> s = max(exp(λ / 2), 1e-10), R = eigenvectors (det fixed to +1)
//
// Component order is torch.triu_indices(3, 3): [xx, xy, xz, yy, yz, zz]. The
// off-diagonals are scaled by sqrt(2) so the 6-vector's L2 norm equals the
// Frobenius norm of log Σ.

import { eigenSymmetric3x3, rotmatToQuat } from './sym3';

/** Smallest scale fed to the log on encode (ffsplat `clamp_min(1e-20)`). */
const LE_MIN_ENCODE_SCALE = 1e-20;
/** Smallest scale produced on decode (ffsplat `clamp(min=1e-10)`). */
const LE_MIN_DECODE_SCALE = 1e-10;

const LN_MIN_ENCODE_SCALE = Math.log(LE_MIN_ENCODE_SCALE);
const LN_MIN_DECODE_SCALE = Math.log(LE_MIN_DECODE_SCALE);
const SQRT2 = Math.sqrt(2);

/** Triu component index of each diagonal entry (xx, yy, zz). */
const LE_DIAGONAL = [0, 3, 5];
/** Triu component index of each off-diagonal entry (xy, xz, yz). */
const LE_OFF_DIAGONAL = [1, 2, 4];

/**
 * Encode a (PLY-space) rotation + log-scale into the 6-value LE vector.
 *
 * @param w - Quaternion w (rot_0). Need not be normalized.
 * @param x - Quaternion x (rot_1).
 * @param y - Quaternion y (rot_2).
 * @param z - Quaternion z (rot_3).
 * @param l0 - Log scale 0 (scale_0).
 * @param l1 - Log scale 1 (scale_1).
 * @param l2 - Log scale 2 (scale_2).
 * @param out - Receives [xx, √2·xy, √2·xz, yy, √2·yz, zz] at `offset`.
 * @param offset - Write offset into `out`.
 * @ignore
 */
const encodeLogCovariance = (
    w: number, x: number, y: number, z: number,
    l0: number, l1: number, l2: number,
    out: Float64Array | Float32Array, offset = 0
): void => {
    const n = Math.sqrt(w * w + x * x + y * y + z * z);
    if (n > 0) {
        w /= n; x /= n; y /= n; z /= n;
    } else {
        w = 1; x = 0; y = 0; z = 0;
    }

    // rotation matrix (row-major) of the unit quaternion
    const r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - w * z), r02 = 2 * (x * z + w * y);
    const r10 = 2 * (x * y + w * z), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - w * x);
    const r20 = 2 * (x * z - w * y), r21 = 2 * (y * z + w * x), r22 = 1 - 2 * (x * x + y * y);

    // 2 log max(s, 1e-20), with s = exp(l)
    const d0 = 2 * Math.max(l0, LN_MIN_ENCODE_SCALE);
    const d1 = 2 * Math.max(l1, LN_MIN_ENCODE_SCALE);
    const d2 = 2 * Math.max(l2, LN_MIN_ENCODE_SCALE);

    // L = R diag(d) Rᵀ
    out[offset]     = r00 * r00 * d0 + r01 * r01 * d1 + r02 * r02 * d2;
    out[offset + 1] = (r00 * r10 * d0 + r01 * r11 * d1 + r02 * r12 * d2) * SQRT2;
    out[offset + 2] = (r00 * r20 * d0 + r01 * r21 * d1 + r02 * r22 * d2) * SQRT2;
    out[offset + 3] = r10 * r10 * d0 + r11 * r11 * d1 + r12 * r12 * d2;
    out[offset + 4] = (r10 * r20 * d0 + r11 * r21 * d1 + r12 * r22 * d2) * SQRT2;
    out[offset + 5] = r20 * r20 * d0 + r21 * r21 * d1 + r22 * r22 * d2;
};

// Decode scratch: symmetric input, Jacobi output (eigenvalues on the
// diagonal, eigenvectors in the columns), sorted rotation, quaternion.
// Single-threaded decode, so module-level is safe.
const symIn = new Float64Array(9);
const symA = new Float64Array(9);
const symV = new Float64Array(9);
const rot = new Float64Array(9);
const quat = new Float64Array(4);

/**
 * Decode a 6-value LE vector into a unit quaternion and log-scales, following
 * ffsplat's `decompose_covariance(apply_exp)`: eigenvalues ascending, the
 * first eigenvector flipped when the eigenbasis is left-handed, and scales
 * clamped to at least 1e-10.
 *
 * @param v - [xx, √2·xy, √2·xz, yy, √2·yz, zz] at `offset`.
 * @param offset - Read offset into `v`.
 * @param out - Receives [w, x, y, z, log s0, log s1, log s2] at `outOffset`.
 * @param outOffset - Write offset into `out`.
 * @ignore
 */
const decodeLogCovariance = (
    v: ArrayLike<number>, offset: number,
    out: Float32Array | Float64Array, outOffset = 0
): void => {
    const xy = v[offset + 1] / SQRT2;
    const xz = v[offset + 2] / SQRT2;
    const yz = v[offset + 4] / SQRT2;
    symIn[0] = v[offset]; symIn[1] = xy; symIn[2] = xz;
    symIn[3] = xy; symIn[4] = v[offset + 3]; symIn[5] = yz;
    symIn[6] = xz; symIn[7] = yz; symIn[8] = v[offset + 5];
    eigenSymmetric3x3(symIn, symA, symV);

    // sort eigenpairs ascending (torch.linalg.eigh order)
    const e = [symA[0], symA[4], symA[8]];
    const order = [0, 1, 2].sort((a, b) => e[a] - e[b]);
    const c0 = order[0], c1 = order[1], c2 = order[2];

    // R columns = sorted eigenvectors
    for (let r = 0; r < 3; r++) {
        rot[r * 3] = symV[r * 3 + c0];
        rot[r * 3 + 1] = symV[r * 3 + c1];
        rot[r * 3 + 2] = symV[r * 3 + c2];
    }

    // fix right-handedness: flip the first eigenvector when det < 0
    const det = rot[0] * (rot[4] * rot[8] - rot[5] * rot[7]) -
        rot[1] * (rot[3] * rot[8] - rot[5] * rot[6]) +
        rot[2] * (rot[3] * rot[7] - rot[4] * rot[6]);
    if (det < 0) {
        rot[0] = -rot[0]; rot[3] = -rot[3]; rot[6] = -rot[6];
    }

    // rotation matrix -> unit quaternion, canonicalized to w >= 0
    rotmatToQuat(rot, 0, quat, 0);
    const sign = quat[0] < 0 ? -1 : 1;
    out[outOffset]     = sign * quat[0];
    out[outOffset + 1] = sign * quat[1];
    out[outOffset + 2] = sign * quat[2];
    out[outOffset + 3] = sign * quat[3];

    // s = max(exp(λ / 2), 1e-10)  =>  log s = max(λ / 2, ln 1e-10)
    out[outOffset + 4] = Math.max(0.5 * e[c0], LN_MIN_DECODE_SCALE);
    out[outOffset + 5] = Math.max(0.5 * e[c1], LN_MIN_DECODE_SCALE);
    out[outOffset + 6] = Math.max(0.5 * e[c2], LN_MIN_DECODE_SCALE);
};

export {
    encodeLogCovariance,
    decodeLogCovariance,
    LE_DIAGONAL,
    LE_OFF_DIAGONAL,
    LE_MIN_ENCODE_SCALE,
    LE_MIN_DECODE_SCALE
};

// Symmetric 3x3 helpers shared by decimation (moment matching) and the
// Log-Euclidean SOG decoder. Engine-free: imported by worker tasks, so it must
// not import DataTable or playcanvas (see workers/tasks.ts).

// Jacobi eigendecomposition for 3x3 symmetric matrix; caller-provided scratch,
// eigenvalues land on A's diagonal, eigenvectors in V's columns.
const eigenSymmetric3x3 = (Ain: Float64Array, A: Float64Array, V: Float64Array) => {
    A.set(Ain);
    V[0] = 1; V[1] = 0; V[2] = 0;
    V[3] = 0; V[4] = 1; V[5] = 0;
    V[6] = 0; V[7] = 0; V[8] = 1;

    for (let iter = 0; iter < 24; iter++) {
        let p = 0, q = 1;
        let maxAbs = Math.abs(A[1]);
        if (Math.abs(A[2]) > maxAbs) {
            p = 0; q = 2; maxAbs = Math.abs(A[2]);
        }
        if (Math.abs(A[5]) > maxAbs) {
            p = 1; q = 2; maxAbs = Math.abs(A[5]);
        }
        if (maxAbs < 1e-12) break;

        const pp = 3 * p + p, qq = 3 * q + q, pq = 3 * p + q;
        const app = A[pp], aqq = A[qq], apq = A[pq];
        const tau = (aqq - app) / (2 * apq);
        // tau = 0 (equal diagonal entries) needs a 45° rotation; Math.sign(0)
        // would give t = 0, zeroing A[pq] without rotating (wrong eigenpairs)
        const t = (tau < 0 ? -1 : 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        for (let k = 0; k < 3; k++) {
            if (k === p || k === q) continue;
            const kp = 3 * k + p, kq = 3 * k + q;
            const pk = 3 * p + k, qk = 3 * q + k;
            const akp = A[kp], akq = A[kq];
            A[kp] = c * akp - s * akq;
            A[pk] = A[kp];
            A[kq] = s * akp + c * akq;
            A[qk] = A[kq];
        }
        A[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
        A[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
        A[pq] = 0; A[3 * q + p] = 0;

        for (let k = 0; k < 3; k++) {
            const kp = 3 * k + p, kq = 3 * k + q;
            const vkp = V[kp], vkq = V[kq];
            V[kp] = c * vkp - s * vkq;
            V[kq] = s * vkp + c * vkq;
        }
    }
};

const rotmatToQuat = (R: Float64Array, o: number, out: Float64Array, oo: number) => {
    const m00 = R[o], m11 = R[o + 4], m22 = R[o + 8];
    const tr = m00 + m11 + m22;
    let qw: number, qx: number, qy: number, qz: number;

    if (tr > 0) {
        const S = Math.sqrt(tr + 1) * 2;
        qw = 0.25 * S;
        qx = (R[o + 7] - R[o + 5]) / S;
        qy = (R[o + 2] - R[o + 6]) / S;
        qz = (R[o + 3] - R[o + 1]) / S;
    } else if (R[o] > R[o + 4] && R[o] > R[o + 8]) {
        const S = Math.sqrt(1 + R[o] - R[o + 4] - R[o + 8]) * 2;
        qw = (R[o + 7] - R[o + 5]) / S;
        qx = 0.25 * S;
        qy = (R[o + 1] + R[o + 3]) / S;
        qz = (R[o + 2] + R[o + 6]) / S;
    } else if (R[o + 4] > R[o + 8]) {
        const S = Math.sqrt(1 + R[o + 4] - R[o] - R[o + 8]) * 2;
        qw = (R[o + 2] - R[o + 6]) / S;
        qx = (R[o + 1] + R[o + 3]) / S;
        qy = 0.25 * S;
        qz = (R[o + 5] + R[o + 7]) / S;
    } else {
        const S = Math.sqrt(1 + R[o + 8] - R[o] - R[o + 4]) * 2;
        qw = (R[o + 3] - R[o + 1]) / S;
        qx = (R[o + 2] + R[o + 6]) / S;
        qy = (R[o + 5] + R[o + 7]) / S;
        qz = 0.25 * S;
    }

    const n = Math.hypot(qw, qx, qy, qz);
    const inv = 1 / Math.max(n, 1e-12);
    out[oo] = qw * inv;
    out[oo + 1] = qx * inv;
    out[oo + 2] = qy * inv;
    out[oo + 3] = qz * inv;
};

export { eigenSymmetric3x3, rotmatToQuat };

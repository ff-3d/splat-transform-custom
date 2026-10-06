// Fast non-cryptographic 64-bit content hash (two independent 32-bit
// murmur3-style lanes) over a typed array's bytes. Used as a memo key for
// deterministic encode stages (WebP encodes, SH k-means) shared across a
// parameter sweep, so identical inputs are processed once.

const mix = (h: number, k: number, c1: number, c2: number): number => {
    k = Math.imul(k, c1);
    k = (k << 15) | (k >>> 17);
    k = Math.imul(k, c2);
    h ^= k;
    h = (h << 13) | (h >>> 19);
    return (Math.imul(h, 5) + 0xe6546b64) | 0;
};

const fmix = (h: number): number => {
    h ^= h >>> 16;
    h = Math.imul(h, 0x85ebca6b);
    h ^= h >>> 13;
    h = Math.imul(h, 0xc2b2ae35);
    h ^= h >>> 16;
    return h >>> 0;
};

/**
 * Hash the bytes of a typed array.
 *
 * @param view - The data to hash.
 * @returns A 16-hex-digit hash followed by the byte length.
 * @ignore
 */
const contentHash = (view: ArrayBufferView): string => {
    const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
    const n4 = bytes.byteLength >>> 2;
    let h1 = 0x9747b28c;
    let h2 = 0x1b873593;
    if ((bytes.byteOffset & 3) === 0) {
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, n4);
        for (let i = 0; i < n4; i++) {
            const k = words[i];
            h1 = mix(h1, k, 0xcc9e2d51, 0x1b873593);
            h2 = mix(h2, k, 0x85ebca77, 0xc2b2ae3d);
        }
    } else {
        for (let i = 0; i < n4; i++) {
            const o = i * 4;
            const k = bytes[o] | (bytes[o + 1] << 8) | (bytes[o + 2] << 16) | (bytes[o + 3] << 24);
            h1 = mix(h1, k, 0xcc9e2d51, 0x1b873593);
            h2 = mix(h2, k, 0x85ebca77, 0xc2b2ae3d);
        }
    }
    let tail = 0;
    for (let i = n4 * 4; i < bytes.byteLength; i++) tail = (tail << 8) | bytes[i];
    h1 = fmix(mix(h1, tail, 0xcc9e2d51, 0x1b873593) ^ bytes.byteLength);
    h2 = fmix(mix(h2, tail, 0x85ebca77, 0xc2b2ae3d) ^ bytes.byteLength);
    return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}:${bytes.byteLength}`;
};

export { contentHash };

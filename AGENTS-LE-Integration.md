# Log-Euclidean (LE) integration into SOG

This is a working record of the `.sog-le-*` codec and the sweep tooling (2026-10-05/06, branch `sog-le`). Read it before continuing this work on another machine. Uncommitted work at the time of writing: see "Files" below.

## Purpose

The work is for a paper: "Towards an Encoding-Oriented Parameterization for 3D Gaussian Splatting". Its claim is that the Log-Euclidean shape chart, `vec(log Σ)` with 6 values, suits codecs better than `(q, log s)` with 7 values. splat-transform's `.sog` is codec **C2** in the paper. `.sog` vs `.sog-LE` is the **equal-value-count comparison**: std `.sog` already stores 6 shape values (smallest-three quaternion + 3 scale indices). That separates "one value fewer" from "better parameterization".

Reference implementation: ffsplat (`references/ffsplat-dev`, gitignored; clone it separately). The format to mimic is `src/ffsplat/conf/format/SOG-XT-1510-logmap.yaml`. The sweep model is `src/ffsplat/cli/batch_sweep.py`.

## Design decisions (agreed with the user)

1. **Keep everything else in `.sog` identical.** That covers Morton order (not PLAS), means (16-bit log split), sh0 + opacity, shN k-means palette, and WebP lossless. Only `quats.webp` + `scales.webp` are replaced by `cov_0.webp` + `cov_1.webp`.
2. **The LE math matches ffsplat exactly**, including its clamps:
   - encode: `log Σ = R diag(2·log max(s, 1e-20)) Rᵀ`; upper triangle in `torch.triu_indices(3,3)` order `[xx, xy, xz, yy, yz, zz]`; off-diagonals ×√2 (`preserve_norm`), so the L2 norm of the vector equals ‖log Σ‖_F.
   - decode: undo √2; symmetric eigendecomposition with eigenvalues **ascending** (eigh order); flip the **first** eigenvector if det < 0; `s = max(exp(λ/2), 1e-10)`; quaternion from the eigenvector matrix, canonicalized to w ≥ 0.
3. **Three quantization modes**, all 8-bit indices, in two RGB textures (`cov_0` = xx,xy,xz; `cov_1` = yy,yz,zz, the ffsplat 6D→2×3D split):
   - `codebook` (**primary paper row**): one shared 1D codebook (`quantize1dColumns`, the same quantizer `.sog` uses for scales, same α = 0.5) over all 6 coordinates. It has the same side info as `.sog` (one 256-entry codebook) and the same index bits. With the √2 scaling, its squared-error objective is the density-weighted LE distortion D_LE. Sharing one table is justified by the chart: a world rotation mixes the 6 LE coordinates orthogonally (R5), so they are commensurate.
   - `minmax` (conservative control): per-channel uniform min–max, i.e. ffsplat `simple_quantize` (range clamped ≥ 1e-8, rounding, values 0..max). It gives LE *less* machinery than `.sog` gives its scales, and it ties C2 to C1 (SOG-XT).
   - `hybrid` (supplement ablation): diagonal → shared codebook, off-diagonals → min–max. This mirrors `.sog`'s machinery role by role.
   - Rejected: two codebooks (doubles the side info; strictly dominates one codebook, so it is unfair to `.sog`), and 6D VQ (a different codec; that is C4).
4. **Rate knobs for plain `.sog`**: scale-codebook size `k` and quaternion levels. Reduced quaternion levels are stored **compactly** (0..max) with `quats.max` in meta.json (the user chose this over snapping to the 255 grid). The stock PlayCanvas viewer only understands the default `max` = 255. The deployed **truncation** (floor) of quaternion components is kept, not rounding.
5. **Size tracking belongs in the sweep entrypoint, not in ffsplat eval.** Only the encoder knows which files make up the encoding, and eval only sees `decoded.ply`. ffsplat eval should read sizes from `size.json` / the CSV and evaluate `decoded.ply`.
6. **Sweep outputs are unbundled directories** (`meta.json` + webp), so directory size = encoded size and a per-texture breakdown comes for free.

## meta.json format

Version 2 is unchanged. LE files have a `covariances` entry and **no** `quats`/`scales`:

```json
"covariances": {
  "parameterization": "log-euclidean",
  "encoding": "codebook" | "minmax" | "hybrid",
  "components": ["xx","xy","xz","yy","yz","zz"],
  "channels": ["codebook"|"minmax", ... x6],
  "codebook": [ ... k floats ],          // if any channel is codebook
  "max": 255, "mins": [...], "maxs": [...], // if any channel is minmax; mins/maxs list only the minmax channels, in component order
  "files": ["cov_0.webp", "cov_1.webp"]
}
```

Decoding a minmax channel: `min + (max_val - min || 1) * byte / max`. Std `.sog` with reduced levels: `"quats": { "max": 63, "files": [...] }` (no `max` key when it is 255, so the default output is byte-identical to upstream), and `scales.codebook` of length k.

## Files

New:
- `src/lib/utils/log-euclidean.ts`: `encodeLogCovariance` / `decodeLogCovariance` (exported from the lib index).
- `src/lib/utils/sym3.ts`: `eigenSymmetric3x3` (Jacobi) + `rotmatToQuat`, moved here from `decimate/moment-match.ts` (which re-exports them; `tools/decimate-exact.mjs` imports them from there). It must stay engine-free because worker tasks import it; import it directly, not via `utils/index`.
- `src/lib/utils/content-hash.ts`: 64-bit content hash used as the memo key.
- `src/lib/writers/write-sog-le.ts`: `leShapeEncoder`, `writeSogLeSource`.
- `src/cli/batch-sweep.ts`: `batchSweep(manifest, options)`, exported from `dist/cli.mjs`.
- `tools/batch-sweep.mjs`: YAML wrapper (uses the `yaml` package, present via typedoc); `tools/sweeps/sog-le.yaml` is an example manifest.
- `test/sog-le.test.mjs`: 11 tests.

Changed:
- `writers/write-sog.ts`: shape coding is pluggable (`SogShapeEncoder`: started before the Morton sort so worker jobs overlap, finished after it). `stdShapeEncoder(scalesCodebookSize, quatsMax)` is the default. New options `scalesCodebookSize`, `quatsMax`, `shapeEncoder`, `cache`.
- `readers/read-sog.ts`: decodes `covariances` via per-channel 256-entry lookup tables, then `decodeLogCovariance`; `unpackQuat` divides by `quats.max`.
- `read.ts` / `write.ts`: `.sog-le-codebook|minmax|hybrid` = bundled zip (output format `sog-le-bundle`; input format `sog`). Unbundled LE output = `meta.json` + `options.sogLe`.
- `types.ts` `Options`: `sogScalesCodebookSize`, `sogQuatsMax`, `sogLe`, `sogLeCodebookSize`, `sogLeMax`, `sogEncodeCache`.
- `cli/index.ts`: flags `--sog-scales-k`, `--sog-quats-max`, `--sog-le <mode>`, `--sog-le-k`, `--sog-le-max`; re-exports `batchSweep`.

## Usage

```bash
npm run build
node bin/cli.mjs in.ply out.sog-le-codebook --sog-le-k 128
node bin/cli.mjs out.sog-le-codebook decoded.ply
node tools/batch-sweep.mjs --manifest tools/sweeps/sog-le.yaml --input in.ply --output out/sweep
```

Sweep manifest keys: `scales.k`, `quats.max` (sog only); `cov.k` (codebook, hybrid); `cov.max` (minmax, hybrid); `sh.iterations`, `webp.effort`. `max` follows ffsplat (values 0..max, levels = max + 1). `sweep_mode`: `product` (default) or `zip`. A group can also have fixed `params`.

Sweep output: `<out>/<group>/<label>/{encoded/, decoded.ply, size.json}`, `<out>/<group>/sweep_metrics.csv`, and `<out>/batch_sweep_metrics.csv` (includes an uncompressed-input row). Labels look like `scales.k=64__quats.max=63`. Variants with an existing `size.json` are skipped (resume); pass `--overwrite` to redo them.

**Caching:** the input is read once. SH k-means and every WebP texture are memoized by content hash across all variants and groups, so means/sh0/shN are computed once per sweep. This also removes GPU k-means nondeterminism within a sweep, so paired variants differ only in shape. `encode_seconds` is therefore not a fair speed comparison between variants.

## Bug found: Jacobi with equal diagonal entries (fixed)

The repo's `eigenSymmetric3x3` (originally in `decimate/moment-match.ts`) computed `t = Math.sign(tau) / (...)`. When the two pivot diagonal entries were **exactly equal** (`tau = 0`), `t = 0`: it skipped the rotation but still set `A[pq] = 0`. It **silently returned wrong eigenvalues**: `[[-10,2],[2,-10]]` gave −10, −10 instead of −12, −8.

- **Effect on LE:** the shared codebook snaps diagonal entries to the same centroid, so this case is common. When the decoder was switched to the shared solver, `.sog-le-codebook` mean shape error rose from 0.098 to 0.135 (p99 0.18 → 0.42), and hybrid was affected too. Minmax was unaffected.
- **Fix** (`utils/sym3.ts`): `t = (tau < 0 ? -1 : 1) / (...)`, i.e. a 45° rotation at tau = 0, the standard Jacobi handling. Regression test: "decodes exactly equal diagonal entries" in `test/sog-le.test.mjs`.
- **Decimation** uses the same solver, so this is also a fix there. Its output changes only for merged covariances with exactly equal diagonal entries (rare); decimation and moment-match parity tests still pass. `test/fixtures/legacy-decimate-math.mjs` still contains the old buggy copy (the parity reference); leave it alone unless a test starts hitting the case.
- The shared solver stops at an absolute off-diagonal threshold of 1e-12 (or 24 rotations). Unquantized round-trip error is about 1e-12 relative, far below the float32 precision of the output.

## Verification results (Bicycle test scene, 256k gaussians, MipNeRF360 popspa-erank dummy)

Test scene: `~/data/dummy/gsplat_popspa_erank_M_dummy/MipNeRF360/Bicycle/point_cloud/iteration_40000/point_cloud.ply` (63.5 MB).

- Default `.sog` matches upstream HEAD byte for byte: means, quats, scales, sh0 and all meta.json fields except shN. shN differs even between two runs of the same build (GPU k-means is nondeterministic).
- Shape error ‖log Σ − log Σ̂‖_F, default 8-bit settings. The decoded PLY is in **Morton order**, so align the input with `sortMortonInterleaved` before comparing.

  | | size | mean | median | p99 |
  |---|---|---|---|---|
  | `.sog` | 4.85 MB | 0.141 | 0.107 | 0.457 |
  | `.sog-le-codebook` | 4.76 MB | 0.098 | 0.095 | 0.179 |
  | `.sog-le-minmax` | 4.75 MB | 0.098 | 0.099 | 0.139 |
  | `.sog-le-hybrid` | 4.81 MB | 0.088 | 0.088 | 0.133 |

- `npm test`: 847 tests, 6 failures, all in render-golden / render-options. They fail identically on untouched HEAD on the original machine (GPU environment), so they are not related to this work. Lint is clean.

## Environment notes (original machine)

- Node comes from nvm: run `source ~/.nvm/nvm.sh` first.
- No system `libvulkan`, so Dawn fails with "Cannot read properties of null (reading 'features')" during SOG output. Fix it with `export LD_LIBRARY_PATH=/scratch/flofle/.local/vulkan/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`. On a new machine, install a Vulkan loader or point at one.
- `references/` and `scratch/` are gitignored (this branch adds them to `.gitignore`), so the ffsplat checkout and `scratch/LE-integration-plan.md` do not travel with the repo.

## Differences from ffsplat to keep in mind

- ffsplat pads to the grid with fake gaussians (scale 1e-6). That can widen the LE min–max range; it is not reproduced here (`.sog` texture padding is excluded from min/max). `-minmax` ranges can therefore differ slightly from SOG-XT-1510-logmap.
- ffsplat sorts with PLAS; `.sog` uses Morton order on positions, so shape codes never influence the layout in C2. (The storyline doc's §6 says ".sog: PLAS"; that is wrong for this codec.)
- `quantize1d` weights its histogram by count^0.5. It is not pure MSE, so say "density-weighted D_LE".

## Open / next steps

- Run the full sweeps (`tools/sweeps/sog-le.yaml`) on all MipNeRF360 scenes × the 3 reconstructions; check that the size ranges of `.sog` and `.sog-le-*` overlap (needed for the iso-size tables T1/T2 and the F5 plot).
- Hook ffsplat eval onto `decoded.ply` + `size.json` / `batch_sweep_metrics.csv`.
- Knobs are 8-bit only (k ≤ 256, max ≤ 255). ffsplat's high-rate logmap settings (max 300–450) would need a 16-bit split of the covariance textures.
- Optional: a std `.sog` fairness ablation with a codebook on the quaternion components.
- Nothing is committed yet. Use conventional commits, e.g. `feat: Add Log-Euclidean SOG variants and batch sweep tool`, and a separate `fix:` commit for the Jacobi bug if preferred.

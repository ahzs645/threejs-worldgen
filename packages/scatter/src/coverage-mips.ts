// Coverage-preserving alpha mipmaps (Castaño, "Computing alpha mipmaps",
// https://www.ludicon.com/castano/blog/articles/computing-alpha-mipmaps/): each mip level's alpha is scaled so that an
// alpha test keeps the base level's coverage, so thin foliage neither thins out nor turns to speckle with distance.
// No three.js, no DOM.

/** Share of a level's texels whose alpha (0–255) is above `threshold` (0–1), after scaling alpha by `scale`. */
export function alphaCoverage(alpha: ArrayLike<number>, threshold: number, scale = 1): number {
  if (!alpha.length) return 0
  const cut = threshold * 255
  let n = 0
  for (let i = 0; i < alpha.length; i++) if (Math.min(255, alpha[i] * scale) > cut) n++
  return n / alpha.length
}

/**
 * The alpha scale that makes a mip level's alpha-tested coverage equal `target` (Castaño, "Computing alpha mipmaps"):
 * a box-filtered level loses coverage under an alpha test (thin needles and gaps average below the threshold), so each
 * level's alpha is scaled by the factor, found by bisection, that restores the base level's coverage.
 */
export function coverageScale(alpha: ArrayLike<number>, threshold: number, target: number): number {
  if (!(target > 0) || !alpha.length) return 1
  const at = (scale: number) => alphaCoverage(alpha, threshold, scale)
  const here = at(1)
  // Already as covered as the base (to half a texel): leave the level alone, so opaque texels stay opaque.
  if (Math.abs(here - target) <= 0.5 / alpha.length) return 1
  let lo: number
  let hi: number
  if (here < target) {
    // The smallest boost that restores the coverage.
    lo = 1
    hi = 8
    if (at(hi) < target) return hi
    for (let i = 0; i < 24; i++) {
      const mid = (lo + hi) / 2
      if (at(mid) < target) lo = mid
      else hi = mid
    }
    return hi
  }
  // Over-covered (rare: a level that merged gaps): the largest cut that brings it down to the base.
  lo = 0
  hi = 1
  for (let i = 0; i < 24; i++) {
    const mid = (lo + hi) / 2
    if (at(mid) > target) hi = mid
    else lo = mid
  }
  // Coverage moves in steps (texels share an alpha): keep whichever side of the target is nearer.
  return Math.abs(at(lo) - target) < Math.abs(here - target) ? lo : 1
}

/** One mip level: RGBA bytes, row-major from the top. */
export interface MipLevel {
  data: Uint8Array
  width: number
  height: number
}

/**
 * Mip levels of an atlas of equal cells (`cellWidth × cellHeight`, both powers of two), each cell reduced on its own
 * (no cell ever bleeds into its neighbour) down to `minCell` pixels on its shorter side. Colour is averaged weighted by
 * alpha (a transparent texel adds no colour), alpha box-filtered, then scaled per cell and level so its alpha-tested
 * coverage stays the base level's ({@link coverageScale}). Level 0 is the atlas itself.
 */
export function coverageMips(
  atlas: MipLevel,
  cellWidth: number,
  cellHeight: number,
  threshold = 0.5,
  minCell = 4
): MipLevel[] {
  const levels: MipLevel[] = [atlas]
  const columns = atlas.width / cellWidth
  const rows = atlas.height / cellHeight
  // Each cell's base coverage.
  const base: number[] = []
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < columns; c++) base.push(cellCoverage(atlas, c * cellWidth, r * cellHeight, cellWidth, cellHeight, threshold))
  let cw = cellWidth
  let ch = cellHeight
  let prev = atlas
  while (Math.min(cw, ch) / 2 >= minCell) {
    const w = prev.width / 2
    const h = prev.height / 2
    const data = new Uint8Array(w * h * 4)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let r = 0
        let g = 0
        let b = 0
        let a = 0
        for (const [dx, dy] of [
          [0, 0],
          [1, 0],
          [0, 1],
          [1, 1]
        ]) {
          const o = ((2 * y + dy) * prev.width + 2 * x + dx) * 4
          const al = prev.data[o + 3]
          r += prev.data[o] * al
          g += prev.data[o + 1] * al
          b += prev.data[o + 2] * al
          a += al
        }
        const o = (y * w + x) * 4
        if (a > 0) {
          data[o] = Math.round(r / a)
          data[o + 1] = Math.round(g / a)
          data[o + 2] = Math.round(b / a)
        } else {
          // Transparent: keep the plain average so filtering into it does not darken.
          for (let k = 0; k < 3; k++) {
            let s = 0
            for (const [dx, dy] of [
              [0, 0],
              [1, 0],
              [0, 1],
              [1, 1]
            ])
              s += prev.data[((2 * y + dy) * prev.width + 2 * x + dx) * 4 + k]
            data[o + k] = Math.round(s / 4)
          }
        }
        data[o + 3] = Math.round(a / 4)
      }
    cw /= 2
    ch /= 2
    const level = { data, width: w, height: h }
    // Restore each cell's coverage.
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < columns; c++) {
        const target = base[r * columns + c]
        const alpha = cellAlpha(level, c * cw, r * ch, cw, ch)
        const scale = coverageScale(alpha, threshold, target)
        if (scale === 1) continue
        const cut = threshold * 255
        for (let y = 0; y < ch; y++)
          for (let x = 0; x < cw; x++) {
            const o = ((r * ch + y) * w + c * cw + x) * 4 + 3
            const v = Math.min(255, data[o] * scale)
            let byte = Math.round(v)
            // Rounding must not move a texel across the alpha test the scale was chosen for.
            if (v > cut && !(byte > cut)) byte = Math.floor(cut) + 1
            else if (!(v > cut) && byte > cut) byte = Math.floor(cut)
            data[o] = byte
          }
      }
    levels.push(level)
    prev = level
  }
  return levels
}

function cellAlpha(level: MipLevel, x0: number, y0: number, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) out[y * w + x] = level.data[((y0 + y) * level.width + x0 + x) * 4 + 3]
  return out
}

function cellCoverage(level: MipLevel, x0: number, y0: number, w: number, h: number, threshold: number): number {
  return alphaCoverage(cellAlpha(level, x0, y0, w, h), threshold)
}

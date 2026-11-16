// What the PGMaps ground is covered with, for its colour only: crown closure under the forest (the package's
// `terrain.canopy` / `farTerrain.canopy`, else derived from the stems it lists) and how far a harvest opening has greened
// up (`openings[].ageYears`). Nothing here moves a vertex, a stem or a class: it feeds the ground's colour. No DOM, no
// three.js: the tests run these in Node.

/** A package grid's layout (`terrain` / `farTerrain`), with its optional crown closure per post. */
export interface CoverGrid {
  west: number
  south: number
  spacing: number
  columns: number
  rows: number
  /** Crown closure 0–1 per post, row-major from the south row; 0 on cut, non-treed and water ground, null unknown. */
  canopy?: Array<number | null> | null
}

/** The years since harvest an opening is drawn at when its record has no date (`ageYears` absent or null). */
export const DEFAULT_CUT_AGE_YEARS = 1

/** An opening's drawn age, years: the package's `ageYears`, else `DEFAULT_CUT_AGE_YEARS`; never negative. */
export function cutAge(ageYears: number | null | undefined): number {
  return typeof ageYears === 'number' && Number.isFinite(ageYears) ? Math.max(0, ageYears) : DEFAULT_CUT_AGE_YEARS
}

/** An opening's dispersed retention, 0–1; 0 when absent. */
export function retainedShare(share: number | null | undefined): number {
  return typeof share === 'number' && Number.isFinite(share) ? Math.min(1, Math.max(0, share)) : 0
}

/**
 * The colour the VIA renders paint fresh cut ground (a proposed block after harvest): a pinkish tan, sampled from the
 * Gun Lake CP007 render's G106B block and the Lakeview Boat Launch render's blocks, under the slash speckle.
 */
export const VIA_CUT_SRGB = '#b89a86'
/** Openings this many years old or younger are fresh cut in the VIA look; the paint fades out over the next year. */
export const VIA_FRESH_CUT_YEARS = 1

/** An sRGB hex colour as linear RGB, 0–1. */
export function srgbToLinear(hex: string): [number, number, number] {
  const v = Number.parseInt(hex.replace('#', ''), 16)
  const c = (byte: number) => {
    const s = byte / 255
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  return [c((v >> 16) & 255), c((v >> 8) & 255), c(v & 255)]
}

/**
 * The fresh cut's albedo, linear: {@link VIA_CUT_SRGB} at 0.88, the ratio between the renders' lit rock (#b1a187)
 * and the albedo that draws it under the VIA sky (pgmaps-ground.ts `naturalGround`).
 */
export const VIA_CUT_LINEAR: readonly [number, number, number] = srgbToLinear(VIA_CUT_SRGB).map((v) => v * 0.88) as [
  number,
  number,
  number
]

/** How much of the VIA cut paint a cut `age` years old takes: all of it to a year, none past two. */
export function viaFreshCut(age: number): number {
  const t = Math.min(1, Math.max(0, age - VIA_FRESH_CUT_YEARS))
  return 1 - t * t * (3 - 2 * t)
}

/**
 * The share of a cut's ground the drawn green-up covers at `age` years: a little fireweed and grass in the first year,
 * shrubs and seedlings over most of it by five, closed by fifteen. Matches the ground shader's stages
 * (`groundMaterial`), so a test can pin them.
 */
export function greenShare(age: number): number {
  const s = (a: number, b: number, x: number) => {
    const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)
  }
  return Math.min(1, 0.12 + 0.6 * s(1, 5, age) + 0.28 * s(5, 15, age))
}

/**
 * Crown closure at a package point from a grid's `canopy`, bilinear over the corners that are known; NaN outside the
 * grid, where all four corners are unknown, or when the grid has no (or a mis-sized) canopy.
 */
export function canopySampler(grid: CoverGrid | null | undefined): ((x: number, y: number) => number) | null {
  const canopy = grid?.canopy
  if (!grid || !canopy || canopy.length !== grid.columns * grid.rows) return null
  const { west, south, spacing, columns, rows } = grid
  return (x, y) => {
    const gx = (x - west) / spacing
    const gy = (y - south) / spacing
    if (!(gx >= 0 && gy >= 0 && gx <= columns - 1 && gy <= rows - 1)) return NaN
    const x0 = Math.min(columns - 2, Math.floor(gx))
    const y0 = Math.min(rows - 2, Math.floor(gy))
    const fx = gx - x0
    const fy = gy - y0
    let sum = 0
    let weight = 0
    const add = (c: number, r: number, w: number) => {
      const v = canopy[r * columns + c]
      if (typeof v !== 'number' || !Number.isFinite(v) || w <= 0) return
      sum += Math.min(1, Math.max(0, v)) * w
      weight += w
    }
    add(x0, y0, (1 - fx) * (1 - fy))
    add(x0 + 1, y0, fx * (1 - fy))
    add(x0, y0 + 1, (1 - fx) * fy)
    add(x0 + 1, y0 + 1, fx * fy)
    return weight > 1e-6 ? sum / weight : NaN
  }
}

/**
 * The package's own crown closure at a point (its grids' `canopy`: near, far, then skyline), NaN where none of them
 * knows it; never derived from stems. Rock carries a stand where this is over `ROCK_STAND_CLOSURE` (pgmaps-nature.ts).
 */
export function packageCanopySampler(input: {
  terrain: CoverGrid
  farTerrain?: CoverGrid | null
  skylineTerrain?: CoverGrid | null
}): (x: number, y: number) => number {
  const samplers = [input.terrain, input.farTerrain, input.skylineTerrain]
    .map((g) => canopySampler(g))
    .filter((s): s is (x: number, y: number) => number => !!s)
  return (x, y) => {
    for (const s of samplers) {
      const v = s(x, y)
      if (Number.isFinite(v)) return v
    }
    return NaN
  }
}

/** The package's stems: `fields` names each record's values (PGMaps' `SCENE_TREE_FIELDS`). */
export interface CoverStems {
  fields?: readonly string[]
  count: number
  data: number[]
}

/** Cell size of the closure derived from stems, metres. */
export const STEM_COVER_CELL_M = 50

/**
 * Crown closure derived from the stems a package lists, for packages without `canopy`: each 50 m cell's crown area
 * (π · (crown width / 2)², the width PGMaps widens on its coarser bands to keep the stand's closure) over the cell's
 * area, capped at 1, smoothed over its neighbours (wider past 2.8 km, where PGMaps places a stem every 80 m). Known
 * only within the stems' reach from the origin (`reach`, metres): a cell there with no stem is open ground (0);
 * NaN past it.
 */
export function stemCoverSampler(
  stems: CoverStems | null | undefined,
  reach: number
): ((x: number, y: number) => number) | null {
  if (!stems || !(stems.count > 0) || !(reach > 0)) return null
  const fields = stems.fields ?? ['x', 'y', 'z', 'species', 'height', 'crownWidth', 'tone']
  const stride = fields.length
  const fx = fields.indexOf('x')
  const fy = fields.indexOf('y')
  const fw = fields.indexOf('crownWidth')
  if (fx < 0 || fy < 0 || fw < 0 || stems.data.length < stems.count * stride) return null
  const cell = STEM_COVER_CELL_M
  const n = Math.ceil(reach / cell) + 1
  const side = 2 * n
  const area = new Float32Array(side * side)
  for (let i = 0; i < stems.count; i++) {
    const o = i * stride
    const c = Math.floor(stems.data[o + fx] / cell) + n
    const r = Math.floor(stems.data[o + fy] / cell) + n
    const w = stems.data[o + fw]
    if (c < 0 || r < 0 || c >= side || r >= side || !(w > 0)) continue
    area[r * side + c] += Math.PI * (w / 2) * (w / 2)
  }
  const closure = new Float32Array(side * side)
  const cellArea = cell * cell
  for (let r = 0; r < side; r++)
    for (let c = 0; c < side; c++) {
      const cx = (c - n + 0.5) * cell
      const cy = (r - n + 0.5) * cell
      const k = Math.hypot(cx, cy) > 2800 ? 2 : 1
      let sum = 0
      let count = 0
      for (let dr = -k; dr <= k; dr++)
        for (let dc = -k; dc <= k; dc++) {
          const rr = r + dr
          const cc = c + dc
          if (rr < 0 || cc < 0 || rr >= side || cc >= side) continue
          sum += area[rr * side + cc]
          count++
        }
      closure[r * side + c] = Math.min(1, sum / count / cellArea)
    }
  return (x, y) => {
    if (Math.hypot(x, y) > reach) return NaN
    // Bilinear between cell centres.
    const gx = x / cell + n - 0.5
    const gy = y / cell + n - 0.5
    const c0 = Math.max(0, Math.min(side - 2, Math.floor(gx)))
    const r0 = Math.max(0, Math.min(side - 2, Math.floor(gy)))
    const tx = Math.min(1, Math.max(0, gx - c0))
    const ty = Math.min(1, Math.max(0, gy - r0))
    const at = (c: number, r: number) => closure[r * side + c]
    return (
      at(c0, r0) * (1 - tx) * (1 - ty) +
      at(c0 + 1, r0) * tx * (1 - ty) +
      at(c0, r0 + 1) * (1 - tx) * ty +
      at(c0 + 1, r0 + 1) * tx * ty
    )
  }
}

/** Where the ground's crown closure comes from. */
export type CoverSource = 'package' | 'stems' | 'none'

/**
 * Crown closure at a package point, 0–1, NaN unknown: the package's near grid, its far grid, its skyline grid, then (for posts the
 * package leaves unknown, or a package with no `canopy` at all) the closure derived from its stems.
 */
export function coverSampler(input: {
  terrain: CoverGrid
  farTerrain?: CoverGrid | null
  skylineTerrain?: CoverGrid | null
  trees?: CoverStems | null
  treeRadiusMeters?: number
}): { at: (x: number, y: number) => number; source: CoverSource } {
  const near = canopySampler(input.terrain)
  const far = canopySampler(input.farTerrain)
  const skyline = canopySampler(input.skylineTerrain)
  const stems = stemCoverSampler(input.trees, input.treeRadiusMeters ?? stemReach(input.trees))
  const samplers = [near, far, skyline, stems].filter((s): s is (x: number, y: number) => number => !!s)
  const source: CoverSource = near || far || skyline ? 'package' : stems ? 'stems' : 'none'
  return {
    source,
    at(x, y) {
      for (const s of samplers) {
        const v = s(x, y)
        if (Number.isFinite(v)) return v
      }
      return NaN
    }
  }
}

/** The farthest stem from the origin, metres. */
function stemReach(stems: CoverStems | null | undefined): number {
  if (!stems) return 0
  const fields = stems.fields ?? ['x', 'y']
  const stride = Math.max(2, fields.length)
  const fx = Math.max(0, fields.indexOf('x'))
  const fy = fields.indexOf('y') >= 0 ? fields.indexOf('y') : 1
  let reach = 0
  for (let i = 0; i < stems.count; i++)
    reach = Math.max(reach, Math.hypot(stems.data[i * stride + fx], stems.data[i * stride + fy]))
  return reach
}

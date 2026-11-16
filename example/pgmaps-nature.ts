// What the PGMaps ground is where the forest is not closed: the package's `terrain.cover` (and `farTerrain.cover`,
// `skylineTerrain.cover`) codes, `SCENE_COVER` in PGMaps' scenePackage.ts. They give the ground its material (pale
// rock, yellow-green herb, dark low shrub, light ground under a sparse stand) and say where the far forest stands no
// trees (rock, water) or fewer (sparse, shrub, herb), and where a recent wildfire left its stand standing dead (burnt:
// snags, pgmaps-burn.ts). No DOM, no three.js: the tests run these in Node.

/** `terrain.cover` codes (PGMaps `SCENE_COVER`). 7 burnt: a recent wildfire, standing dead snags over regrowth. */
export const SCENE_COVER = { forest: 0, sparse: 1, shrub: 2, herb: 3, rock: 4, water: 5, cut: 6, burnt: 7 } as const
/** How many cover codes there are. */
export const COVER_CLASSES = 8

/** A package grid's layout with its optional cover code per post (row-major from the south row; null unknown). */
export interface CoverClassGrid {
  west: number
  south: number
  spacing: number
  columns: number
  rows: number
  cover?: Array<number | null> | null
}

/**
 * The most crown closure the far forest stands on each class, indexed by code: a closed stand on forest and on cut
 * ground (its retention is the package's own), an open one on sparse forest, a scattered tree on shrub and herb ground,
 * none on rock or water, and no green stand on burnt ground (its stems stand as snags, `burntShare`). A cap, not a
 * scale: a package closure already below it is kept, so a sparse stand's VRI closure is not thinned twice.
 */
export const CLASS_CLOSURE_CAP: readonly number[] = [1, 0.3, 0.12, 0.08, 0, 0, 1, 0]

/**
 * Cover weights at a package point: the share of each code (`SCENE_COVER`) among the known posts at the four corners,
 * bilinear, so a class edge blends over one grid cell instead of stepping at the posts. Grids are read in order (the
 * near grid, then the far one, then the skyline's): the first that holds the point with a known corner answers. null
 * where no grid knows the ground (outside every grid, every corner unknown, or no grid carries `cover`). The array
 * returned is reused by the next call: read it before sampling again.
 */
export function coverClassSampler(
  grids: ReadonlyArray<CoverClassGrid | null | undefined>
): ((x: number, y: number) => Float32Array | null) | null {
  const usable = grids.filter(
    (g): g is CoverClassGrid & { cover: Array<number | null> } =>
      !!g && !!g.cover && g.cover.length === g.columns * g.rows && g.columns > 1 && g.rows > 1
  )
  if (!usable.length) return null
  const out = new Float32Array(COVER_CLASSES)
  return (x, y) => {
    for (const g of usable) {
      const gx = (x - g.west) / g.spacing
      const gy = (y - g.south) / g.spacing
      if (!(gx >= 0 && gy >= 0 && gx <= g.columns - 1 && gy <= g.rows - 1)) continue
      const x0 = Math.min(g.columns - 2, Math.floor(gx))
      const y0 = Math.min(g.rows - 2, Math.floor(gy))
      const fx = gx - x0
      const fy = gy - y0
      out.fill(0)
      let weight = 0
      const add = (c: number, r: number, w: number) => {
        const v = g.cover[r * g.columns + c]
        if (w <= 0 || typeof v !== 'number' || !Number.isInteger(v) || v < 0 || v >= COVER_CLASSES) return
        out[v] += w
        weight += w
      }
      add(x0, y0, (1 - fx) * (1 - fy))
      add(x0 + 1, y0, fx * (1 - fy))
      add(x0, y0 + 1, (1 - fx) * fy)
      add(x0 + 1, y0 + 1, fx * fy)
      if (weight <= 1e-6) continue
      for (let k = 0; k < COVER_CLASSES; k++) out[k] /= weight
      return out
    }
    return null
  }
}

/**
 * Below this crown closure from the package's own `canopy`, rock (cover 4) is bare: above it, the stand stands on it.
 * PGMaps marks a treed VRI polygon whose cover components record bedrock or talus as rock over the whole polygon and
 * keeps its stand's closure there (rock showing between the crowns); bare bluffs and VRI rock carry closure 0.
 */
export const ROCK_STAND_CLOSURE = 0.05

/**
 * Whether rock at a point carries a stand: the package's own crown closure there (its `canopy`, never one derived
 * from stems, which PGMaps grows on bluffs too) is over {@link ROCK_STAND_CLOSURE}.
 */
export const rockStands = (packageClosure: number): boolean => packageClosure > ROCK_STAND_CLOSURE

/**
 * The closure cap at a point from its cover weights (`CLASS_CLOSURE_CAP`); 1 where the cover is unknown. `standing`:
 * rock there carries a stand ({@link rockStands}), capped by nothing but its own closure.
 */
export function closureCap(weights: ArrayLike<number> | null, standing = false): number {
  if (!weights) return 1
  let cap = 0
  for (let k = 0; k < COVER_CLASSES; k++)
    cap += weights[k] * (standing && k === SCENE_COVER.rock ? 1 : CLASS_CLOSURE_CAP[k])
  return cap
}

/** The share of stems a stand at `closure` keeps under `cap`: all of them at or under it, `cap / closure` over it. */
export function thinShare(closure: number, cap: number): number {
  if (!(cap > 0)) return 0
  if (!(closure > cap)) return 1
  return cap / closure
}

/**
 * The share of the ground that is open (rock, herb, shrub, water, burnt): the canopy surface stands on none of it. A
 * burnt stand's dead crowns hold no canopy: its snags are drawn one by one, and past them the ground carries them.
 */
export function openShare(weights: ArrayLike<number> | null, standing = false): number {
  if (!weights) return 0
  return (
    (standing ? 0 : weights[SCENE_COVER.rock]) +
    weights[SCENE_COVER.herb] +
    weights[SCENE_COVER.shrub] +
    weights[SCENE_COVER.water] +
    weights[SCENE_COVER.burnt]
  )
}

/** The share of the ground that is burnt (cover 7); 0 where the cover is unknown. */
export function burntShare(weights: ArrayLike<number> | null): number {
  return weights ? weights[SCENE_COVER.burnt] : 0
}

/** Whether a package stem stands on burnt ground (half or more of the blended cover there): drawn as a snag. */
export const onBurntGround = (weights: ArrayLike<number> | null): boolean => burntShare(weights) >= 0.5

/**
 * The closure the canopy surface is raised and coloured by: none where half or more of the ground is open (rock,
 * herb, shrub, water or burnt: those are never lifted by a stand's height), else the closure under the class cap.
 */
export function canopyClosure(closure: number, weights: ArrayLike<number> | null, standing = false): number {
  if (openShare(weights, standing) >= 0.5) return 0
  return Math.min(closure, closureCap(weights, standing))
}

/** A stem's own number in [0, 1), from its position: the same stem keeps or drops the same way in every build. */
export function stemHash(x: number, y: number): number {
  let h = Math.imul(Math.round(x * 16) | 0, 0x27d4eb2d) ^ Math.imul(Math.round(y * 16) | 0, 0x165667b1)
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b)
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35)
  h ^= h >>> 16
  return (h >>> 0) / 4294967296
}

/** Whether a stem at (x, y) stands, keeping `share` of them. */
export const keepStem = (x: number, y: number, share: number): boolean => share >= 1 || stemHash(x, y) < share

/**
 * The ground's material weights at a point, for the vertex attribute `aNature`: rock, herb, shrub and sparse forest,
 * each 0–1 (shares of the known cover). Forest, cut, water, burnt (its own attribute, `aBurn`) and unknown ground keep
 * the ground's own look (zeros).
 */
export function natureWeights(weights: ArrayLike<number> | null, out: Float32Array, at: number): void {
  out[at] = weights ? weights[SCENE_COVER.rock] : 0
  out[at + 1] = weights ? weights[SCENE_COVER.herb] : 0
  out[at + 2] = weights ? weights[SCENE_COVER.shrub] : 0
  out[at + 3] = weights ? weights[SCENE_COVER.sparse] : 0
}

/** Diagnostics: how many of a grid's posts carry each code (forest … burnt), and how many are unknown; null without `cover`. */
export function coverCounts(grid: CoverClassGrid | null | undefined): { codes: number[]; unknown: number } | null {
  const cover = grid?.cover
  if (!cover) return null
  const codes = new Array(COVER_CLASSES).fill(0)
  let unknown = 0
  for (const v of cover)
    if (typeof v === 'number' && Number.isInteger(v) && v >= 0 && v < COVER_CLASSES) codes[v]++
    else unknown++
  return { codes, unknown }
}

/** The share of the ground that is rock or water: no stem is drawn where it is half or more (`onBareGround`). */
export function bareShare(weights: ArrayLike<number> | null, standing = false): number {
  return weights ? (standing ? 0 : weights[SCENE_COVER.rock]) + weights[SCENE_COVER.water] : 0
}

/**
 * Whether a package stem stands on rock or water by the cover codes (half or more of the blended cover there). PGMaps
 * still lists stems on bluff posts (its figures use them); the picture shows the rock through them instead. `standing`:
 * rock there carries a stand ({@link rockStands}), whose stems are drawn over it.
 */
export const onBareGround = (weights: ArrayLike<number> | null, standing = false): boolean =>
  bareShare(weights, standing) >= 0.5

/**
 * The sea's drawn level `distance` metres from the eye on the skyline grid, whose heights carry the earth's curve
 * (PGMaps `earthCurvature`): 0 − d² / (2R) × (1 − k).
 */
export function droppedSeaLevel(distance: number, curvature: { earthRadiusMeters: number; refraction: number }): number {
  return -((distance * distance) / (2 * curvature.earthRadiusMeters)) * (1 - curvature.refraction)
}

/** A water post within this many metres above the dropped sea level is sea, and is laid at that level. */
export const SEA_SNAP_M = 8

/**
 * The height a skyline water post (cover 5) is drawn at: the dropped sea level when its height is unknown or within
 * {@link SEA_SNAP_M} of it (the sea, which PGMaps sends no polygons for on that grid), else its own (a lake inland).
 */
export function skylineWaterHeight(height: number | null, seaLevel: number): number {
  if (height === null || !Number.isFinite(height) || height <= seaLevel + SEA_SNAP_M) return seaLevel
  return height
}

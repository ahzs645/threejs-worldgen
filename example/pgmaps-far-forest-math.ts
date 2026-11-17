// The far forest's arithmetic, kept free of three.js and the DOM so the tests run it in Node: where stems dissolve into
// the canopy surface (Visual Nature Studio's "dissolve" rule), the generated tree silhouettes and the baked stand patches
// they are splatted into, coverage-preserving mipmaps (Castaño) for both atlases, the cap on sideways stretch, the
// water's wind roughness (Cox–Munk), the final stand's sector, and a synthetic real-density stand for the standalone
// capture test.
//
// Package coordinates throughout: local metres, x east, y north, z up.

// ---------------------------------------------------------------- the dissolve distance

/**
 * Metres past which a tree of `height` metres is under `px` pixels tall in a frame `frameHeightPx` pixels tall with a
 * `verticalFovDegrees` field: VNS's dissolve rule, measured relative to the image as VNS does. A pinhole's focal length
 * in pixels is (height / 2) / tan(vfov / 2); a tree `h` metres tall `d` metres off is h · f / d pixels tall.
 */
export function canopyCutoffMeters(height: number, frameHeightPx: number, verticalFovDegrees: number, px = 4): number {
  if (!(height > 0) || !(frameHeightPx > 0) || !(px > 0)) return Infinity
  const vfov = Math.min(179, Math.max(0.1, verticalFovDegrees))
  const focal = frameHeightPx / 2 / Math.tan((vfov * Math.PI) / 360)
  return (height * focal) / px
}

/** The median of a list (NaN for none); the list is not changed. */
export function median(values: ArrayLike<number>): number {
  const sorted = Array.from(values)
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b)
  if (!sorted.length) return NaN
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

// ---------------------------------------------------------------- sideways stretch

/** The most a drawn tree is widened (or narrowed) against its grown shape. */
export const MAX_STRETCH = 1.5

/**
 * A stem's width scale, given its height scale, kept within `max` of it: a card or impostor grown at one crown ratio
 * is never drawn more than 1.5 times wider (or narrower) than its own shape. Widening a stem to hold a stand's closure
 * drew far stems 55–58 m wide on 22 m stems, the squat "shrub" look; closure is carried by the stand instead.
 */
export function cappedWidthScale(widthScale: number, heightScale: number, max = MAX_STRETCH): number {
  if (!(heightScale > 0)) return widthScale
  if (!(widthScale > 0)) return heightScale
  return Math.min(heightScale * max, Math.max(heightScale / max, widthScale))
}

// ---------------------------------------------------------------- coverage-preserving mipmaps

// Shared with threejs-scatter's impostor bake (Castaño's coverage-preserving alpha mips).
export { alphaCoverage, coverageMips, coverageScale, type MipLevel } from '../packages/scatter/src/coverage-mips.ts'
import { contentHash } from './pgmaps-capture-math.ts'

// ---------------------------------------------------------------- tree silhouettes

/** The silhouette families drawn: one per PGMaps species. */
export const SPECIES_ORDER = [
  'pine',
  'spruce',
  'fir',
  'douglasFir',
  'aspen',
  'birch',
  'hemlock',
  'blackSpruce',
  'cottonwood'
] as const
export type SpeciesName = (typeof SPECIES_ORDER)[number]
/** Species index into {@link SPECIES_ORDER}; unknown species draw as spruce. */
export const speciesSlot = (name: string): number => {
  const k = (SPECIES_ORDER as readonly string[]).indexOf(name)
  return k < 0 ? 1 : k
}

/**
 * What a sprite draws: a species' living tree, or a snag (a fire-killed stem: bare trunk and dead branch stubs, burnt
 * ground, pgmaps-burn.ts). A snag's slot follows the species' ({@link SNAG_SLOT}).
 */
export type SpriteKind = SpeciesName | 'snag'
/** The snag's slot in the far forest's sprite and colour tables, after the species'. */
export const SNAG_SLOT = SPECIES_ORDER.length

/**
 * Crown width ÷ height a silhouette is drawn at (PGMaps' `SPECIES_CROWN_RATIO`; real crowns, not widened). A snag's is
 * the reach of its branch stubs.
 */
export const SILHOUETTE_CROWN_RATIO: Record<SpriteKind, number> = {
  pine: 0.16,
  spruce: 0.2,
  fir: 0.15,
  douglasFir: 0.24,
  aspen: 0.3,
  birch: 0.28,
  hemlock: 0.2,
  blackSpruce: 0.14,
  cottonwood: 0.36,
  snag: 0.1
}
/**
 * The crown ratio a patch's stem count is reckoned by: a burnt stand stands as many snags as it stood trees (a
 * spruce-pine stand's crowns), though each covers far less.
 */
const DENSITY_CROWN_RATIO: Record<SpriteKind, number> = { ...SILHOUETTE_CROWN_RATIO, snag: 0.2 }
/** Share of the height that is clear stem below the crown. */
export const CROWN_BASE: Record<SpeciesName, number> = {
  pine: 0.5,
  spruce: 0.12,
  fir: 0.1,
  douglasFir: 0.25,
  aspen: 0.42,
  birch: 0.45,
  hemlock: 0.16,
  blackSpruce: 0.12,
  cottonwood: 0.38
}

/**
 * A snag's silhouette at (x, v): x across in metres per metre of height, v up 0…1. A bare trunk tapering from about
 * 2.8% of the height across at the base to 1% at the top (bolder than the wood, as the VIA renders draw a snag far
 * off: a dark line a few pixels wide), the top broken off on some (`seed`), and dead branch stubs (1.3–3.8% of the
 * height long, drooping) up its upper two thirds. Dark grey wood, lit on one side.
 */
export function snagSilhouette(x: number, v: number, seed: number): SilhouetteSample {
  const out: SilhouetteSample = { inside: false, albedo: 0, nx: 0, ny: 0 }
  const top = 0.86 + 0.14 * fract(Math.sin(seed * 91.17) * 4373.1)
  if (v < 0 || v > top) return out
  const half = 0.014 - 0.009 * (v / top)
  const lean = 0.004 * noise1(v * 3, seed + 9)
  const dx = x - lean
  if (Math.abs(dx) < half) {
    out.inside = true
    out.albedo = 0.62 + 0.18 * noise1(v * 40, seed + 4)
    out.nx = dx / half
    out.ny = 0.1
    return out
  }
  if (v < 0.32) return out
  // Branch stubs: whorl heights from the seed, alternating sides, shorter toward the top.
  const stubs = 11
  for (let k = 0; k < stubs; k++) {
    const r1 = fract(Math.sin((seed + 1) * 12.9898 + k * 78.233) * 43758.5453)
    const r2 = fract(Math.sin((seed + 7) * 39.3468 + k * 11.135) * 24634.6345)
    const vk = 0.34 + (top - 0.38) * ((k + 0.5 * r1) / stubs)
    const side = (k + seed) % 2 === 0 ? 1 : -1
    const length = (0.013 + 0.025 * r2) * (1 - 0.45 * ((vk - 0.34) / Math.max(0.1, top - 0.34)))
    const along = dx * side
    if (along < 0 || along > length) continue
    const centre = vk - 0.35 * along
    if (Math.abs(v - centre) < 0.004) {
      out.inside = true
      out.albedo = 0.55 + 0.15 * r1
      out.nx = side * 0.4
      out.ny = 0.45
      return out
    }
  }
  return out
}

/** One sample of a silhouette: coverage (0/1), albedo multiplier (0–1), and the view-space normal's x and y (−1…1). */
interface SilhouetteSample {
  inside: boolean
  albedo: number
  nx: number
  ny: number
}

const fract = (v: number) => v - Math.floor(v)
/** A smooth value noise in [−1, 1] along one axis, from a seed. */
function noise1(t: number, seed: number): number {
  const i = Math.floor(t)
  const f = t - i
  const h = (n: number) => fract(Math.sin((n + seed * 12.9898) * 78.233) * 43758.5453) * 2 - 1
  const s = f * f * (3 - 2 * f)
  return h(i) * (1 - s) + h(i + 1) * s
}

/**
 * A tree's silhouette at (u, v): u across, −1…1 of the crown's half-width at its widest; v up, 0 at the base and 1 at
 * the top. Conifers are spires (spruce and fir: a cone of whorls from low on the stem; Douglas-fir broader and ragged;
 * lodgepole pine a narrow rounded crown on a tall clean stem); aspen and birch a rounded lumpy crown. `seed` varies the
 * ragged edge. The shading is the crown's own: the normal turns from the sun side to the shade side across the crown
 * and up toward the top, and each whorl is darker underneath.
 */
export function silhouette(species: SpriteKind, u: number, v: number, seed: number): SilhouetteSample {
  if (species === 'snag') return snagSilhouette((u * SILHOUETTE_CROWN_RATIO.snag) / 2, v, seed)
  const out: SilhouetteSample = { inside: false, albedo: 0, nx: 0, ny: 0 }
  if (v < 0 || v > 1) return out
  const base = CROWN_BASE[species]
  const broadleaf = species === 'aspen' || species === 'birch' || species === 'cottonwood'
  // The stem below the crown: a thin dark line (sub-pixel far off).
  if (v < base + 0.02) {
    if (Math.abs(u) < 0.09) {
      out.inside = true
      out.albedo = 0.38
      out.nx = u / 0.09
      out.ny = 0
    }
    return out
  }
  const t = (v - base) / (1 - base) // 0 at the crown base, 1 at the top
  let w: number
  let ny = 0.25
  let albedo: number
  if (broadleaf) {
    // An ellipse with lumps.
    const e = Math.sqrt(Math.max(0, 1 - (2 * t - 1) * (2 * t - 1)))
    w = e * (0.85 + 0.15 * noise1(t * 7 + 3, seed)) * (0.9 + 0.1 * noise1(u * 4, seed + 5))
    ny = 0.1 + 0.45 * t
    albedo = (0.75 + 0.25 * noise1(u * 6 + t * 9, seed + 1)) * (0.3 + 0.7 * t)
  } else if (species === 'pine') {
    // A narrow crown, rounded at the top, its widest a third of the way up.
    const s = t < 0.35 ? 0.55 + 0.45 * (t / 0.35) : Math.sqrt(Math.max(0, 1 - ((t - 0.35) / 0.65) ** 2))
    w = s * (0.8 + 0.2 * noise1(t * 9, seed))
    ny = 0.1 + 0.35 * t
    albedo = (0.75 + 0.25 * noise1(t * 11 + u * 3, seed + 2)) * (0.3 + 0.7 * t)
  } else {
    // A spire of whorls: the cone's half-width grows down from the leader; each whorl flares at its base.
    const whorls = species === 'douglasFir' ? 7 : species === 'fir' ? 11 : 9
    const phase = fract((1 - t) * whorls + 0.3 * noise1(t * 3, seed))
    const cone = Math.pow(1 - t, species === 'fir' ? 1.05 : 0.92)
    const flare = species === 'douglasFir' ? 0.3 : 0.18
    w = cone * (1 - flare + flare * phase) * (1 + (species === 'douglasFir' ? 0.18 : 0.08) * noise1(t * 23, seed + 3))
    ny = 0.08 + 0.2 * (1 - phase)
    // Low in the crown it is dark (the stand shades it: in a closed stand only the upper crowns see the sky), and a
    // little darker under each whorl.
    albedo = (0.78 + 0.22 * (1 - phase * phase)) * (0.1 + 0.9 * Math.pow(t, 0.9))
  }
  if (w <= 0 || Math.abs(u) > w) return out
  out.inside = true
  out.albedo = Math.min(1, Math.max(0, albedo))
  out.nx = Math.max(-1, Math.min(1, (u / w) * 0.9))
  out.ny = Math.max(-1, Math.min(1, ny))
  return out
}

/** A raster of RGBA bytes: R albedo, G normal x, B normal y (both 0.5 + n / 2), A coverage. */
export interface SpriteRaster {
  data: Uint8Array
  width: number
  height: number
}

/**
 * A tree card: the silhouette rasterised into `width × height` pixels (the tree fills the height; the crown's widest
 * part spans `crownRatio × height` metres scaled into the width), each pixel the mean of `ss × ss` samples.
 * `aspect` is the card's width ÷ height in metres, so the crown keeps its proportions.
 */
export function rasterTree(
  species: SpriteKind,
  width: number,
  height: number,
  aspect: number,
  seed: number,
  ss = 3
): SpriteRaster {
  const data = new Uint8Array(width * height * 4)
  const crown = SILHOUETTE_CROWN_RATIO[species]
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let cover = 0
      let albedo = 0
      let nx = 0
      let ny = 0
      for (let j = 0; j < ss; j++)
        for (let i = 0; i < ss; i++) {
          // Card x in metres across (per metre of height): −aspect/2 … aspect/2; u relative to the crown half-width.
          const cx = ((x + (i + 0.5) / ss) / width - 0.5) * aspect
          const v = 1 - (y + (j + 0.5) / ss) / height
          const s = silhouette(species, cx / (crown / 2), v, seed)
          if (!s.inside) continue
          cover++
          albedo += s.albedo
          nx += s.nx
          ny += s.ny
        }
      const o = (y * width + x) * 4
      if (cover) {
        data[o] = Math.round((albedo / cover) * 255)
        data[o + 1] = Math.round((0.5 + nx / cover / 2) * 255)
        data[o + 2] = Math.round((0.5 + ny / cover / 2) * 255)
      } else {
        data[o] = 128
        data[o + 1] = 128
        data[o + 2] = 160
      }
      data[o + 3] = Math.round((cover / (ss * ss)) * 255)
    }
  return { data, width, height }
}

// ---------------------------------------------------------------- the stand patch

/** The patch a stand impostor shows: a disc this many metres across, one per {@link PATCH_SPACING} cell. */
export const PATCH_RADIUS = 17
/** Patches stand on this lattice, metres (one slab per cell). */
export const PATCH_SPACING = 30
/** The stand height a patch is baked at, metres; a slab is scaled to its own stand's height. */
export const PATCH_HEIGHT = 22
/** Angles (degrees) between the view and the patch's ground a patch is baked from. */
export const PATCH_ELEVATIONS: readonly number[] = [0, 6, 12, 20, 30, 42, 60]
/** Crown closures a patch is baked at; a slab takes the nearer (dithered) of the two for its stand. */
export const PATCH_CLOSURES: readonly number[] = [0.18, 0.4, 0.75]

/** The nearest baked elevation's index, and the fraction toward the next, for an angle in degrees. */
export function elevationFrame(degrees: number): { index: number; next: number; t: number } {
  const e = PATCH_ELEVATIONS
  const a = Math.min(e[e.length - 1], Math.max(e[0], degrees))
  for (let i = 0; i < e.length - 1; i++)
    if (a <= e[i + 1]) {
      const t = (a - e[i]) / (e[i + 1] - e[i])
      return { index: t < 0.5 ? i : i + 1, next: i + 1, t }
    }
  return { index: e.length - 1, next: e.length - 1, t: 0 }
}

/** The steepest angle (degrees) between the eye's ray and a patch's ground a bake is drawn for. */
export const MAX_PATCH_ANGLE = 60

/**
 * How far up the picture a point `depth` metres back in a patch stands, seen at `elevationDegrees` to the patch's
 * ground: the eye looks about level (a road view: the angle comes from the ground's slope, not from looking down), so
 * ground `y` metres back rises y · tan α while the trees on it stand their full height.
 */
export const patchRise = (depth: number, elevationDegrees: number): number =>
  depth * Math.tan((Math.min(MAX_PATCH_ANGLE, Math.max(0, elevationDegrees)) * Math.PI) / 180)

/**
 * The height (metres) a patch's picture spans, seen at `elevationDegrees` to its ground: its depth risen (2R · tan α)
 * and its trees at full height. The slab is drawn this tall.
 */
export const patchSpan = (elevationDegrees: number, standHeight = PATCH_HEIGHT, radius = PATCH_RADIUS): number =>
  patchRise(2 * radius, elevationDegrees) + standHeight

/** A small deterministic generator (mulberry32). */
export function rng(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Stems in a baked patch: x across, y away from the eye (metres, in the disc), height and crown width. */
export interface PatchStem {
  x: number
  y: number
  height: number
  crown: number
  seed: number
}

/**
 * The stems of a disc of `radius` at `closure` and real crowns: stratified jitter at the spacing whose crowns cover
 * `closure` of the ground (n = closure · area / crown area), heights about the stand's (±20%), crowns
 * `crownRatio × height`. Deterministic for a seed.
 */
export function patchStems(
  species: SpriteKind,
  closure: number,
  seed: number,
  radius = PATCH_RADIUS,
  standHeight = PATCH_HEIGHT
): PatchStem[] {
  const random = rng(seed)
  const crownRatio = SILHOUETTE_CROWN_RATIO[species]
  const crownArea = Math.PI * ((DENSITY_CROWN_RATIO[species] * standHeight) / 2) ** 2
  const n = Math.max(1, Math.round((closure * Math.PI * radius * radius) / crownArea))
  const cell = Math.sqrt((Math.PI * radius * radius) / n)
  const stems: PatchStem[] = []
  for (let y = -radius; y < radius; y += cell)
    for (let x = -radius; x < radius; x += cell) {
      const px = x + random() * cell
      const py = y + random() * cell
      if (px * px + py * py > radius * radius) continue
      const height = standHeight * (0.8 + 0.4 * random())
      stems.push({
        x: px,
        y: py,
        height,
        crown: crownRatio * height * (0.85 + 0.3 * random()),
        seed: Math.floor(random() * 1e6)
      })
    }
  return stems
}

/** The width ÷ height (metres) of every tree card and patch sprite: wide enough for the broadest crown, stretched. */
export const CARD_ASPECT = 0.45

/**
 * Bakes a patch into a `width × height` cell seen at `elevationDegrees` to its ground, orthographic: the picture spans
 * 2R across and {@link patchSpan} up; stems are drawn far to near, each one of the species' `sprites` (tree cards from
 * {@link rasterTree} at {@link CARD_ASPECT}) scaled to its height and crown, alpha over. The patch's ground stays
 * transparent, so the terrain's own forest floor (dark) shows through the gaps. Trees far back in the patch, seen past
 * the crowns in front, are darkened a little (the stand's own shade).
 */
export function bakePatch(
  stems: readonly PatchStem[],
  species: SpriteKind,
  sprites: readonly SpriteRaster[],
  elevationDegrees: number,
  width: number,
  height: number,
  radius = PATCH_RADIUS,
  standHeight = PATCH_HEIGHT,
  pad = 2
): SpriteRaster {
  const span = patchSpan(elevationDegrees, standHeight, radius)
  const innerW = width - 2 * pad
  const innerH = height - 2 * pad
  const pxPerM = innerW / (2 * radius)
  const pyPerM = innerH / span
  // Premultiplied float accumulation.
  const acc = new Float32Array(width * height * 4)
  const ordered = [...stems].sort((p, q) => q.y - p.y) // far first
  const crownRatio = SILHOUETTE_CROWN_RATIO[species]
  for (const s of ordered) {
    const sprite = sprites[s.seed % sprites.length]
    const baseV = patchRise(s.y + radius, elevationDegrees) // metres up the picture from the disc's front edge
    const tall = s.height
    if (tall <= 0.01) continue
    // The card's width in metres: its aspect × height, widened to this stem's own crown (within the stretch cap).
    const cardW = CARD_ASPECT * s.height * cappedWidthScale(s.crown / s.height / crownRatio, 1)
    const x0 = Math.max(0, Math.floor(pad + (s.x - cardW / 2 + radius) * pxPerM))
    const x1 = Math.min(width, Math.ceil(pad + (s.x + cardW / 2 + radius) * pxPerM))
    const y0 = Math.max(0, Math.floor(pad + innerH - (baseV + tall) * pyPerM))
    const y1 = Math.min(height, Math.ceil(pad + innerH - baseV * pyPerM))
    // Farther in the patch, a little darker: seen through the crowns in front.
    const shade = 1 - 0.4 * ((s.y + radius) / (2 * radius))
    const sw = sprite.width
    const sh = sprite.height
    const sd = sprite.data
    for (let py = y0; py < y1; py++)
      for (let px = x0; px < x1; px++) {
        let alpha = 0
        let albedo = 0
        let nx = 0
        let ny = 0
        for (let j = 0; j < 2; j++)
          for (let i = 0; i < 2; i++) {
            const mx = (px + (i + 0.5) / 2 - pad) / pxPerM - radius
            const my = (pad + innerH - (py + (j + 0.5) / 2)) / pyPerM
            const u = (mx - s.x) / cardW + 0.5
            const v = 1 - (my - baseV) / tall
            if (u < 0 || u >= 1 || v < 0 || v >= 1) continue
            const o = (Math.floor(v * sh) * sw + Math.floor(u * sw)) * 4
            const al = sd[o + 3] / 255
            if (!al) continue
            alpha += al
            albedo += (sd[o] / 255) * al
            nx += (sd[o + 1] / 127.5 - 1) * al
            ny += (sd[o + 2] / 127.5 - 1) * al
          }
        if (!alpha) continue
        const cover = alpha / 4
        const o = (py * width + px) * 4
        const keep = 1 - cover
        acc[o] = acc[o] * keep + (albedo / alpha) * shade * cover
        acc[o + 1] = acc[o + 1] * keep + (nx / alpha) * cover
        acc[o + 2] = acc[o + 2] * keep + (ny / alpha) * cover
        acc[o + 3] = acc[o + 3] * keep + cover
      }
  }
  const data = new Uint8Array(width * height * 4)
  for (let i = 0; i < width * height; i++) {
    const al = acc[i * 4 + 3]
    const o = i * 4
    if (al > 1e-4) {
      data[o] = Math.round(Math.min(1, acc[o] / al) * 255)
      data[o + 1] = Math.round((0.5 + Math.max(-1, Math.min(1, acc[o + 1] / al)) / 2) * 255)
      data[o + 2] = Math.round((0.5 + Math.max(-1, Math.min(1, acc[o + 2] / al)) / 2) * 255)
    } else {
      data[o] = 90
      data[o + 1] = 128
      data[o + 2] = 160
    }
    data[o + 3] = Math.round(Math.min(1, al) * 255)
  }
  return { data, width, height }
}

/** Copies a raster into an atlas at cell (column, row). */
export function blit(atlas: SpriteRaster, cell: SpriteRaster, column: number, row: number): void {
  for (let y = 0; y < cell.height; y++) {
    const from = y * cell.width * 4
    const to = ((row * cell.height + y) * atlas.width + column * cell.width) * 4
    atlas.data.set(cell.data.subarray(from, from + cell.width * 4), to)
  }
}

// ---------------------------------------------------------------- water roughness from wind

/**
 * Cox & Munk's mean-square slope of a wind-roughened surface, σ² ≈ 0.003 + 0.00512 · U (U in m/s at 12.5 m), for the
 * open sea. A lake is fetch-limited: its slopes are a fraction of the sea's (`fetch`, 0.25 by default, an inference
 * to tune against photos), and calm water is a mirror (σ² = 0 at U = 0, the 0.003 floor easing in by 2 m/s).
 * Returns σ² and the matching GGX roughness α = σ (the slope standard deviation).
 */
export function coxMunk(windMs: number, fetch = 0.25): { slopeVariance: number; roughness: number } {
  const u = Number.isFinite(windMs) ? Math.max(0, windMs) : 0
  const floor = 0.003 * Math.min(1, u / 2)
  const slopeVariance = fetch * (floor + 0.00512 * u)
  return { slopeVariance, roughness: Math.sqrt(slopeVariance) }
}

/**
 * Vertical blur of a reflection, in pixels of a frame whose focal length is `focalPx`, for a slope variance: a
 * reflected ray spreads over about 2σ in the vertical plane, so a reflected point smears over 2σ · f pixels up and
 * down (the shoreline streaks of rippled water); across, a sixth of that.
 */
export function reflectionBlurPx(slopeVariance: number, focalPx: number): { vertical: number; horizontal: number } {
  const vertical = 2 * Math.sqrt(Math.max(0, slopeVariance)) * Math.max(0, focalPx)
  return { vertical, horizontal: vertical / 6 }
}

// ---------------------------------------------------------------- the final stand's sector

/** Bearing (degrees clockwise from north) of a package point from an eye. */
export const bearingOf = (eye: readonly number[], x: number, y: number): number =>
  ((Math.atan2(x - eye[0], y - eye[1]) * 180) / Math.PI + 360) % 360

/** Whether a bearing lies in a span [from, to] clockwise (degrees; `to` may pass 360). */
export function inBearings(bearing: number, span: readonly [number, number]): boolean {
  const [from, to] = span
  const width = to - from
  if (!(width < 360)) return true
  const d = (((bearing - from) % 360) + 360) % 360
  return d <= ((width % 360) + 360) % 360 || width === 0
}

/** Whether a package point is inside a final stand's reach: within `radius` of the eye and inside its bearings. */
export function inStandSector(
  eye: readonly number[],
  stand: { radiusMeters: number; bearings: readonly [number, number] },
  x: number,
  y: number
): boolean {
  if (Math.hypot(x - eye[0], y - eye[1]) > stand.radiusMeters) return false
  return inBearings(bearingOf(eye, x, y), stand.bearings)
}

/**
 * What placing a final stand reads besides the ground it stands on, as a key: its fields, its species and every one of
 * its stems (`contentHash`, nothing sampled), the eye and the distance from it the stand starts at. On the same ground
 * the same key places the same cards, so a placed stand is kept and shown again for it. The stand's radius and
 * bearings only say where the stand patches give way to it: they are applied each time and are not part of the key.
 */
export function finalStandKey(
  stand: { fields: readonly string[]; species: readonly string[]; count: number; data: ArrayLike<number> },
  eye: readonly number[],
  inner: number
): string {
  return contentHash({
    fields: [...stand.fields],
    species: [...stand.species],
    count: stand.count,
    data: stand.data,
    eye: [eye[0], eye[1]],
    inner
  })
}

// ---------------------------------------------------------------- a synthetic real-density stand

/** Stems per hectare of a closed stand (PGMaps caps the final stand about here). */
export const REAL_STEMS_PER_HA = 800

/**
 * A real-density stand for the standalone capture test (`?captureTest`, `pgmaps.syntheticStand`): stems on a jittered
 * lattice over a sector of an eye, at `density × closure` stems per hectare where the ground's crown closure is known
 * and above 0.05, standing on `groundAt`, species and heights drawn from `mix` (each species' share and median height).
 * Returns the request's `stand` block (fields as a package's `trees`).
 */
export function syntheticStand(
  eye: readonly number[],
  bearings: [number, number],
  radius: number,
  closureAt: (x: number, y: number) => number,
  groundAt: (x: number, y: number) => number,
  mix: Array<{ species: string; share: number; height: number }>,
  options: { inner?: number; density?: number; seed?: number; maxStems?: number } = {}
): {
  fields: string[]
  species: string[]
  count: number
  data: Float32Array
  radiusMeters: number
  bearings: [number, number]
} {
  const inner = options.inner ?? 0
  const density = options.density ?? REAL_STEMS_PER_HA
  const random = rng(options.seed ?? 7)
  const species = mix.map((m) => m.species)
  const total = mix.reduce((s, m) => s + m.share, 0) || 1
  const cell = 50
  const out: number[] = []
  const max = options.maxStems ?? 3_000_000
  const n = Math.ceil(radius / cell)
  const inReach = (cx: number, cy: number) => {
    const d = Math.hypot(cx - eye[0], cy - eye[1])
    return d <= radius + cell && d >= inner - cell && inBearings(bearingOf(eye, cx, cy), bearings)
  }
  // Over the cap, every cell is thinned alike (never the rows read last left empty): the expected count first.
  let expected = 0
  for (let r = -n; r < n; r++)
    for (let c = -n; c < n; c++) {
      const cx = eye[0] + (c + 0.5) * cell
      const cy = eye[1] + (r + 0.5) * cell
      if (!inReach(cx, cy)) continue
      const closure = closureAt(cx, cy)
      if (closure > 0.05) expected += (density * Math.min(1, closure) * cell * cell) / 10_000
    }
  const thin = expected > max ? (max / expected) * 0.98 : 1
  for (let r = -n; r < n && out.length / 7 < max; r++)
    for (let c = -n; c < n; c++) {
      const cx = eye[0] + (c + 0.5) * cell
      const cy = eye[1] + (r + 0.5) * cell
      if (!inReach(cx, cy)) continue
      const closure = closureAt(cx, cy)
      if (!(closure > 0.05)) continue
      const count = (density * Math.min(1, closure) * cell * cell * thin) / 10_000
      const stems = Math.floor(count) + (random() < count % 1 ? 1 : 0)
      for (let k = 0; k < stems; k++) {
        const x = cx + (random() - 0.5) * cell
        const y = cy + (random() - 0.5) * cell
        const dd = Math.hypot(x - eye[0], y - eye[1])
        if (dd > radius || dd < inner || !inBearings(bearingOf(eye, x, y), bearings)) continue
        let pick = random() * total
        let s = 0
        while (s < mix.length - 1 && pick > mix[s].share) pick -= mix[s++].share
        const height = mix[s].height * (0.75 + 0.5 * random())
        const crown =
          (SILHOUETTE_CROWN_RATIO[
            (SPECIES_ORDER as readonly string[]).includes(mix[s].species) ? (mix[s].species as SpeciesName) : 'spruce'
          ] ?? 0.2) * height
        out.push(x, y, groundAt(x, y), s, height, crown, random())
      }
    }
  return {
    fields: ['x', 'y', 'z', 'species', 'height', 'crownWidth', 'tone'],
    species,
    count: out.length / 7,
    data: new Float32Array(out),
    radiusMeters: radius,
    bearings
  }
}

// ---------------------------------------------------------------- jitter

/** The Halton sequence's k-th value in a base (k from 1). */
export function halton(k: number, base: number): number {
  let f = 1
  let r = 0
  let i = k
  while (i > 0) {
    f /= base
    r += f * (i % base)
    i = Math.floor(i / base)
  }
  return r
}

/** Sub-pixel offsets (−0.5…0.5 pixels) of `n` samples: Halton (2, 3), the first at the pixel centre. */
export function jitterOffsets(n: number): Array<[number, number]> {
  const out: Array<[number, number]> = [[0, 0]]
  for (let k = 1; out.length < n; k++) out.push([halton(k, 2) - 0.5, halton(k, 3) - 0.5])
  return out.slice(0, Math.max(1, n))
}

/** sRGB byte → linear, and back (for averaging samples in linear light). */
export const SRGB_TO_LINEAR: Float32Array = (() => {
  const t = new Float32Array(256)
  for (let i = 0; i < 256; i++) {
    const c = i / 255
    t[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }
  return t
})()
export function linearToSrgbByte(v: number): number {
  const c = Math.min(1, Math.max(0, v))
  const s = c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055
  return Math.round(s * 255)
}

/**
 * {@link linearToSrgbByte} as a table, for a whole picture (20 million values): `steps[b]` is the smallest linear value
 * the function gives byte `b` or more for, found from the function itself by bisection (it never steps down), and
 * `guess[k]` its byte at k / 4096. The byte of v is `guess` at v's slice, moved up while v is at or past the next step.
 */
let srgbTable: { steps: Float64Array; guess: Uint8Array } | null = null
function srgbSteps() {
  if (srgbTable) return srgbTable
  const steps = new Float64Array(257)
  steps[0] = -Infinity
  steps[256] = Infinity
  for (let b = 1; b < 256; b++) {
    let below = 0
    let at = 1
    for (;;) {
      const mid = (below + at) / 2
      if (mid === below || mid === at) break
      if (linearToSrgbByte(mid) >= b) at = mid
      else below = mid
    }
    steps[b] = at
  }
  const guess = new Uint8Array(4097)
  for (let k = 0; k <= 4096; k++) guess[k] = linearToSrgbByte(k / 4096)
  return (srgbTable = { steps, guess })
}

/** {@link linearToSrgbByte} of every value of a list, by its table of steps: the same bytes, several times faster. */
export function linearToSrgbBytes(values: ArrayLike<number>): Uint8Array {
  const { steps, guess } = srgbSteps()
  const out = new Uint8Array(values.length)
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    let b = 0
    if (v >= 1) b = 255
    else if (v > 0) {
      b = guess[(v * 4096) | 0]
      while (v >= steps[b + 1]) b++
    }
    out[i] = b
  }
  return out
}

/**
 * The picture that `samples` pictures added up in linear light average to: RGBA floats in (each colour the sum of the
 * samples' linear values; the alpha is not read), RGBA sRGB bytes out, opaque: `linearToSrgbByte(sum / samples)`.
 */
export function meanSrgbBytes(sum: ArrayLike<number>, samples: number): Uint8ClampedArray {
  const { steps, guess } = srgbSteps()
  const out = new Uint8ClampedArray(sum.length)
  for (let i = 0; i + 3 < sum.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const v = sum[i + c] / samples
      let b = 0
      if (v >= 1) b = 255
      else if (v > 0) {
        b = guess[(v * 4096) | 0]
        while (v >= steps[b + 1]) b++
      }
      out[i + c] = b
    }
    out[i + 3] = 255
  }
  return out
}

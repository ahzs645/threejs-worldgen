// The PGMaps far forest: what stands past the scatter's own stems (the road view's 600 m stand).
//
//   single trees     real-density cards from the scatter's stems out to 2.5 km (CARD_BAND), on the package's closure,
//                    species mix and heights.
//   stand impostors  one slab per 30 m cell of forest out to the dissolve distance: a 34 m disc of real-density trees
//                    (real crowns, about 3.7 m, at the stand's crown closure) baked from 7 angles to its ground, drawn
//                    camera-facing at the disc's front edge with the angle its own ground and the eye make. This
//                    replaces the far stand's single stems, each of which stood for 400–1,000 trees and was widened
//                    to 55–58 m to hold the closure: squat blobs with no spires and no gaps.
//   canopy surface   past the dissolve distance (VNS's rule: a tree under `canopyCutoffPx` pixels tall), the terrain
//                    raised by the stand's height where the closure says forest, coloured by species mix and closure,
//                    mottled at crown scale with vertical streaks (spires), lit by its own slope and shaded on the side
//                    away from the sun.
//   cards            single trees: the package's far stems that stand on open ground (retention, riparian strips,
//                    where no slab stands), and in a `final` capture the frame's real-density stand.
//
// Every sprite (single trees and baked patches) is generated (pgmaps-far-forest-math.ts), never an asset, and keeps its
// alpha-tested coverage at every mip level (Castaño). Slabs and cards carry the scatter's instance attributes
// (`scatterPosition`, `scatterParams`), so the capture's class pass codes them by the ground under them like any stem;
// the canopy surface takes the class of the ground under it.
import * as THREE from 'three/webgpu'
import {
  atan,
  attribute,
  cameraPosition,
  cameraViewMatrix,
  clamp,
  cos,
  distance,
  dot,
  float,
  floor,
  fwidth,
  hash,
  length,
  max,
  mix,
  mod,
  mx_noise_float,
  normalize,
  positionGeometry,
  positionWorld,
  screenCoordinate,
  select,
  sin,
  smoothstep,
  sqrt,
  step,
  texture,
  uniform,
  uniformArray,
  uv,
  vec2,
  vec3,
  vec4
} from 'three/tsl'
import {
  bakePatch,
  canopyCutoffMeters,
  cappedWidthScale,
  CARD_ASPECT,
  coverageMips,
  finalStandKey,
  median,
  PATCH_CLOSURES,
  PATCH_ELEVATIONS,
  PATCH_HEIGHT,
  PATCH_RADIUS,
  PATCH_SPACING,
  patchRise,
  patchStems,
  rasterTree,
  rng,
  SILHOUETTE_CROWN_RATIO,
  SNAG_SLOT,
  syntheticStand,
  SPECIES_ORDER,
  speciesSlot,
  type SpriteKind,
  type SpriteRaster
} from './pgmaps-far-forest-math'
import { snagForm, standingSnagShare } from './pgmaps-burn'
import type { Ground } from './pgmaps-ground'
import { coverSampler, packageCanopySampler, type CoverGrid } from './pgmaps-cover'
import {
  canopyClosure,
  closureCap,
  coverClassSampler,
  keepStem,
  rockStands,
  stemHash,
  thinShare,
  type CoverClassGrid
} from './pgmaps-nature'

/** Every sprite family the atlases hold: the species, then the snag ({@link SNAG_SLOT}). */
const SPRITE_KINDS: readonly SpriteKind[] = [...SPECIES_ORDER, 'snag']

export type ForestLook = 'via' | 'photo'

/** Slabs stand out to here from the package origin, metres; past it (or the dissolve distance) the canopy surface. */
export const SLAB_MAX = 12_000
/**
 * Between the scatter's stems and here (metres from the package origin) the forest is single trees at real density
 * (cards), not slabs: a patch is 20–90 px across there, and reads as a block.
 */
export const CARD_BAND = 2500
/** Stems per hectare of a closed stand in the card band (real crowns; scaled by the closure). */
export const CARD_BAND_DENSITY = 650
/** The card band's stems are capped at this many (a very forested 360° view). */
const CARD_BAND_MAX = 450_000
/** Crown closure assumed where the package gives none and the ground is neither cut nor water (a regional stand). */
const DEFAULT_CLOSURE = 0.45
/** No slab or canopy where the closure is under this. */
const MIN_CLOSURE = 0.1
/** The canopy surface's top over the ground, as a share of the stand's height (cones: the mean top is below the tips). */
const CANOPY_RAISE = 0.78
/** The canopy surface's colour against a lit crown's (the stand's shade averaged in). */
const CANOPY_TONE = 0.55
/** Pixels a median tree is tall at the dissolve distance, by default (VNS's 4 px). */
export const DEFAULT_CUTOFF_PX = 4
/** The live dissolve distance is kept within these, metres. */
const LIVE_CUTOFF_MIN = 1500

/**
 * sRGB colours of each species' foliage (`SPECIES_ORDER`), then the snag's wood ({@link SNAG_SLOT}), for the photo
 * look and the VIA convention.
 */
// Grey-greens: distant conifer foliage is darker and far less saturated than a leaf seen close (photo far forest
// reads #4a6263 with the haze, the VIA render's #2c3728 without). Snags: charred grey-black wood; the VIA renders
// draw a burnt stand's stems a dark purple-grey (#2f1f3a in the Gun Lake CP007 render).
const SPECIES_PHOTO = ['#5b6356', '#525a55', '#4c5550', '#575f55', '#676e5e', '#6a7061', '#4a4846']
// The VIA renders' canopy: dark olive, about #2a3a2a overall with its shade.
const SPECIES_VIA = ['#555f4f', '#4b5449', '#465045', '#505a4c', '#5d6652', '#606854', '#5c4268']

/** The package fields the far forest reads. */
export interface FarForestPackage {
  trees: { fields: string[]; species: string[]; count: number; data: number[] }
  species: Record<string, { crownBase: number; crownRatio?: number; broadleaf: boolean } | undefined>
  terrain: CoverGrid & CoverClassGrid
  farTerrain?: (CoverGrid & CoverClassGrid) | null
  skylineTerrain?: (CoverGrid & CoverClassGrid) | null
  treeRadiusMeters?: number
}

/** A real-density stand for a `final` capture (the request's `stand`). */
export interface FinalStand {
  fields: readonly string[]
  species: readonly string[]
  count: number
  data: Float32Array | number[]
  radiusMeters: number
  bearings: [number, number]
}

export interface FarForest {
  /** Slabs, cards and the canopy surface; add to the scene. */
  object: THREE.Group
  /** Meshes whose instances carry `scatterPosition` / `scatterParams` (class codes in their yaw), for the capture. */
  coded(): THREE.Mesh[]
  /**
   * Crown closure at a package point as the far forest reads it (the package's, a regional stand where unknown), before
   * the cover code's cap: a stand generated on it is thinned by the class where it is drawn (`showFinal`).
   */
  closureAt(x: number, y: number): number
  /** The canopy surface, for the capture's screened pass. */
  canopy: THREE.Mesh | null
  /** Slabs drawn (stand patches), single-tree cards from the package, and the stand's median height. */
  stats: {
    patches: number
    cards: number
    medianHeight: number
    packageStems: number
    /** Whether the package's cover codes thinned the stand (none on rock and water, fewer on open ground). */
    coverCodes: boolean
    /** Card-band stems the cover codes left out. */
    thinned: number
    /** Single cards drawn as snags on burnt ground (cover 7). */
    snags: number
    /** The final stand's stems the cover codes left out (the last `showFinal`). */
    finalThinned: number
  }
  setLook(look: ForestLook): void
  /** The dissolve distance in force, metres from the camera. */
  readonly cutoff: number
  setCutoff(metres: number): void
  /** Holds the dissolve distance (a capture's), or releases it to the live view (`null`). */
  pin(metres: number | null): void
  /** Whether a capture holds the dissolve distance. */
  readonly pinned: boolean
  /** The dissolve distance for a frame `heightPx` pixels tall with this vertical field (`px`: VNS's threshold). */
  cutoffFor(heightPx: number, verticalFovDegrees: number, px?: number, height?: number): number
  /** Once a frame: the sun, for the canopy's shaded side. */
  update(sun: THREE.Vector3): void
  /**
   * Draws a `final` stand: its stems past `inner` metres from `eye` (package metres) as cards, and hides the slabs and
   * package cards inside its reach. Returns the stems placed and their median height. `groundAt` seats them.
   */
  showFinal(stand: FinalStand, eye: readonly number[], inner: number, groundAt: (x: number, y: number) => number): { stems: number; medianHeight: number }
  clearFinal(): void
  dispose(): void
}

// ---------------------------------------------------------------- the generated atlases (built once per page)

interface Atlas {
  texture: THREE.DataTexture
  columns: number
  rows: number
}

const TREE_CELL = [64, 128] as const
const TREE_VARIANTS = 2
let treeAtlas: Atlas | null = null
let treeSprites: SpriteRaster[][] | null = null

/** Each species' (and the snag's) card sprites, for the patch bakes (narrower raster than the atlas cells, same shape). */
function spritesFor(): SpriteRaster[][] {
  if (!treeSprites)
    treeSprites = SPRITE_KINDS.map((name, s) =>
      [1, 2, 3, 4].map((k) => rasterTree(name, 24, 96, CARD_ASPECT, s * 31 + k * 7, 2))
    )
  return treeSprites
}

function atlasTexture(levels: Array<{ data: Uint8Array; width: number; height: number }>): THREE.DataTexture {
  const t = new THREE.DataTexture(levels[0].data, levels[0].width, levels[0].height, THREE.RGBAFormat, THREE.UnsignedByteType)
  t.mipmaps = levels.map((l) => ({ data: l.data, width: l.width, height: l.height })) as unknown as THREE.DataTexture['mipmaps']
  t.generateMipmaps = false
  t.minFilter = THREE.LinearMipmapLinearFilter
  t.magFilter = THREE.LinearFilter
  t.colorSpace = THREE.NoColorSpace
  t.needsUpdate = true
  return t
}

/** Single-tree cards: each species and the snag × 2 variants, 64 × 128 cells, coverage-preserving mips. */
function cardAtlas(): Atlas {
  if (treeAtlas) return treeAtlas
  const columns = 4
  const cells = SPRITE_KINDS.length * TREE_VARIANTS
  const rows = Math.ceil(cells / columns)
  const [cw, ch] = TREE_CELL
  const atlas: SpriteRaster = { data: new Uint8Array(columns * cw * rows * ch * 4), width: columns * cw, height: rows * ch }
  SPRITE_KINDS.forEach((name, s) => {
    for (let v = 0; v < TREE_VARIANTS; v++) {
      const k = s * TREE_VARIANTS + v
      const cell = rasterTree(name, cw, ch, CARD_ASPECT, s * 31 + v * 7 + 1, 3)
      blitInto(atlas, cell, k % columns, Math.floor(k / columns))
    }
  })
  treeAtlas = { texture: atlasTexture(coverageMips(atlas, cw, ch, 0.5, 4)), columns, rows }
  return treeAtlas
}

const PATCH_CELL = 64
let patchAtlasCache: Atlas | null = null

/**
 * Baked stand patches: columns by elevation (`PATCH_ELEVATIONS`), rows by species (then the snag) × closure
 * (`PATCH_CLOSURES`).
 */
function patchAtlas(): Atlas {
  if (patchAtlasCache) return patchAtlasCache
  const columns = PATCH_ELEVATIONS.length
  const rows = SPRITE_KINDS.length * PATCH_CLOSURES.length
  const atlas: SpriteRaster = {
    data: new Uint8Array(columns * PATCH_CELL * rows * PATCH_CELL * 4),
    width: columns * PATCH_CELL,
    height: rows * PATCH_CELL
  }
  const sprites = spritesFor()
  SPRITE_KINDS.forEach((name, s) =>
    PATCH_CLOSURES.forEach((closure, c) => {
      const stems = patchStems(name, closure, 1000 + s * 17 + c)
      PATCH_ELEVATIONS.forEach((elevation, e) => {
        const cell = bakePatch(stems, name, sprites[s], elevation, PATCH_CELL, PATCH_CELL)
        blitInto(atlas, cell, e, s * PATCH_CLOSURES.length + c)
      })
    })
  )
  patchAtlasCache = { texture: atlasTexture(coverageMips(atlas, PATCH_CELL, PATCH_CELL, 0.5, 4)), columns, rows }
  return patchAtlasCache
}

function blitInto(atlas: SpriteRaster, cell: SpriteRaster, column: number, row: number) {
  for (let y = 0; y < cell.height; y++)
    atlas.data.set(
      cell.data.subarray(y * cell.width * 4, (y + 1) * cell.width * 4),
      ((row * cell.height + y) * atlas.width + column * cell.width) * 4
    )
}

/** Frees the generated atlases (page unload). */
export function disposeFarForestAtlases(): void {
  treeAtlas?.texture.dispose()
  patchAtlasCache?.texture.dispose()
  treeAtlas = null
  patchAtlasCache = null
}

// ---------------------------------------------------------------- shared shader pieces

function lookUniforms() {
  const toLinear = (hex: string) => new THREE.Color(hex)
  return {
    via: uniform(1),
    photo: uniformArray(SPECIES_PHOTO.map(toLinear), 'color'),
    viaColours: uniformArray(SPECIES_VIA.map(toLinear), 'color'),
    cutoff: uniform(8000),
    sun: uniform(new THREE.Vector3(0.3, 0.8, 0.2).normalize()),
    // The final stand's reach (three.js x, z of the eye; bearing from, width (radians); inner and outer radius).
    finalOn: uniform(0),
    finalEye: uniform(new THREE.Vector2()),
    finalFrom: uniform(0),
    finalWidth: uniform(0),
    finalInner: uniform(0),
    finalRadius: uniform(0)
  }
}
type Looks = ReturnType<typeof lookUniforms>

// biome-ignore lint/suspicious/noExplicitAny: TSL nodes
type N = any

/** A species' foliage colour in the look in force. */
const speciesColour = (u: Looks, index: N): N =>
  mix(u.photo.element(index) as N, u.viaColours.element(index) as N, u.via)

/** Inside the final stand's reach (three.js x, z of a point). */
const inFinal = (u: Looks, x: N, z: N): N => {
  const dx = x.sub(u.finalEye.x)
  const dy = z.sub(u.finalEye.y).negate()
  const r = length(vec2(dx, dy))
  const bearing = atan(dx, dy) // clockwise from north, −π…π
  const off = mod(bearing.sub(u.finalFrom).add(Math.PI * 4), Math.PI * 2)
  return u.finalOn
    .greaterThan(0.5)
    .and(off.lessThanEqual(u.finalWidth))
    .and(r.greaterThanEqual(u.finalInner))
    .and(r.lessThanEqual(u.finalRadius))
}

/**
 * The lit sprite: albedo × species colour × tint, and the crown's own normal from the atlas (x across the card, y up,
 * the rest toward the eye) turned into view space, so the scene's sun lights one side of each crown and not the other.
 */
function spriteNodes(u: Looks, sample: N, right: N, toward: N, colour: N): { normal: N; color: N } {
  const nx = sample.g.mul(2).sub(1)
  const ny = sample.b.mul(2).sub(1)
  const nz = sqrt(max(float(1).sub(nx.mul(nx)).sub(ny.mul(ny)), 0.04))
  const n: N = normalize(right.mul(nx).add(vec3(0, 1, 0).mul(ny)).add(toward.mul(nz)))
  // A crown's sun side and shade side, across the card: the skylight lights both nearly alike, so the side away from
  // the sun is darkened here as its own crown shades it. With the sun behind the eye or ahead of it, both sides take
  // the mean.
  const sunAcross = dot(normalize(vec2(u.sun.x, u.sun.z).add(vec2(1e-4, 0))), vec2(right.x, right.z))
  const side = clamp(nx.mul(sunAcross).mul(0.55).add(0.8), 0.3, 1.25)
  return {
    normal: cameraViewMatrix.mul(vec4(n, 0)).xyz.normalize(),
    color: vec4(colour.mul(sample.r).mul(side), sample.a)
  }
}

/** A unit quad, x −0.5…0.5, y 0…1, uv (0, 0) at the bottom left; drawn instanced. */
function quad(): THREE.InstancedBufferGeometry {
  const g = new THREE.InstancedBufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute([-0.5, 0, 0, 0.5, 0, 0, 0.5, 1, 0, -0.5, 1, 0], 3))
  g.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2))
  g.setIndex([0, 1, 2, 0, 2, 3])
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e9)
  return g
}

// ---------------------------------------------------------------- stand patches (slabs)

/**
 * The slab material. Per instance: `scatterPosition` (x, ground at the centre, z, stand height), `scatterParams`
 * (yaw with the class code, tint, random, 0), `slabInfo` (ground slope dh/dx and dh/dz, atlas row, 0).
 */
function slabMaterial(u: Looks, atlas: Atlas): THREE.MeshStandardNodeMaterial {
  const aPos = attribute('scatterPosition', 'vec4')
  const aPar = attribute('scatterParams', 'vec4')
  const aInfo = attribute('slabInfo', 'vec4')
  const centre = aPos.xyz
  const toEye = cameraPosition.sub(centre)
  const flat = vec2(toEye.x, toEye.z)
  const dist = max(length(flat), 1)
  const toward2 = flat.div(dist)
  const toward = vec3(toward2.x, 0, toward2.y)
  // Screen right seen from the eye: cross(−toward, up).
  const right = vec3(toward2.y, 0, toward2.x.negate())
  const grad = aInfo.xy
  // The angle between the eye's ray and the patch's ground: the eye's elevation plus the slope rising away from it.
  const rise = dot(grad, toward2).negate()
  const angle = clamp(atan(toEye.y.div(dist)).add(atan(rise)), 0, (PATCH_ELEVATIONS[PATCH_ELEVATIONS.length - 1] * Math.PI) / 180)
  const degrees = angle.mul(180 / Math.PI)
  // The nearest baked elevation.
  let column: N = float(0)
  for (let i = 0; i < PATCH_ELEVATIONS.length - 1; i++)
    column = column.add(step((PATCH_ELEVATIONS[i] + PATCH_ELEVATIONS[i + 1]) / 2, degrees))
  // The baked angle's rise over the patch's depth (`patchRise`): the trees stand their full height on it.
  const rises = uniformArray(PATCH_ELEVATIONS.map((e) => patchRise(2 * PATCH_RADIUS, e)), 'float')
  const span = (rises.element(column.toInt()) as N).add(aPos.w)
  // Drawn at the patch's front edge, so none of its own ground stands in front of it.
  const base = vec3(
    centre.x.add(toward2.x.mul(PATCH_RADIUS)),
    centre.y.add(dot(grad, toward2).mul(PATCH_RADIUS)).sub(0.8),
    centre.z.add(toward2.y.mul(PATCH_RADIUS))
  )
  // Dissolved past the cut-off (each slab at its own fraction of the last tenth), and inside a final stand's reach.
  const cut = u.cutoff.mul(aPar.z.mul(0.1).add(0.9))
  const hidden = dist.greaterThan(cut).or(inFinal(u, centre.x, centre.z))
  const size = select(hidden, float(0), float(1))
  const material = new THREE.MeshStandardNodeMaterial()
  material.positionNode = base
    .add(right.mul(positionGeometry.x.mul(2 * PATCH_RADIUS)))
    .add(vec3(0, 1, 0).mul(positionGeometry.y.mul(span)))
    .mul(size)
  const cell = vec2(column, aInfo.z)
  const atlasUv = vec2(cell.x.add(uv().x).div(atlas.columns), cell.y.add(float(1).sub(uv().y)).div(atlas.rows))
  const sample = texture(atlas.texture, atlasUv)
  const species = floor(aInfo.z.div(PATCH_CLOSURES.length)).toInt()
  const shaded = spriteNodes(u, sample, right, toward, speciesColour(u, species).mul(aPar.y))
  material.normalNode = shaded.normal
  material.colorNode = shaded.color
  material.alphaTest = 0.5
  material.roughness = 0.92
  material.metalness = 0
  material.name = 'far forest: stand patches'
  return material
}

// ---------------------------------------------------------------- single-tree cards

/**
 * The card material. Per instance: `scatterPosition` (x, ground, z, height), `scatterParams` (yaw with the class code,
 * tint, random, species × 4 + width scale). Cards stand vertical, turned to the eye about their stem; past the
 * cut-off they dissolve into the canopy like the slabs. `final`: the final stand's cards (never hidden by its reach).
 */
function cardMaterial(u: Looks, atlas: Atlas, final: boolean): THREE.MeshStandardNodeMaterial {
  const aPos = attribute('scatterPosition', 'vec4')
  const aPar = attribute('scatterParams', 'vec4')
  const base = aPos.xyz
  const toEye = cameraPosition.sub(base)
  const flat = vec2(toEye.x, toEye.z)
  const dist = max(length(flat), 0.01)
  const toward2 = flat.div(dist)
  const toward = vec3(toward2.x, 0, toward2.y)
  const right = vec3(toward2.y, 0, toward2.x.negate())
  const species = floor(aPar.w.div(4))
  const widthScale = aPar.w.sub(species.mul(4))
  const height = aPos.w
  const cut = u.cutoff.mul(aPar.z.mul(0.1).add(0.9))
  const hidden = final ? dist.greaterThan(cut) : dist.greaterThan(cut).or(inFinal(u, base.x, base.z))
  const size = select(hidden, float(0), float(1))
  const material = new THREE.MeshStandardNodeMaterial()
  material.positionNode = base
    .add(right.mul(positionGeometry.x.mul(height.mul(CARD_ASPECT).mul(widthScale))))
    .add(vec3(0, 1, 0).mul(positionGeometry.y.mul(height)))
    .sub(vec3(0, 0.3, 0))
    .mul(size)
  const variant = floor(aPar.z.mul(TREE_VARIANTS - 0.001))
  const k = species.mul(TREE_VARIANTS).add(variant)
  const column = mod(k, atlas.columns)
  const row = floor(k.div(atlas.columns))
  const atlasUv = vec2(column.add(uv().x).div(atlas.columns), row.add(float(1).sub(uv().y)).div(atlas.rows))
  const sample = texture(atlas.texture, atlasUv)
  const shaded = spriteNodes(u, sample, right, toward, speciesColour(u, species.toInt()).mul(aPar.y))
  material.normalNode = shaded.normal
  material.colorNode = shaded.color
  material.alphaTest = 0.5
  material.roughness = 0.92
  material.metalness = 0
  material.name = final ? 'far forest: final stand' : 'far forest: single stems'
  return material
}

/** A card mesh from a list of stems: x, y (package), ground, height, crown, species slot, tone. */
function cardMesh(stems: Float32Array, count: number, material: THREE.Material, name: string): THREE.Group {
  const position = new Float32Array(count * 4)
  const params = new Float32Array(count * 4)
  for (let n = 0; n < count; n++) {
    const o = n * 7
    const x = stems[o]
    const y = stems[o + 1]
    const ground = stems[o + 2]
    const height = stems[o + 3]
    const crown = stems[o + 4]
    const slot = stems[o + 5]
    const tone = stems[o + 6]
    const species = SPRITE_KINDS[slot] ?? 'spruce'
    const ratio = height > 0 ? crown / height : SILHOUETTE_CROWN_RATIO[species]
    const width = cappedWidthScale(ratio / SILHOUETTE_CROWN_RATIO[species], 1)
    const random = fract01(x * 12.9898 + y * 78.233)
    const p = n * 4
    position[p] = x
    position[p + 1] = ground
    position[p + 2] = -y
    position[p + 3] = height
    params[p] = random * Math.PI * 2
    params[p + 1] = 0.88 + 0.24 * (Number.isFinite(tone) ? tone : random)
    params[p + 2] = random
    params[p + 3] = slot * 4 + width
  }
  return tiled({ scatterPosition: position, scatterParams: params }, count, material, name)
}

/** Side of the square tiles instances are grouped into, metres: each tile is one draw, culled to the view on its own. */
const TILE = 1000

/**
 * Instanced quads grouped into {@link TILE} tiles by their `scatterPosition` (x, ground, z, height; three.js axes), one
 * mesh per tile with a bounding sphere that holds it, so the live view (about a sixth of the circle) and the lake's
 * mirrored view draw only the tiles they see. Every attribute is 4 floats an instance; the material is shared.
 */
function tiled(attributes: Record<string, Float32Array>, count: number, material: THREE.Material, name: string): THREE.Group {
  const position = attributes.scatterPosition
  // Each instance's tile, tiles numbered in the order they are first met (so the meshes, and the instances within
  // each, keep the order of the list). A tile is keyed by a number, not by text: a final stand is 1.4 million cards.
  const tiles = new Map<number | string, number>()
  const tileOf = new Int32Array(count)
  const tileX: number[] = []
  const tileZ: number[] = []
  const sizes: number[] = []
  for (let n = 0; n < count; n++) {
    const tx = Math.floor(position[n * 4] / TILE)
    const tz = Math.floor(position[n * 4 + 2] / TILE)
    const near = tx >= -32768 && tx < 32768 && tz >= -32768 && tz < 32768
    const key = near ? (tx + 32768) * 65536 + (tz + 32768) : `${tx},${tz}`
    let t = tiles.get(key)
    if (t === undefined) {
      tiles.set(key, (t = sizes.length))
      tileX.push(tx)
      tileZ.push(tz)
      sizes.push(0)
    }
    tileOf[n] = t
    sizes[t]++
  }
  const filled = Object.entries(attributes).map(([attribute, data]) => {
    const outs = sizes.map((size) => new Float32Array(size * 4))
    const next = new Int32Array(sizes.length)
    for (let n = 0; n < count; n++) {
      const t = tileOf[n]
      const out = outs[t]
      const k = next[t]++ * 4
      const o = n * 4
      out[k] = data[o]
      out[k + 1] = data[o + 1]
      out[k + 2] = data[o + 2]
      out[k + 3] = data[o + 3]
    }
    return { attribute, outs }
  })
  const group = new THREE.Group()
  group.name = name
  for (let t = 0; t < sizes.length; t++) {
    const tx = tileX[t]
    const tz = tileZ[t]
    const key = `${tx},${tz}`
    const size = sizes[t]
    const geometry = quad()
    let low = Infinity
    let high = -Infinity
    for (const { attribute, outs } of filled) {
      const out = outs[t]
      geometry.setAttribute(attribute, new THREE.InstancedBufferAttribute(out, 4))
      if (attribute === 'scatterPosition')
        for (let k = 0; k < size; k++) {
          low = Math.min(low, out[k * 4 + 1])
          high = Math.max(high, out[k * 4 + 1] + out[k * 4 + 3])
        }
    }
    geometry.instanceCount = size
    // The tile, its quads' reach past its edge (a patch's front edge and width, a card's crown), and its heights.
    const centre = new THREE.Vector3((tx + 0.5) * TILE, (low + high) / 2 + 40, (tz + 0.5) * TILE)
    geometry.boundingSphere = new THREE.Sphere(centre, Math.hypot(TILE * 0.71 + 60, (high - low) / 2 + 160))
    const mesh = new THREE.Mesh(geometry, material)
    mesh.name = `${name} ${key}`
    mesh.frustumCulled = true
    mesh.castShadow = false
    mesh.receiveShadow = false
    mesh.matrixAutoUpdate = false
    group.add(mesh)
  }
  return group
}

/** The meshes of a tiled group (or a single mesh). */
const meshesOf = (o: THREE.Object3D | null): THREE.Mesh[] => {
  const out: THREE.Mesh[] = []
  o?.traverse((c) => {
    if ((c as THREE.Mesh).isMesh) out.push(c as THREE.Mesh)
  })
  return out
}

/** Frees a tiled group's geometries and its shared material (`keepMaterial`: the material is another's to free). */
function disposeTiled(o: THREE.Object3D | null, keepMaterial = false) {
  const meshes = meshesOf(o)
  for (const m of meshes) m.geometry.dispose()
  if (!keepMaterial) (meshes[0]?.material as THREE.Material | undefined)?.dispose()
  o?.removeFromParent()
}

const fract01 = (v: number) => {
  const s = Math.sin(v) * 43758.5453
  return s - Math.floor(s)
}

// ---------------------------------------------------------------- the canopy surface

/**
 * The canopy surface's material: the closure (`aCanopy.x`) shown as crowns (the stand's species colour, its broadleaf
 * share `aCanopy.z`) over their own shade; crown-scale noise stretched up (spires: about 4 m across, 14 m tall, filtered
 * out once it is sub-pixel), clumps and the stand mosaic; darker on the side away from the sun. Dissolves in past the
 * cut-off (a hashed dither over its last tenth) and nowhere the closure is under {@link MIN_CLOSURE}.
 */
function canopyMaterial(u: Looks): THREE.MeshStandardNodeMaterial {
  const a = attribute('aCanopy', 'vec4')
  const closure = a.x
  const d = distance(positionWorld.xz, cameraPosition.xz)
  const dither = hash(screenCoordinate.x.add(screenCoordinate.y.mul(4099)))
  const visible = closure
    .greaterThan(MIN_CLOSURE)
    .and(d.greaterThan(u.cutoff.mul(dither.mul(0.1).add(0.9))))
  // Colours: the stand's conifers (spruce/fir) or broadleaves, in the look in force.
  const conifer = speciesColour(u, 1)
  const broadleaf = speciesColour(u, 4)
  const crown = mix(conifer, broadleaf, a.z)
  const gap = crown.mul(0.22)
  // Screen-aligned axes at this point: across the view (crowns sit side by side) and up (spires).
  const view = positionWorld.xz.sub(cameraPosition.xz)
  const across = normalize(vec2(view.y.negate(), view.x))
  const x = dot(positionWorld.xz, across)
  const footprint = max(fwidth(x), 1e-3)
  const keep = float(1).sub(smoothstep(1.2, 3.5, footprint))
  const spires = mx_noise_float(vec3(x.div(3.8), positionWorld.y.div(13), 0.5)).mul(keep)
  const clumps = mx_noise_float(vec3(positionWorld.xz.div(32), 1.5))
  const mosaic = mx_noise_float(vec3(positionWorld.xz.div(140), 2.5))
  // The share of the view that is lit crown: with the closure, mottled; its mean where the mottle is filtered out.
  const shown = smoothstep(0.05, 0.75, closure).mul(0.55).add(0.35)
  const mottle = spires.mul(0.55).add(clumps.mul(0.3)).add(mosaic.mul(0.2))
  const crowns = clamp(shown.add(mottle.mul(0.6)), 0, 1)
  // The side of the crowns away from the sun: looking into the sun the eye sees their shade, with it behind them their
  // lit faces (the hotspot).
  const toCamera = normalize(vec2(view.x.negate(), view.y.negate()))
  const sunFlat = normalize(vec2(u.sun.x, u.sun.z).add(vec2(1e-4, 0)))
  const hotspot = mix(float(0.6), float(1), dot(toCamera, sunFlat).mul(0.5).add(0.5))
  // Calibrated against the stand patches at the dissolve distance (their mean, crowns and the shade between them), so
  // the switch shows no step: a patch's pixels average about half its lit crown colour.
  const colour: N = mix(gap, crown, crowns).mul(mosaic.mul(0.12).add(1)).mul(hotspot).mul(CANOPY_TONE)
  const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.95, metalness: 0 })
  material.colorNode = vec4(colour, select(visible, float(1), float(0)))
  material.alphaTest = 0.5
  material.name = 'far forest: canopy surface'
  return material
}

/**
 * The canopy mesh: square rings round the package origin (40 m posts to 8 km, 100 m to 20 km, 250 m beyond, each
 * overlapping the next by a post), the ground raised by {@link CANOPY_RAISE} × the stand height where the closure is
 * forest, sunk under the ground elsewhere. Starts at 1 km (the dissolve distance is never nearer).
 */
function canopyGeometry(
  closureAt: (x: number, y: number) => number,
  heightAt: (x: number, y: number) => number,
  broadleafAt: (x: number, y: number) => number,
  groundAt: (x: number, y: number) => number,
  extent: number
): THREE.BufferGeometry {
  const rings: Array<[from: number, to: number, step: number]> = []
  const plan: Array<[number, number, number]> = [
    [1000, 8000, 40],
    [8000, 20000, 100],
    [20000, 60000, 250]
  ]
  for (const [from, to, step] of plan) {
    if (from >= extent) break
    rings.push([from, Math.min(to, extent), step])
  }
  const positions: number[] = []
  const canopy: number[] = []
  const indices: number[] = []
  rings.forEach(([from, to, step], k) => {
    const next = rings[k + 1]
    const outer = to + (next ? next[2] : 0)
    const n = Math.ceil((2 * outer) / step)
    const first = positions.length / 3
    const half = (n * step) / 2
    for (let r = 0; r <= n; r++)
      for (let c = 0; c <= n; c++) {
        const x = -half + c * step
        const y = -half + r * step
        const raw = closureAt(x, y)
        const closure = Number.isFinite(raw) ? raw : 0
        const h = heightAt(x, y)
        const raise = closure > MIN_CLOSURE ? h * CANOPY_RAISE * Math.min(1, (closure - MIN_CLOSURE) / 0.2 + 0.35) : -3
        positions.push(x, groundAt(x, y) + raise, -y)
        canopy.push(closure, h, broadleafAt(x, y), 0)
      }
    const inner = from - step
    for (let r = 0; r < n; r++)
      for (let c = 0; c < n; c++) {
        const x0 = -half + c * step
        const y0 = -half + r * step
        if (x0 >= -inner && x0 + step <= inner && y0 >= -inner && y0 + step <= inner) continue
        const a = first + r * (n + 1) + c
        const b = a + 1
        const d = a + n + 1
        const e = d + 1
        // Counter-clockwise seen from above (row r + 1 is north, three.js −z), so the faces and normals point up.
        indices.push(a, b, e, a, e, d)
      }
  })
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  g.setAttribute('aCanopy', new THREE.Float32BufferAttribute(canopy, 4))
  g.setIndex(indices)
  g.computeVertexNormals()
  g.computeBoundingSphere()
  return g
}

// ---------------------------------------------------------------- the stand's mix, by area

const MIX_CELL = 240

/** Species shares and mean height of the package's stems per 240 m cell, with the whole package's as a fallback. */
function standMix(pkg: FarForestPackage) {
  const fields = pkg.trees.fields
  const stride = fields.length
  const [ix, iy, is, ih] = ['x', 'y', 'species', 'height'].map((f) => fields.indexOf(f))
  const cells = new Map<number, { counts: number[]; height: number; n: number }>()
  const total = new Array(SPECIES_ORDER.length).fill(0)
  const heights: number[] = []
  const key = (c: number, r: number) => (c + 4096) * 8192 + (r + 4096)
  const slotOf = pkg.trees.species.map((name) => speciesSlot(name))
  for (let i = 0; i < pkg.trees.count; i++) {
    const o = i * stride
    const x = pkg.trees.data[o + ix]
    const y = pkg.trees.data[o + iy]
    const slot = slotOf[pkg.trees.data[o + is]] ?? 1
    const h = pkg.trees.data[o + ih]
    if (!Number.isFinite(h)) continue
    const k = key(Math.floor(x / MIX_CELL), Math.floor(y / MIX_CELL))
    let cell = cells.get(k)
    if (!cell) cells.set(k, (cell = { counts: new Array(SPECIES_ORDER.length).fill(0), height: 0, n: 0 }))
    cell.counts[slot]++
    cell.height += h
    cell.n++
    total[slot]++
    if (i % 7 === 0 || pkg.trees.count < 20000) heights.push(h)
  }
  const medianHeight = Number.isFinite(median(heights)) ? median(heights) : PATCH_HEIGHT
  const all = { counts: total.some((v) => v > 0) ? total : [1, 3, 2, 0, 0, 0], height: medianHeight, n: 1 }
  return {
    medianHeight,
    at(x: number, y: number): { counts: number[]; height: number } {
      const c = Math.floor(x / MIX_CELL)
      const r = Math.floor(y / MIX_CELL)
      const own = cells.get(key(c, r))
      if (own && own.n >= 4) return { counts: own.counts, height: own.height / own.n }
      const counts = new Array(SPECIES_ORDER.length).fill(0)
      let h = 0
      let n = 0
      for (let dr = -1; dr <= 1; dr++)
        for (let dc = -1; dc <= 1; dc++) {
          const cell = cells.get(key(c + dc, r + dr))
          if (!cell) continue
          cell.counts.forEach((v, k) => (counts[k] += v))
          h += cell.height
          n += cell.n
        }
      if (n >= 4) return { counts, height: h / n }
      return { counts: all.counts, height: all.height }
    }
  }
}

// ---------------------------------------------------------------- build

/**
 * Builds the far forest of a package: real-density cards from `nearRadius` (the scatter's stems) to CARD_BAND, slabs on
 * every forested 30 m cell from there out to
 * {@link SLAB_MAX}, single cards for the package's own far stems (`farStems`, indices) that stand where no slab does,
 * and the canopy surface out to the terrain's edge.
 */
export function buildFarForest(
  pkg: FarForestPackage,
  ground: Ground,
  nearRadius: number,
  farStems: readonly number[]
): FarForest {
  const u = lookUniforms()
  const object = new THREE.Group()
  object.name = 'far forest'
  const cover = coverSampler(pkg as Parameters<typeof coverSampler>[0])
  const extentOf = ground.skylineExtent ?? ground.farExtent ?? ground.drawnExtent
  const extent = Math.min(-extentOf.west, extentOf.east, -extentOf.south, extentOf.north)
  const inExtent = (x: number, y: number) => x >= extentOf.west && x <= extentOf.east && y >= extentOf.south && y <= extentOf.north
  /** Crown closure, with a regional stand where the package says nothing and the ground is not cut or water. */
  const closureAt = (x: number, y: number) => {
    if (!inExtent(x, y) || ground.inWater(x, y)) return 0
    const c = cover.at(x, y)
    if (Number.isFinite(c)) return c
    return ground.isCut(x, y) ? 0 : DEFAULT_CLOSURE
  }
  // The ground's cover codes (pgmaps-nature.ts): no trees on rock or water, fewer on sparse, shrub and herb ground.
  const classAt = coverClassSampler([pkg.terrain, pkg.farTerrain, pkg.skylineTerrain])
  // Rock under a standing stand (the package's own closure there, `rockStands`) keeps its stand.
  const packageCanopy = packageCanopySampler(pkg)
  const standingAt = (x: number, y: number) => rockStands(packageCanopy(x, y))
  const capAt = (x: number, y: number) => (classAt ? closureCap(classAt(x, y), standingAt(x, y)) : 1)
  /** The closure the stand is drawn at: the package's under its cover code's cap. */
  const standClosureAt = (x: number, y: number) => Math.min(closureAt(x, y), capAt(x, y))
  /** The share of a stand generated on `closureAt` that stands at a point (thinned stem by stem, `keepStem`). */
  const stemShare = (x: number, y: number) => thinShare(closureAt(x, y), capAt(x, y))
  /**
   * `stemShare` at a point already known to be out of the water, for the final stand's 1.4 million stems: the same
   * figure with the package's own closure read once. It is both what `standingAt` asks and, where the package knows it,
   * the first answer `cover.at` gives (the same grids' `canopy`, in the same order).
   */
  const dryStemShare = (x: number, y: number) => {
    if (!inExtent(x, y)) return thinShare(0, capAt(x, y))
    const own = packageCanopy(x, y)
    const c = Number.isFinite(own) ? own : cover.at(x, y)
    const closure = Number.isFinite(c) ? c : ground.isCut(x, y) ? 0 : DEFAULT_CLOSURE
    return thinShare(closure, classAt ? closureCap(classAt(x, y), rockStands(own)) : 1)
  }
  /**
   * Burnt ground (cover 7, pgmaps-burn.ts): years since the fire, NaN elsewhere. Its stand stands as snags: the stand's
   * own closure (else the regional stand's) reckons how many, thinned as they fall (`standingSnagShare`).
   */
  const burntAge = (x: number, y: number) => ground.burntAge(x, y)
  const snagClosureAt = (x: number, y: number, age: number) => {
    const raw = closureAt(x, y)
    return (raw > MIN_CLOSURE ? raw : DEFAULT_CLOSURE) * standingSnagShare(age)
  }
  /** A stand's stem on burnt ground as a snag (its form, `snagForm`), or null where it has fallen. */
  const asSnag = (x: number, y: number, height: number, crown: number, age: number) => {
    const u = stemHash(x, y)
    if (!keepStem(y, x, standingSnagShare(age))) return null
    return snagForm(height, crown > 0 ? crown : height * 0.2, u)
  }
  let thinned = 0
  let snags = 0
  const mix6 = standMix(pkg)

  // Slabs.
  const patches = patchAtlas()
  const random = rng(0x5eed)
  const pos: number[] = []
  const par: number[] = []
  const info: number[] = []
  const slabReach = Math.min(SLAB_MAX, extent)
  const n = Math.ceil(slabReach / PATCH_SPACING)
  const start = Math.max(nearRadius, CARD_BAND) + PATCH_RADIUS * 0.5
  for (let r = -n; r < n; r++)
    for (let c = -n; c < n; c++) {
      const jitterX = (random() - 0.5) * PATCH_SPACING * 0.5
      const jitterY = (random() - 0.5) * PATCH_SPACING * 0.5
      const pick = random()
      const level = random()
      const keep = random()
      const tint = random()
      const x = (c + 0.5) * PATCH_SPACING + jitterX
      const y = (r + 0.5) * PATCH_SPACING + jitterY
      const d = Math.hypot(x, y)
      if (d < start || d > slabReach) continue
      // Sparse stands: fewer patches (each at the lowest baked closure); else the two baked closures either side,
      // dithered patch by patch so the stand's closure holds on average.
      const kept = (c: number) => c > MIN_CLOSURE && !(c < PATCH_CLOSURES[0] && keep > c / PATCH_CLOSURES[0])
      // On burnt ground the patch is a stand of snags, at the burnt stand's closure.
      const age = burntAge(x, y)
      const burnt = Number.isFinite(age)
      const raw = burnt ? snagClosureAt(x, y, age) : closureAt(x, y)
      if (!kept(raw)) continue
      // A patch the package's closure keeps takes its yaw here, whether or not its cover code then drops it, so
      // dropping patches on rock and open ground never moves the stand's other patches.
      const yaw = random() * Math.PI * 2
      const closure = burnt ? raw : Math.min(raw, capAt(x, y))
      if (!kept(closure)) continue
      let closureLevel = 0
      while (closureLevel < PATCH_CLOSURES.length - 1 && closure > PATCH_CLOSURES[closureLevel + 1]) closureLevel++
      if (closureLevel < PATCH_CLOSURES.length - 1) {
        const lo = PATCH_CLOSURES[closureLevel]
        const t = (closure - lo) / (PATCH_CLOSURES[closureLevel + 1] - lo)
        if (level < t) closureLevel++
      }
      const local = mix6.at(x, y)
      const sum = local.counts.reduce((s, v) => s + v, 0) || 1
      let target = pick * sum
      let species = 0
      while (species < SPECIES_ORDER.length - 1 && target >= local.counts[species]) target -= local.counts[species++]
      if (burnt) species = SNAG_SLOT
      const g = ground.groundAt(x, y)
      const gx = (ground.groundAt(x + 10, y) - ground.groundAt(x - 10, y)) / 20
      const gy = (ground.groundAt(x, y + 10) - ground.groundAt(x, y - 10)) / 20
      pos.push(x, g, -y, Math.max(6, local.height))
      par.push(yaw, 0.86 + 0.28 * tint, keep, 0)
      info.push(gx, -gy, species * PATCH_CLOSURES.length + closureLevel, 0)
    }
  const patchCount = pos.length / 4
  const slabs = tiled(
    { scatterPosition: new Float32Array(pos), scatterParams: new Float32Array(par), slabInfo: new Float32Array(info) },
    patchCount,
    slabMaterial(u, patches),
    'far forest: stand patches'
  )
  object.add(slabs)

  // The package's far stems where no slab stands (open ground: retention, strips), as single cards.
  const cards = cardAtlas()
  const fields = pkg.trees.fields
  const stride = fields.length
  const [ix, iy, is, ih, iw, it] = ['x', 'y', 'species', 'height', 'crownWidth', 'tone'].map((f) => fields.indexOf(f))
  const slotOf = pkg.trees.species.map((name) => speciesSlot(name))
  const single: number[] = []
  for (const i of farStems) {
    const o = i * stride
    const x = pkg.trees.data[o + ix]
    const y = pkg.trees.data[o + iy]
    // `farStems` already leaves out stems on rock and water posts (`ground.onBare`); the rest are the package's own.
    if (ground.inWater(x, y)) continue
    const age = burntAge(x, y)
    if (Number.isFinite(age)) {
      // Burnt: the card band's and the slabs' snags stand for these; past the slabs, each is its own snag.
      if (Math.hypot(x, y) <= slabReach) continue
      const snag = asSnag(x, y, pkg.trees.data[o + ih], iw >= 0 ? pkg.trees.data[o + iw] : 0, age)
      if (!snag) continue
      single.push(x, y, ground.groundAt(x, y), snag.height, snag.crownWidth, SNAG_SLOT, it >= 0 ? pkg.trees.data[o + it] : 0.5)
      snags++
      continue
    }
    if (standClosureAt(x, y) > MIN_CLOSURE) continue
    single.push(
      x,
      y,
      ground.groundAt(x, y),
      pkg.trees.data[o + ih],
      iw >= 0 ? pkg.trees.data[o + iw] : 0,
      slotOf[pkg.trees.data[o + is]] ?? 1,
      it >= 0 ? pkg.trees.data[o + it] : 0.5
    )
  }
  // The card band: real-density single trees from the near stand out to CARD_BAND, on the closure, mix and heights.
  const bandMix = SPECIES_ORDER.map((species, k) => {
    let share = 0
    let height = 0
    let n = 0
    const fieldsAll = pkg.trees.fields
    const st = fieldsAll.length
    const fs = fieldsAll.indexOf('species')
    const fh = fieldsAll.indexOf('height')
    for (let i = 0; i < pkg.trees.count; i += 3) {
      if (slotOf[pkg.trees.data[i * st + fs]] !== k) continue
      share++
      height += pkg.trees.data[i * st + fh]
      n++
    }
    return { species, share, height: n ? height / n : mix6.medianHeight }
  }).filter((m) => m.share > 0)
  if (nearRadius < CARD_BAND) {
    const band = syntheticStand(
      [0, 0, 0],
      [0, 360],
      CARD_BAND,
      (x, y) => {
        // Burnt ground grows the stand it had (its closure, else the regional stand's): those stems stand as snags.
        if (Number.isFinite(burntAge(x, y))) return closureAt(x, y) > MIN_CLOSURE ? closureAt(x, y) : DEFAULT_CLOSURE
        return closureAt(x, y) > MIN_CLOSURE ? closureAt(x, y) : 0
      },
      ground.groundAt,
      bandMix.length ? bandMix : [{ species: 'spruce', share: 1, height: mix6.medianHeight }],
      { inner: nearRadius, density: CARD_BAND_DENSITY, seed: 0xba4d, maxStems: CARD_BAND_MAX }
    )
    const slots = band.species.map((name) => speciesSlot(name))
    for (let i = 0; i < band.count; i++) {
      const o = i * 7
      const age = burntAge(band.data[o], band.data[o + 1])
      if (Number.isFinite(age)) {
        const snag = asSnag(band.data[o], band.data[o + 1], band.data[o + 4], band.data[o + 5], age)
        if (!snag) continue
        single.push(band.data[o], band.data[o + 1], band.data[o + 2], snag.height, snag.crownWidth, SNAG_SLOT, band.data[o + 6])
        snags++
        continue
      }
      if (!keepStem(band.data[o], band.data[o + 1], stemShare(band.data[o], band.data[o + 1]))) {
        thinned++
        continue
      }
      single.push(band.data[o], band.data[o + 1], band.data[o + 2], band.data[o + 4], band.data[o + 5], slots[band.data[o + 3]] ?? 1, band.data[o + 6])
    }
  }
  const singleCards = single.length
    ? cardMesh(new Float32Array(single), single.length / 7, cardMaterial(u, cards, false), 'far forest: single stems')
    : null
  if (singleCards) object.add(singleCards)

  // The canopy surface.
  const heightAt = (x: number, y: number) => Math.max(8, mix6.at(x, y).height)
  const broadleafAt = (x: number, y: number) => {
    const m = mix6.at(x, y).counts
    const sum = m.reduce((s, v) => s + v, 0) || 1
    return (m[4] + m[5]) / sum
  }
  // Raised only where the ground is not open: rock, herb and shrub ground keeps its own height and look.
  const canopyAt = (x: number, y: number) =>
    canopyClosure(closureAt(x, y), classAt ? classAt(x, y) : null, standingAt(x, y))
  const canopy = extent > 1000 ? new THREE.Mesh(canopyGeometry(canopyAt, heightAt, broadleafAt, ground.groundAt, extent), canopyMaterial(u)) : null
  if (canopy) {
    canopy.name = 'far forest: canopy surface'
    canopy.castShadow = false
    canopy.receiveShadow = false
    object.add(canopy)
  }
  console.log(
    `[far forest] ${patchCount} stand patches to ${Math.round(slabReach)} m, ${single.length / 7} single stems (${snags} snags on burnt ground, ${thinned} thinned by cover${classAt ? '' : ', no cover codes'}), canopy ${canopy ? (canopy.geometry.index!.count / 3) : 0} triangles, median height ${mix6.medianHeight.toFixed(1)} m`
  )

  let finalMesh: THREE.Group | null = null
  /**
   * The last final stand placed (its cards, tiled, with their buffers), kept after a capture hides it: the same stand
   * from the same eye (`finalStandKey`) on this ground is shown again without being placed. One stand at a time.
   */
  let finalKept: {
    key: string
    groundAt: (x: number, y: number) => number
    mesh: THREE.Group | null
    stems: number
    medianHeight: number
    dropped: number
  } | null = null
  const forgetFinal = () => {
    if (finalKept?.mesh) disposeTiled(finalKept.mesh, true)
    finalKept = null
  }
  // The final stand's material, made once with the far forest (a kept stand and the next one share it) and drawn from
  // the first frame by a card of no size, so its pipeline is compiled with the scene's and not in the first capture.
  const finalMaterial = cardMaterial(u, cards, true)
  const finalPrimer = cardMesh(new Float32Array(7), 1, finalMaterial, 'far forest: final stand (stand-in)')
  for (const mesh of meshesOf(finalPrimer)) mesh.frustumCulled = false
  object.add(finalPrimer)
  let pinned = false
  const stats = {
    patches: patchCount,
    cards: single.length / 7,
    medianHeight: mix6.medianHeight,
    packageStems: farStems.length,
    coverCodes: !!classAt,
    thinned,
    snags,
    finalThinned: 0
  }
  return {
    object,
    canopy,
    stats,
    closureAt,
    coded: () => [...meshesOf(slabs), ...meshesOf(singleCards), ...meshesOf(finalMesh)],
    setLook(look) {
      u.via.value = look === 'via' ? 1 : 0
    },
    get cutoff() {
      return u.cutoff.value
    },
    setCutoff(metres) {
      if (!pinned) u.cutoff.value = metres
    },
    pin(metres) {
      pinned = metres !== null
      if (metres !== null) u.cutoff.value = metres
    },
    get pinned() {
      return pinned
    },
    cutoffFor(heightPx, vfov, px = DEFAULT_CUTOFF_PX, height = mix6.medianHeight) {
      return Math.max(LIVE_CUTOFF_MIN, canopyCutoffMeters(height, heightPx, vfov, px))
    },
    update(sun) {
      u.sun.value.copy(sun)
    },
    showFinal(stand, eye, inner, groundAt) {
      const key = finalStandKey(stand, eye, inner)
      if (!finalKept || finalKept.key !== key || finalKept.groundAt !== groundAt) {
        const fieldsOf = [...stand.fields]
        const sStride = fieldsOf.length
        const [sx, sy, ss, sh, sw, st] = ['x', 'y', 'species', 'height', 'crownWidth', 'tone'].map((f) => fieldsOf.indexOf(f))
        const slots = stand.species.map((name) => speciesSlot(name))
        const list = new Float32Array(stand.count * 7)
        const heights: number[] = []
        let count = 0
        let dropped = 0
        // A stem within a hair of `inner` is measured as before (Math.hypot); the rest are told by their squares.
        const nearInner = inner * inner * (1 + 1e-9)
        for (let i = 0; i < stand.count; i++) {
          const o = i * sStride
          const x = stand.data[o + sx]
          const y = stand.data[o + sy]
          const dx = x - eye[0]
          const dy = y - eye[1]
          if (!(dx * dx + dy * dy > nearInner) && Math.hypot(dx, dy) < inner) continue
          if (ground.inWater(x, y)) continue
          let h = stand.data[o + sh]
          if (!(h > 0)) continue
          let crown = sw >= 0 ? stand.data[o + sw] : 0
          let slot = slots[stand.data[o + ss]] ?? 1
          const age = burntAge(x, y)
          if (Number.isFinite(age)) {
            // Burnt: the stand's stems stand as snags, fewer as they fall.
            const snag = asSnag(x, y, h, crown, age)
            if (!snag) {
              dropped++
              continue
            }
            h = snag.height
            crown = snag.crownWidth
            slot = SNAG_SLOT
          } else if (!keepStem(x, y, dryStemShare(x, y))) {
            // None on rock or water posts, fewer on sparse, shrub and herb ground (the cover code's cap).
            dropped++
            continue
          }
          const k = count * 7
          list[k] = x
          list[k + 1] = y
          list[k + 2] = groundAt(x, y)
          list[k + 3] = h
          list[k + 4] = crown
          list[k + 5] = slot
          list[k + 6] = st >= 0 ? stand.data[o + st] : 0.5
          if (count % 5 === 0) heights.push(h)
          count++
        }
        const middle = median(heights)
        this.clearFinal()
        forgetFinal()
        finalKept = {
          key,
          groundAt,
          mesh: count ? cardMesh(list, count, finalMaterial, 'far forest: final stand') : null,
          stems: count,
          medianHeight: Number.isFinite(middle) ? middle : mix6.medianHeight,
          dropped
        }
      }
      const { mesh, stems: count, medianHeight, dropped } = finalKept
      this.clearFinal()
      if (mesh) {
        finalMesh = mesh
        object.add(mesh)
      }
      u.finalOn.value = 1
      u.finalEye.value.set(eye[0], -eye[1])
      const from = (stand.bearings[0] * Math.PI) / 180
      const width = Math.min(Math.PI * 2, ((stand.bearings[1] - stand.bearings[0]) * Math.PI) / 180)
      u.finalFrom.value = Math.atan2(Math.sin(from), Math.cos(from))
      u.finalWidth.value = width < 0 ? width + Math.PI * 2 : width
      u.finalInner.value = inner
      u.finalRadius.value = stand.radiusMeters
      stats.finalThinned = dropped
      return { stems: count, medianHeight }
    },
    clearFinal() {
      // Taken out of the scene, not freed: the same stand may be asked for again (`finalKept`).
      if (finalMesh) {
        finalMesh.removeFromParent()
        finalMesh = null
      }
      u.finalOn.value = 0
    },
    dispose() {
      this.clearFinal()
      forgetFinal()
      disposeTiled(finalPrimer)
      object.removeFromParent()
      disposeTiled(slabs)
      disposeTiled(singleCards)
      canopy?.geometry.dispose()
      ;(canopy?.material as THREE.Material | undefined)?.dispose()
    }
  }
}

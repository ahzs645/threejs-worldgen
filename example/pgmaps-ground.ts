// A PGMaps package's ground: the terrain with forest floor, harvested ground (soil, duff, dry grass) inside openings
// and exposed mineral soil on road scars, weathered (wet, snow) through threejs-weathering; stumps and slash on the cut
// ground; and the package's water bodies as threejs-water lakes, with their beds carved into the terrain.
//
// Package coordinates are local metres, x east, y north, z up; three.js is y up with north at −z, so a package point
// (x, y, z) is drawn at (x, z, −y).
import * as THREE from 'three/webgpu'
import {
  abs,
  atan,
  attribute,
  cameraPosition,
  distance,
  exp2,
  float,
  floor,
  fwidth,
  hash as nodeHash,
  length,
  log2,
  max,
  mix,
  normalWorld,
  mx_noise_float,
  positionWorld,
  reflector,
  smoothstep,
  uniform,
  vec3
} from 'three/tsl'
import { hash, mulberry32 } from 'threejs-scatter'
import { Lakes, WaterPrepass, type HydrologyResult } from 'threejs-water'
import { applyMaterialStatesTo, type MaterialStateUniforms } from 'threejs-weathering'
import { openingSeed, pileSeed } from './pgmaps-tree-bins'
import {
  coverSampler,
  cutAge,
  packageCanopySampler,
  retainedShare,
  VIA_CUT_LINEAR,
  VIA_FRESH_CUT_YEARS,
  type CoverSource,
  type CoverStems
} from './pgmaps-cover'
import {
  burntShare,
  coverClassSampler,
  coverCounts,
  droppedSeaLevel,
  natureWeights,
  onBareGround,
  onBurntGround,
  rockStands,
  SCENE_COVER,
  skylineWaterHeight
} from './pgmaps-nature'
import { burnAge, burnAgeSampler, type FireInput, type SceneFire } from './pgmaps-burn'
import { coxMunk, reflectionBlurPx } from './pgmaps-far-forest-math'

type Ring = Array<[number, number]>

/** A water body: polygon rings in package metres (outer first, then holes) and the surface height. */
export interface WaterBody {
  rings: Ring[]
  /** Water surface, metres; `null` = the lowest ground along the outer ring. */
  elevationMeters: number | null
  /**
   * The sea (PGMaps: level 0, islands as holes). Its surface claims the lake grid's cells first, so a basemap water
   * polygon lying over it is drawn once.
   */
  sea?: boolean
}

/** A package terrain grid: row-major from the south row, west to east; null where the source has no height. */
export interface TerrainGrid {
  west: number
  south: number
  spacing: number
  columns: number
  rows: number
  heights: Array<number | null>
  /**
   * Crown closure per post, 0–1, row-major like `heights`; 0 on cut, non-treed and water ground, null unknown. Shades
   * the ground under the forest (colour only). Absent from older packages, which derive it from `trees`.
   */
  canopy?: Array<number | null> | null
  /**
   * Natural ground cover per post (PGMaps `SCENE_COVER`: 0 forest, 1 sparse, 2 shrub, 3 herb, 4 rock, 5 water, 6 cut),
   * row-major like `heights`; null unknown. Gives open ground its material (pgmaps-nature.ts). Absent from older packages.
   * 7 burnt: a recent wildfire (pgmaps-burn.ts).
   */
  cover?: Array<number | null> | null
  /** The fire season per burnt post (PGMaps `burntYear`), null elsewhere; aged by the package's `fires`. */
  burntYear?: Array<number | null> | null
}

export interface GroundInput {
  terrain: TerrainGrid
  /**
   * Coarser terrain past `terrain`, out to the skyline (a capture's package), the same layout; null posts unknown. Drawn
   * wherever `terrain` has no posts, in place of the invented apron, which then only starts past it.
   */
  farTerrain?: TerrainGrid | null
  /**
   * Coarsest terrain past `farTerrain`, out to the real skyline (a capture's package), the same layout. Its heights
   * already have the earth's curve taken off (`earthCurvature`): drawn as given, only where neither `terrain` nor
   * `farTerrain` has posts, and the invented apron only starts past it.
   */
  skylineTerrain?: (TerrainGrid & { earthCurvature?: { earthRadiusMeters: number; refraction: number } }) | null
  /**
   * Cut ground. `ageYears` (years since harvest; absent or null: `DEFAULT_CUT_AGE_YEARS`) greens it up;
   * `retainedShare` (dispersed retention, 0–1) shades it and thins its stumps. Its retained stems are in `trees`.
   */
  openings: Array<{ rings: Ring[]; ageYears?: number | null; retainedShare?: number | null }>
  /** The package's stems, for crown closure where the package gives no `canopy`. */
  trees?: CoverStems | null
  treeRadiusMeters?: number
  debris: { count: number; data: number[] }
  roadScars: Array<{ kind: 'cut' | 'fill'; ring: Ring }>
  water?: WaterBody[] | null
  /** The wildfires behind cover 7 (PGMaps `fires`): each season's years since the fire (pgmaps-burn.ts). */
  fires?: SceneFire[] | null
  createdAt?: string | null
  environment?: FireInput['environment']
}

export function inRing(ring: Ring, x: number, y: number) {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i]
    const [xj, yj] = ring[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/** Distance from a point to a ring's edges. */
function edgeDistance(ring: Ring, x: number, y: number) {
  let best = Infinity
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [ax, ay] = ring[j]
    const [bx, by] = ring[i]
    const dx = bx - ax
    const dy = by - ay
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1)))
    best = Math.min(best, Math.hypot(ax + dx * t - x, ay + dy * t - y))
  }
  return best
}

/** A polygon with holes and its bounding box, for quick point tests. */
interface Area {
  rings: Ring[]
  box: [number, number, number, number]
  /** From the polygon's own outer ring (`openingSeed`), never its place in the package's list. */
  seed: number
  /** An opening's drawn age, years, and dispersed retention, 0–1 (`cutAge`, `retainedShare`); 0 for other areas. */
  age: number
  retained: number
}
const areaOf = (rings: Ring[], age = 0, retained = 0): Area => {
  const box: Area['box'] = [Infinity, Infinity, -Infinity, -Infinity]
  for (const [x, y] of rings[0] ?? []) {
    box[0] = Math.min(box[0], x)
    box[1] = Math.min(box[1], y)
    box[2] = Math.max(box[2], x)
    box[3] = Math.max(box[3], y)
  }
  return { rings, box, seed: openingSeed(rings), age, retained }
}
const inArea = (a: Area, x: number, y: number) =>
  x >= a.box[0] &&
  x <= a.box[2] &&
  y >= a.box[1] &&
  y <= a.box[3] &&
  inRing(a.rings[0], x, y) &&
  !a.rings.slice(1).some((r) => inRing(r, x, y))

/** An axis-aligned rectangle, package metres. */
export interface Extent {
  west: number
  south: number
  east: number
  north: number
}

/** Fills a grid's nulls along its rows, then its columns, from the nearest known post; returns how many it filled. */
function fillGaps(heights: Array<number | null>, columns: number, rows: number): number {
  let gaps = 0
  for (const h of heights) if (h === null) gaps++
  if (!gaps) return 0
  const fill = (index: (i: number, j: number) => number, outer: number, inner: number) => {
    for (let i = 0; i < outer; i++) {
      let last: number | null = null
      for (let j = 0; j < inner; j++) last = heights[index(i, j)] ?? last ?? null
      for (let j = inner - 1; j >= 0; j--) {
        const k = index(i, j)
        if (heights[k] === null) heights[k] = last
        else last = heights[k]
      }
    }
  }
  fill((r, c) => r * columns + c, rows, columns)
  fill((c, r) => r * columns + c, columns, rows)
  return gaps
}

/** Bilinear height of a grid at a package point, clamped to the grid; `unknown` posts read as NaN. */
function gridSampler(grid: TerrainGrid, heights: ArrayLike<number | null>, unknown: number) {
  const { west, south, spacing, columns, rows } = grid
  const at = (column: number, row: number) => heights[row * columns + column] ?? unknown
  return (x: number, y: number): number => {
    const gx = Math.min(columns - 1.0001, Math.max(0, (x - west) / spacing))
    const gy = Math.min(rows - 1.0001, Math.max(0, (y - south) / spacing))
    const x0 = Math.floor(gx)
    const y0 = Math.floor(gy)
    const fx = gx - x0
    const fy = gy - y0
    return (
      at(x0, y0) * (1 - fx) * (1 - fy) +
      at(x0 + 1, y0) * fx * (1 - fy) +
      at(x0, y0 + 1) * (1 - fx) * fy +
      at(x0 + 1, y0 + 1) * fx * fy
    )
  }
}

const gridExtent = (g: TerrainGrid): Extent => ({
  west: g.west,
  south: g.south,
  east: g.west + (g.columns - 1) * g.spacing,
  north: g.south + (g.rows - 1) * g.spacing
})
const inside = (e: Extent, x: number, y: number) => x >= e.west && x <= e.east && y >= e.south && y <= e.north

/**
 * Writes a ground vertex's cover attributes: `aCover` (closure, known, age, retention) into `out` and `aNature` (rock,
 * herb, shrub, sparse shares) into `nature`, both at `at` (4 a vertex), and `aBurn` (burnt share, years since the fire
 * × share, the share of rock that carries a stand) into `burn` at `at / 4 × 3`.
 */
type CoverOf = (
  x: number,
  y: number,
  cut: Area | undefined,
  out: Float32Array,
  at: number,
  nature: Float32Array,
  burn: Float32Array
) => void

/** Water bodies with their surface level resolved. */
interface Pond extends Area {
  level: number
  sea: boolean
}

/** The package's lakes: threejs-water's, with a planar reflection about the largest one's level. */
export interface Water {
  lakes: Lakes
  prepass: WaterPrepass
  box: THREE.Box3
  /** The mirrored render (three.js `reflector()`), its target at the largest lake's level; `resolutionScale` live. */
  mirror: ReturnType<typeof reflector>
  /** The level it mirrors about, metres. */
  mirrorLevel: number
  /**
   * The look: `'via'` a crisp mirror pulled toward slate blue with a thin tan shore band; `'photo'` rippled by the wind
   * (Cox–Munk roughness, vertical streaks) reflecting the sky. `windMs` 0 is a mirror in either.
   */
  setLook(look: 'via' | 'photo', windMs: number): void
  /** Once a frame: the reflection's streak length follows the frame's focal length, pixels. */
  updateView(focalPx: number): void
}

export interface Ground {
  /** Ground height at a package point (x east, y north), with lake beds carved. */
  groundAt(x: number, y: number): number
  /** The terrain meshes, stumps and slash. */
  group: THREE.Group
  /** The water, or null. Add with the group; call `update` per frame. */
  water: Water | null
  /** Whether a package point is inside a harvest opening. */
  isCut(x: number, y: number): boolean
  /**
   * Where the ground's crown closure came from (`package`: its `canopy`; `stems`: derived from its stems; `none`), and
   * the openings' drawn ages and retention (`given`: how many carried `ageYears`). Diagnostics only.
   */
  cover: {
    source: CoverSource
    openings: number
    given: number
    ages: number[]
    retained: number[]
    /** Whether the package carries `cover` codes (rock, herb, shrub, sparse ground) on any grid. */
    classes: boolean
    /** Terrain vertices on burnt ground (cover 7). */
    burntVertices: number
  }
  /**
   * The look of open ground: `'via'` the VIA renders' pale tan rock and yellow-green herb openings, `'photo'` greyer
   * rock and drier, darker grass, as a photograph shows them.
   */
  setLook(look: 'via' | 'photo'): void
  /** Whether a package point is under water (no stems, stumps or slash there). */
  inWater(x: number, y: number): boolean
  /**
   * Years since the fire where the package's cover codes put a point on burnt ground (`onBurntGround`), else NaN: its
   * listed stems there are drawn as snags (pgmaps-burn.ts).
   */
  burntAge(x: number, y: number): number
  /**
   * Whether the package's cover codes put a point on rock or water (`onBareGround`): its listed stems there are not
   * drawn, so the picture shows the bluff. False where the package has no codes.
   */
  onBare(x: number, y: number): boolean
  /** The terrain drawn from the package's grid, package metres. */
  drawnExtent: Extent
  /** The terrain drawn from `farTerrain`, package metres (it has a hole where `drawnExtent` is); null without one. */
  farExtent: Extent | null
  /** The terrain drawn from `skylineTerrain`, package metres (a hole where the grids inside it are); null without one. */
  skylineExtent: Extent | null
  /** Share of the grid's posts that had no height: taken from `farTerrain` (`fromFar`), or invented from neighbours. */
  filled: { fromFar: number; invented: number; farInvented: number; skylineInvented: number }
  /** Radius of the invented apron's inner (the real terrain's edge) and outer edge from the origin, metres. */
  apronInner: number
  apronRadius: number
  /**
   * The terrain meshes from the package's grid (`terrain`), from `farTerrain` (`far`), from `skylineTerrain`
   * (`skyline`), the invented apron, and stumps and slash (`debris`), for the capture's class pass.
   */
  parts: {
    terrain: THREE.Mesh[]
    far: THREE.Mesh | null
    /** The mesh from `skylineTerrain`, for the class pass's skyline drape; null without one. */
    skyline: THREE.Mesh | null
    apron: THREE.Mesh
    debris: THREE.Object3D[]
  }
  dispose(): void
}

/** Builds the ground of a package. `states` are the atmosphere's wetness and snow (shared, live). */
export async function buildGround(
  pkg: GroundInput,
  states: MaterialStateUniforms,
  options: {
    renderer: THREE.WebGPURenderer
    camera: THREE.Camera
    sun: THREE.DirectionalLight
    treeRadius: number
    /** The sky as drawn behind the scene, for the water to reflect (`Atmosphere.skyTexture`); null: a sky gradient. */
    sky?: THREE.Texture | null
  }
): Promise<Ground> {
  const { west, south, spacing, columns, rows } = pkg.terrain
  const nearGrid = gridExtent(pkg.terrain)
  // The skyline's coarse grid, when the package has one that reaches past this grid. Its gaps take their neighbours'.
  const given = pkg.farTerrain
  const farGrid = given && given.heights?.length === given.columns * given.rows ? given : null
  let farHeights: Array<number | null> | null = null
  let farInvented = 0
  if (farGrid) {
    const e = gridExtent(farGrid)
    if (e.west < nearGrid.west && e.east > nearGrid.east && e.south < nearGrid.south && e.north > nearGrid.north) {
      farHeights = farGrid.heights.slice()
      farInvented = fillGaps(farHeights, farGrid.columns, farGrid.rows) / farHeights.length
    }
  }
  // The real skyline's coarsest grid, when it reaches past the grids inside it. Its heights already carry the earth's
  // curve; its gaps take their neighbours'.
  const inner = farGrid && farHeights ? gridExtent(farGrid) : nearGrid
  const sky = pkg.skylineTerrain
  let skylineGrid: TerrainGrid | null = null
  let skylineHeights: Array<number | null> | null = null
  let skylineInvented = 0
  if (sky && sky.heights?.length === sky.columns * sky.rows) {
    const e = gridExtent(sky)
    if (e.west < inner.west && e.east > inner.east && e.south < inner.south && e.north > inner.north) {
      skylineGrid = sky
      skylineHeights = sky.heights.slice()
      // PGMaps sends no sea polygons on this grid (a flat sea cannot follow its curvature drop): its water posts near
      // sea level are laid at the dropped sea level and drawn as water (`aWater`, `ringMesh`).
      const curvature = sky.earthCurvature
      if (curvature && sky.cover?.length === skylineHeights.length)
        for (let j = 0; j < sky.rows; j++)
          for (let i = 0; i < sky.columns; i++) {
            const k = j * sky.columns + i
            if (sky.cover[k] !== SCENE_COVER.water) continue
            const level = droppedSeaLevel(Math.hypot(sky.west + i * sky.spacing, sky.south + j * sky.spacing), curvature)
            skylineHeights[k] = skylineWaterHeight(skylineHeights[k], level)
          }
      skylineInvented = fillGaps(skylineHeights, sky.columns, sky.rows) / skylineHeights.length
    }
  }
  // Gaps (no source DEM there) take the coarse grid's height where it has one (`farTerrain`, read before its own gaps
  // were filled), then the nearest known height along their row, then their column, so the mesh stays whole.
  const heights = pkg.terrain.heights.slice()
  let fromFar = 0
  if (farGrid && farHeights) {
    const raw = gridSampler(farGrid, farGrid.heights, NaN)
    for (let row = 0; row < rows; row++)
      for (let column = 0; column < columns; column++) {
        const k = row * columns + column
        if (heights[k] !== null) continue
        const h = raw(west + column * spacing, south + row * spacing)
        if (Number.isFinite(h)) {
          heights[k] = h
          fromFar++
        }
      }
  }
  const invented = fillGaps(heights, columns, rows)
  // A loop, not Math.min(...): spreading a 160,000-cell grid overflows the call stack.
  let base = Infinity
  for (const h of heights) if (h !== null && h < base) base = h
  const nearAt = gridSampler(pkg.terrain, heights, base)
  const farAt = farGrid && farHeights ? gridSampler(farGrid, farHeights, base) : null
  const farExtent = farGrid && farHeights ? gridExtent(farGrid) : null
  const skylineAt = skylineGrid && skylineHeights ? gridSampler(skylineGrid, skylineHeights, base) : null
  const skylineExtent = skylineGrid && skylineHeights ? gridExtent(skylineGrid) : null
  /** The source terrain at a package point, bilinear: this grid, then the coarse one past it, then the skyline's. */
  const terrainAt = (x: number, y: number): number => {
    if (inside(nearGrid, x, y)) return nearAt(x, y)
    if (farAt && farExtent && (!skylineAt || inside(farExtent, x, y))) return farAt(x, y)
    return skylineAt ? skylineAt(x, y) : nearAt(x, y)
  }

  // Water: each body's level, and a bed carved under it that deepens away from the shore (so the surface never
  // z-fights a DEM that was flattened to the lake, and the water has depth to absorb light through).
  const ponds: Pond[] = []
  for (const body of pkg.water ?? []) {
    if (!body?.rings?.[0] || body.rings[0].length < 3) continue
    const area = areaOf(body.rings)
    let level = body.elevationMeters ?? Infinity
    if (!Number.isFinite(level)) {
      const outer = body.rings[0]
      for (let i = 0; i < outer.length; i++) {
        const [ax, ay] = outer[i]
        const [bx, by] = outer[(i + 1) % outer.length]
        const n = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 2))
        for (let k = 0; k < n; k++)
          level = Math.min(level, terrainAt(ax + ((bx - ax) * k) / n, ay + ((by - ay) * k) / n))
      }
    }
    if (Number.isFinite(level)) ponds.push({ ...area, level, sea: body.sea === true })
  }
  const carve = (x: number, y: number, h: number) => {
    for (const p of ponds) {
      if (!inArea(p, x, y)) continue
      const shore = Math.min(...p.rings.map((r) => edgeDistance(r, x, y)))
      h = Math.min(h, p.level - Math.min(4.5, 0.35 + shore * 0.09))
    }
    return h
  }
  const groundAt = ponds.length ? (x: number, y: number) => carve(x, y, terrainAt(x, y)) : terrainAt

  const listed = pkg.openings.filter((o) => o.rings?.[0]?.length >= 3)
  const openings = listed.map((o) => areaOf(o.rings, cutAge(o.ageYears), retainedShare(o.retainedShare)))
  const openingAt = (x: number, y: number) => openings.find((a) => inArea(a, x, y))
  const cover = coverSampler(pkg)
  /**
   * The ground's cover attribute at a point: x crown closure × known, y known (0/1), z age × cut, w retention × cut;
   * the shader divides x by y and z, w by the cut flag, so a value stays exact along an edge it is interpolated across.
   */
  const classAt = coverClassSampler([pkg.terrain, farGrid, skylineGrid])
  // Rock carrying a stand (PGMaps: a treed VRI polygon with bedrock or talus components keeps its closure there): its
  // stems stand, and the rock shows between their crowns.
  const packageCanopy = packageCanopySampler(pkg)
  const standingAt = (x: number, y: number) => rockStands(packageCanopy(x, y))
  const fireAge = burnAgeSampler(pkg)
  /** Years since the fire at a point (the package's, else `DEFAULT_BURN_AGE_YEARS`). */
  const ageOfBurn = (x: number, y: number) => burnAge(fireAge?.(x, y))
  let burntVertices = 0
  const coverOf: CoverOf = (x, y, cut, out, at, nature, burn) => {
    const closure = cut ? 0 : cover.at(x, y)
    const known = Number.isFinite(closure) ? 1 : 0
    out[at] = known ? closure : 0
    out[at + 1] = known
    out[at + 2] = cut ? cut.age : 0
    out[at + 3] = cut ? cut.retained : 0
    // Rock, herb, shrub and sparse shares, and the burnt share: cut ground keeps its own look by age, whatever its code.
    const weights = cut || !classAt ? null : classAt(x, y)
    natureWeights(weights, nature, at)
    const burnt = burntShare(weights)
    const b = (at / 4) * 3
    burn[b] = burnt
    burn[b + 1] = burnt > 0 ? burnt * ageOfBurn(x, y) : 0
    burn[b + 2] = weights && weights[SCENE_COVER.rock] > 0 && standingAt(x, y) ? weights[SCENE_COVER.rock] : 0
    if (burnt > 0) burntVertices++
  }
  const scars = pkg.roadScars.filter((s) => s.ring?.length >= 3).map((s) => ({ ...areaOf([s.ring]), kind: s.kind }))
  const isCut = (x: number, y: number) => !!openingAt(x, y)
  const inWater = (x: number, y: number) => ponds.some((p) => inArea(p, x, y))

  const look = uniform(1)
  const material = groundMaterial(states, options.treeRadius, look)
  const group = new THREE.Group()
  /** A terrain mesh `size` metres across centred on the package origin, `step` metres between vertices. */
  const mesh = (size: number, step: number) => {
    const n = Math.round(size / step)
    const geometry = new THREE.PlaneGeometry(size, size, n, n).rotateX(-Math.PI / 2)
    const position = geometry.attributes.position
    const kind = new Float32Array(position.count * 4)
    const coverage = new Float32Array(position.count * 4)
    const nature = new Float32Array(position.count * 4)
    const burn = new Float32Array(position.count * 3)
    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i)
      const y = -position.getZ(i)
      position.setY(i, groundAt(x, y))
      const opening = openingAt(x, y)
      coverOf(x, y, opening, coverage, i * 4, nature, burn)
      const cut = opening ? 1 : 0
      const scar = scars.find((s) => inArea(s, x, y))
      // x: harvested, y: road cut bank, z: road fill, w: open to the sky (snow lies deeper there than under canopy)
      kind[i * 4] = cut
      kind[i * 4 + 1] = scar?.kind === 'cut' ? 1 : 0
      kind[i * 4 + 2] = scar?.kind === 'fill' ? 1 : 0
      kind[i * 4 + 3] = cut || scar ? 1 : 0.55
    }
    geometry.setAttribute('aGround', new THREE.BufferAttribute(kind, 4))
    geometry.setAttribute('aCover', new THREE.BufferAttribute(coverage, 4))
    geometry.setAttribute('aNature', new THREE.BufferAttribute(nature, 4))
    geometry.setAttribute('aBurn', new THREE.BufferAttribute(burn, 3))
    geometry.setAttribute('aWater', new THREE.BufferAttribute(new Float32Array(position.count), 1))
    geometry.computeVertexNormals()
    const m = new THREE.Mesh(geometry, material)
    m.receiveShadow = true
    return m
  }
  const near = mesh(1400, 5)
  const size = Math.min(columns, rows) * spacing - spacing
  // A capture's package (one with skyline terrain) draws its grid at the grid's own spacing, so stems far off stand on
  // the very ground they were seated on; the road view's package keeps the coarser mesh it always had.
  const midStep = farExtent || skylineExtent ? spacing : spacing * 2
  const far = mesh(size, midStep)
  far.position.y = -0.3 // under the near mesh where they overlap
  const half = size / 2
  const drawnExtent = {
    west: Math.max(-half, west),
    south: Math.max(-half, south),
    east: Math.min(half, west + (columns - 1) * spacing),
    north: Math.min(half, south + (rows - 1) * spacing)
  }
  const midHole = { west: -half, south: -half, east: half, north: half }
  const midEdge = meshEdge(far, Math.round(size / midStep))
  // Each coarser grid is a ring round the one inside it, stitched to that one's very edge.
  const ring = (grid: TerrainGrid | null, hole: Extent, edge: EdgePoint[] | null, water = false) =>
    grid && edge ? ringMesh(grid, edge, hole, groundAt, openingAt, coverOf, material, water) : null
  const skyline = ring(farGrid && farExtent ? farGrid : null, midHole, midEdge)
  const outermost = ring(
    skylineGrid,
    skyline && farExtent ? farExtent : midHole,
    skyline && farGrid ? gridEdge(skyline, farGrid) : midEdge,
    true
  )
  const reached = outermost && skylineExtent ? skylineExtent : skyline && farExtent ? farExtent : null
  const skylineHalf = reached ? Math.min(-reached.west, reached.east, -reached.south, reached.north) : half
  const apronInner = skylineHalf
  // Past the real skyline's grid the apron only fills the corners and beyond; past `farTerrain` it reaches farther.
  const apronRadius = outermost
    ? Math.max(APRON_RADIUS, skylineHalf * Math.SQRT2 + 2000)
    : skyline
      ? Math.max(APRON_RADIUS, skylineHalf * 2)
      : APRON_RADIUS
  const apron = horizonApron(skylineHalf * 2, groundAt, material, apronRadius)
  near.name = 'terrain near'
  far.name = 'terrain far'
  apron.name = 'terrain apron'
  group.add(near, far, apron)
  if (skyline) {
    skyline.name = 'terrain skyline'
    skyline.position.y = -0.3
    group.add(skyline)
  }
  if (outermost) {
    outermost.name = 'terrain real skyline'
    outermost.position.y = -0.3
    group.add(outermost)
  }

  // Stumps and slash only near the eye, where they are more than a pixel; past it the cut ground's colour carries them.
  const debrisReach = options.treeRadius + DEBRIS_MARGIN
  const stumps = stumpsOf(openings, groundAt, ponds, debrisReach)
  const slash = slashOf(pkg.debris, openings, groundAt, inWater, openingAt, debrisReach)
  const coverInfo: Ground['cover'] = {
    source: cover.source,
    openings: openings.length,
    given: listed.filter((o) => typeof o.ageYears === 'number').length,
    ages: openings.map((a) => a.age),
    retained: openings.map((a) => a.retained),
    classes: !!classAt,
    burntVertices
  }
  console.log('[pgmaps-ground] cover', JSON.stringify(coverInfo))
  {
    // Diagnostics: the package's crown closure by grid (share known, mean of the known posts, share ≥ 0.4).
    const stats = (c: Array<number | null> | null | undefined) => {
      if (!c) return null
      let known = 0
      let sum = 0
      let closed = 0
      for (const v of c)
        if (typeof v === 'number') {
          known++
          sum += v
          if (v >= 0.4) closed++
        }
      return {
        known: +(known / c.length).toFixed(3),
        mean: +(sum / (known || 1)).toFixed(3),
        closed: +(closed / (known || 1)).toFixed(3)
      }
    }
    console.log(
      '[pgmaps-ground] canopy',
      JSON.stringify({
        near: stats(pkg.terrain.canopy),
        far: stats(pkg.farTerrain?.canopy),
        skyline: stats(pkg.skylineTerrain?.canopy)
      })
    )
    // The package's cover codes by grid: posts per code (forest, sparse, shrub, herb, rock, water, cut) and unknown.
    console.log(
      '[pgmaps-ground] cover classes',
      JSON.stringify({
        near: coverCounts(pkg.terrain),
        far: coverCounts(pkg.farTerrain),
        skyline: coverCounts(pkg.skylineTerrain)
      })
    )
  }
  console.log(
    '[pgmaps-ground] terrain',
    JSON.stringify({
      far: skyline ? farExtent : null,
      skyline: outermost ? skylineExtent : null,
      apronInner,
      apronRadius
    })
  )
  // Wood takes the same weather as the ground: dark and glossy when wet, snow on whatever faces up.
  for (const m of [stumps, ...slash])
    applyMaterialStatesTo(
      m.material as THREE.MeshStandardNodeMaterial,
      { wetness: states.wetness, snow: states.snow.mul(0.6) },
      { streaks: false, puddles: false }
    )
  group.add(stumps, ...slash)

  let water: Ground['water'] = null
  if (ponds.length) {
    water = await lakesOf(ponds, options)
    group.add(water.lakes.object)
  }

  return {
    groundAt,
    group,
    water,
    isCut,
    inWater,
    onBare: (x, y) => !!classAt && onBareGround(classAt(x, y), standingAt(x, y)),
    burntAge: (x, y) => (classAt && !openingAt(x, y) && onBurntGround(classAt(x, y)) ? ageOfBurn(x, y) : NaN),
    cover: coverInfo,
    setLook(next) {
      look.value = next === 'via' ? 1 : 0
    },
    drawnExtent,
    farExtent: skyline ? farExtent : null,
    skylineExtent: outermost ? skylineExtent : null,
    filled: {
      fromFar: heights.length ? fromFar / heights.length : 0,
      invented: heights.length ? invented / heights.length : 0,
      farInvented: skyline ? farInvented : 0,
      skylineInvented: outermost ? skylineInvented : 0
    },
    apronInner,
    apronRadius,
    parts: { terrain: [near, far], far: skyline, skyline: outermost, apron, debris: [stumps, ...slash] },
    dispose() {
      group.removeFromParent()
      group.traverse((o) => {
        const m = o as THREE.Mesh
        if (m.isMesh) m.geometry.dispose()
      })
      material.dispose()
      for (const m of [stumps, ...slash]) (m.material as THREE.Material).dispose()
      if (water) {
        water.lakes.dispose()
        water.prepass.dispose()
      }
    }
  }
}

/**
 * The ground's look. Under the stand, forest floor (needle litter and moss); farther off, where the drawn stems thin to
 * cards, the ground stands for the forest itself: a closed canopy, dark and mottled, as thick as the crown closure says
 * (`aCover`), and open ground stays open (dry grass and rock). Without a known closure the canopy takes over with distance
 * as it always has. Cut ground greens up with its age: grey-brown soil, duff and grey slash with a little fireweed in
 * the first years, green shrubs over brown by five, green by ten, a young stand's darker green past fifteen. Road cut
 * banks are pale mineral soil, fill darker. Fine noise fades with distance so it never shimmers. Wetness (puddles) and
 * snow (deeper in the open) come from the shared state uniforms. Colour only: nothing here moves the ground.
 */
// biome-ignore lint/suspicious/noExplicitAny: TSL nodes
type Node = any

function groundMaterial(states: MaterialStateUniforms, treeRadius: number, via: Node) {
  const kind = attribute('aGround', 'vec4')
  const cover = attribute('aCover', 'vec4')
  const p = positionWorld.xz
  const d = distance(positionWorld, cameraPosition)
  const fine = float(1).sub(smoothstep(60, 220, d))
  // Crown-scale detail (about 9 m) holds to 4 km, where a capture's pixel is still a few metres, and fades by 7 km.
  const mid = float(1).sub(smoothstep(4000, 7000, d))
  const n1 = mx_noise_float(p.mul(0.045))
  const n2 = mx_noise_float(p.mul(0.33)).mul(float(1).sub(smoothstep(150, 600, d)))
  const n3 = mx_noise_float(p.mul(2.3)).mul(fine)
  const n4 = mx_noise_float(p.mul(0.012))
  const n5 = mx_noise_float(p.mul(0.11)).mul(mid)
  const n6 = mx_noise_float(p.mul(0.0045))
  const n7 = mx_noise_float(p.mul(0.028)).mul(float(1).sub(smoothstep(9000, 16000, d)))

  // Under the stand: brown needle litter with mossy patches.
  const litter = vec3(0.095, 0.07, 0.044)
  const moss = vec3(0.06, 0.085, 0.032)
  const floor = mix(litter, moss, smoothstep(-0.25, 0.45, n1.add(n2.mul(0.4)))).mul(n3.mul(0.18).add(1))
  // The forest seen from afar: dark crowns and their shadows, mottled by crown clumps (fading out with distance) and by
  // the stand mosaic, never a lit lawn.
  const crowns = mix(vec3(0.008, 0.016, 0.011), vec3(0.017, 0.031, 0.019), smoothstep(-0.45, 0.55, n4.add(n6.mul(0.6))))
  // Lit crowns and the shadowed gaps between them, in clumps: the texture of a closed stand seen across a valley.
  const clumps = smoothstep(-0.55, 0.55, n5.mul(0.6).add(n7.mul(0.5)).add(n1.mul(0.3)))
  // The far forest's trees (pgmaps-far-forest.ts) stand on this ground: what shows between them is the stand's own
  // shade, so it is kept darker than the crowns lit above it.
  const canopy = crowns.mul(clumps.mul(0.9).add(0.55)).mul(0.3)
  // Open ground far off (closure 0, not cut): grass and low shrubs, grey rock where it is steep.
  const herbs = mix(vec3(0.045, 0.05, 0.028), vec3(0.06, 0.058, 0.036), smoothstep(-0.2, 0.5, n1.add(n6.mul(0.5))))
  const rock = vec3(0.11, 0.105, 0.095)
  const steep = smoothstep(0.82, 0.62, normalWorld.y).mul(
    smoothstep(-0.3, 0.3, n5.add(n1.mul(0.5)))
      .mul(0.5)
      .add(0.5)
  )
  const open = mix(herbs, rock, steep).mul(n5.mul(0.2).add(1))
  // Crown closure, where the package (or its stems) says: shown on the ground past the stems drawn as meshes.
  const known = cover.y
  const closure = cover.x.div(known.max(1e-3)).clamp(0, 1)
  const away = smoothstep(150, 420, d)
  const closed = mix(
    smoothstep(treeRadius * 0.55, treeRadius * 0.95, d),
    smoothstep(0.02, 0.22, closure).mul(away),
    known
  )
  // Open ground by its cover code (`aNature`: rock, herb, shrub and sparse-forest shares, pgmaps-nature.ts).
  const nature = attribute('aNature', 'vec4')
  const burnAttr = attribute('aBurn', 'vec3')
  // The share of the rock that carries a stand (`aBurn.z`): the canopy stands over it, the rock shows between crowns.
  const rockStand = burnAttr.z.div(nature.x.max(1e-3)).clamp(0, 1)
  const natural = naturalGround(nature, via, d, { n1, n2, n3, n4, n5, n6 }, canopy, clumps, rockStand)
  // Burnt ground (`aBurn`: share, years since the fire × share, pgmaps-burn.ts): regrowth, char and the snags' streaks.
  const burnt = burntGround(burnAttr, via, d, { n1, n2, n3, n5, n6 }, treeRadius)
  // Rock, herb, shrub and burnt ground take the canopy off entirely; under a sparse stand the ground shows through half.
  const shaded = closed.mul(float(1).sub(natural.opening)).mul(float(1).sub(burnt.share))
  const forest = mix(
    mix(mix(mix(floor, mix(floor, open, away), known), natural.color, natural.share), canopy, shaded),
    burnt.color,
    burnt.share
  )

  // Harvested, by age (years; the cut flag divides it back out along an opening's edge).
  const age = cover.z.div(kind.x.max(1e-3))
  const retained = cover.w.div(kind.x.max(1e-3)).clamp(0, 1)
  const fresh = float(1).sub(smoothstep(0.5, 2.5, age))
  // Fresh: grey-brown churned soil and duff, grey weathering slash, rust-red needles on the newest slash.
  const soil = vec3(0.08, 0.056, 0.042)
  const duff = vec3(0.05, 0.038, 0.03)
  const slashGrey = vec3(0.1, 0.084, 0.072)
  const rust = vec3(0.075, 0.05, 0.036)
  let bare = mix(soil, duff, smoothstep(-0.3, 0.45, n5.add(n2.mul(0.6))))
  bare = mix(bare, slashGrey, smoothstep(-0.05, 0.55, n1.mul(0.7).add(n5.mul(0.5)).add(n3.mul(0.3))).mul(0.6))
  bare = mix(
    bare,
    rust,
    smoothstep(0.1, 0.6, n5.mul(-1).add(n2.mul(0.4)))
      .mul(fresh)
      .mul(0.35)
  )
  // Green-up: fireweed, grass and shrubs over the brown, then a closed young stand (`greenShare` in pgmaps-cover.ts).
  const share = smoothstep(1, 5, age)
    .mul(0.6)
    .add(smoothstep(5, 15, age).mul(0.28))
    .add(0.12)
    .min(1)
  const patch = n1.mul(0.55).add(n5.mul(0.35)).add(n2.mul(0.25)).add(n3.mul(0.15)).mul(0.5).add(0.5)
  const greened = smoothstep(share.oneMinus().sub(0.12), share.oneMinus().add(0.12), patch)
  const shrubs = mix(vec3(0.035, 0.045, 0.021), vec3(0.04, 0.06, 0.02), smoothstep(4, 10, age))
  // A young stand: the forest's own crowns and shadows, a little lighter.
  const young = canopy.mul(4.5)
  const green = mix(shrubs, young, smoothstep(15, 30, age)).mul(n5.mul(0.25).add(n3.mul(0.15)).add(1))
  // Dispersed retention shades the ground under its standing trees (the trees themselves are in the stand).
  const shadeRetained = retained.mul(-0.35).add(1)
  const photoCut = mix(bare, green, greened).mul(n3.mul(0.2).add(1)).mul(shadeRetained)
  // The VIA look paints fresh cut ground (the first year: a proposed block after harvest) as the renders do: pinkish
  // tan with a fine slash speckle (pgmaps-cover.ts `VIA_CUT_SRGB`); older openings green up as above.
  const viaFresh = float(1).sub(smoothstep(VIA_FRESH_CUT_YEARS, VIA_FRESH_CUT_YEARS + 1, age)).mul(via)
  const cut = mix(photoCut, viaCutPaint(n5).mul(shadeRetained), viaFresh)

  // Road scars: pale gravelly cut banks, darker loose fill.
  const bank = vec3(0.3, 0.255, 0.19).mul(n3.mul(0.25).add(1))
  const fillSoil = vec3(0.19, 0.15, 0.1).mul(n3.mul(0.2).add(1))

  let color = mix(forest, cut, kind.x)
  color = mix(color, bank, kind.y)
  color = mix(color, fillSoil, kind.z)
  // Water on the skyline grid (`aWater`: its sea and lakes, which have no polygons there): dark and smooth, so it takes
  // the sky's sheen as the lakes do; the VIA look slate blue, a photograph's a deeper blue.
  const wet = attribute('aWater', 'float')
  color = mix(color, mix(vec3(0.018, 0.04, 0.075), vec3(0.03, 0.045, 0.08), via), wet)
  const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.96 })
  material.roughnessNode = mix(float(0.96), float(0.3), wet)
  material.colorNode = color
  // A canopy is lit mostly from above and its crowns shade each other: the sky's image light (and the sheen it puts on
  // a slope seen edge-on across a valley) reaches only part of it. Picture only, like the colour.
  material.aoNode = float(1).sub(mix(shaded, smoothstep(12, 30, age).mul(0.8), kind.x).mul(0.6))
  applyMaterialStatesTo(
    material,
    { wetness: states.wetness, snow: states.snow.mul(kind.w) },
    { streaks: false, puddles: true }
  )
  return material
}

/**
 * Fresh cut ground in the VIA look: the renders' pinkish tan (`VIA_CUT_LINEAR`, sampled from the Gun Lake G106B block
 * and the Lakeview blocks) peppered with lighter bleached slash and darker soil and shadow, as their texture reads at any
 * distance: hashed cells about one to two pixels across (`pepper`), blended between two cell sizes so nothing pops as
 * the distance changes, over a little crown-scale mottle (`n5`) and a 4 m blotch. Terrain shading comes from the
 * lighting, as on any ground.
 */
function viaCutPaint(n5: Node): Node {
  const p = positionWorld.xz
  const blotch = mx_noise_float(vec3(p.mul(0.25), 11.5)).mul(float(1).sub(smoothstep(0.35, 0.9, fwidth(p.x.mul(0.25)))))
  const grain = pepper(p, 3)
  const [r, g, b] = VIA_CUT_LINEAR
  const tan = vec3(r, g, b).mul(n5.mul(0.08).add(blotch.mul(0.1)).add(1))
  const light = vec3(r * 1.35, g * 1.32, b * 1.3)
  const dark = vec3(r * 0.45, g * 0.4, b * 0.38)
  return mix(mix(tan, light, smoothstep(0.84, 0.92, grain).mul(0.8)), dark, smoothstep(0.68, 0.84, grain.oneMinus()).mul(0.7))
}

/**
 * A value in [0, 1) per world cell, the cells one to two pixels across at this point (`fwidth`) and never under
 * 0.25 m: a pixel-scale pepper at any distance. The two power-of-two cell sizes either side are blended, so a cell's
 * value fades rather than pops as the distance changes. Cell indices are offset positive (a float's conversion to an
 * unsigned integer clamps negatives) and the rows hashed apart, so each stays exact to 60 km.
 */
function pepper(p: Node, seed: number): Node {
  const footprint = max(max(fwidth(p.x), fwidth(p.y)).mul(1.2), 0.25)
  const level = log2(footprint)
  const below = floor(level)
  const at = (k: Node) => {
    const size = exp2(k)
    const c: Node = floor(p.div(size)).add(1048576)
    const row = floor(nodeHash(c.y.add(seed * 7919)).mul(1e6))
    return nodeHash(c.x.add(row).add(k.mul(131)))
  }
  return mix(at(below.add(1)), at(below.add(2)), level.sub(below))
}

/**
 * Burnt ground (cover 7, pgmaps-burn.ts) as the VIA renders draw a recent fire (Gun Lake CP007): yellow-green herb and
 * grey-green shrub regrowth (`regrowthShare`, `shrubShare` by the years since the fire) over black char, grey ash and
 * dark down wood; and, past the snags drawn one by one, the stand's dead stems as dark purple-grey streaks standing up
 * the view (as the canopy surface's spires), fading in from the near stand's edge, a mean grey-dark mottle once a
 * streak is under a pixel, and thinning as the snags fall (`standingSnagShare`). `a`: share, years × share.
 */
function burntGround(
  a: Node,
  via: Node,
  d: Node,
  n: { n1: Node; n2: Node; n3: Node; n5: Node; n6: Node },
  treeRadius: number
): { color: Node; share: Node } {
  const { n1, n2, n3, n5, n6 } = n
  const share = smoothstep(0.3, 0.7, a.x.add(n5.mul(0.6).add(n1.mul(0.45)).add(n2.mul(0.3)).clamp(-1, 1).mul(0.3)))
  const age = a.y.div(a.x.max(1e-3))
  const s = (lo: number, hi: number) => smoothstep(lo, hi, age)
  // regrowthShare and shrubShare (pgmaps-burn.ts).
  const regrowth = s(0, 3).mul(0.72).add(s(3, 10).mul(0.18)).add(0.1).min(1)
  const shrubby = s(2, 12).mul(0.35).add(0.3)
  const standing = float(1).sub(s(3, 15).mul(0.5)).sub(s(15, 30).mul(0.3)).max(0.2)
  // Char and ash, with dark down wood.
  const char = vec3(0.022, 0.02, 0.019)
  const ash = mix(vec3(0.05, 0.049, 0.047), vec3(0.06, 0.058, 0.056), via)
  const floorBurnt = mix(char, ash, smoothstep(0, 0.6, n2.mul(0.6).add(n3.mul(0.4)).add(n1.mul(0.3))).mul(0.4))
  // Regrowth: the renders' yellow-green herb (#b4b45a lit) and grey-green sage shrub; a photograph's drier and darker.
  const herb = mix(vec3(0.085, 0.088, 0.045), vec3(0.115, 0.12, 0.05), via)
  const shrub = mix(vec3(0.05, 0.056, 0.04), vec3(0.058, 0.068, 0.05), via)
  const shrubPatch = smoothstep(
    shrubby.oneMinus().sub(0.15),
    shrubby.oneMinus().add(0.15),
    n5.mul(0.55).add(n6.mul(0.35)).add(n2.mul(0.25)).mul(0.5).add(0.5)
  )
  const green = mix(herb, shrub, shrubPatch).mul(n3.mul(0.15).add(n5.mul(0.2)).add(n1.mul(0.12)).add(1))
  const patch = n1.mul(0.5).add(n5.mul(0.4)).add(n2.mul(0.3)).add(n3.mul(0.2)).mul(0.5).add(0.5)
  const greened = smoothstep(regrowth.oneMinus().sub(0.12), regrowth.oneMinus().add(0.12), patch)
  const ground = mix(floorBurnt, green, greened)
  // The snags past the near stand: streaks up the view, across = metres along the view's arc.
  const view = positionWorld.xz.sub(cameraPosition.xz)
  const across = atan(view.x, view.y).mul(length(view))
  const keep = float(1).sub(smoothstep(0.8, 2.5, fwidth(across.div(2.4))))
  const streak = smoothstep(0.1, 0.4, mx_noise_float(vec3(across.div(2.4), positionWorld.y.div(9), 9.5)))
  // A streak's coverage where it is drawn, its mean (about 0.45 of the ground behind them: a burnt slope across a valley
  // reads dark, #534f4f against its herb's #77764c in the Gun Lake render) where it is filtered out.
  const snagCover = mix(float(0.45), streak, keep).mul(standing)
  const away = smoothstep(treeRadius * 0.8, treeRadius * 1.8, d)
  // The renders' snags read a dark purple-grey (#2f1f3a in the Gun Lake render); a photograph's charred grey-black.
  const snag = mix(vec3(0.026, 0.025, 0.024), vec3(0.03, 0.016, 0.042), via)
  const color = mix(ground, snag, snagCover.mul(away).mul(0.9))
  return { color, share }
}

/**
 * Open ground by its cover code, as the VIA renders draw it and a photograph shows it (`via` 1 / 0):
 *
 *   rock     pale grey-tan bluffs and talus (the renders' #b0a187 lit; a photo's greyer #74746a): talus streaks down
 *            the fall line, ledges and cracks across it, darker the steeper it stands, readable at 2–10 km.
 *   herb     yellow-green and tan grass openings (a photo's drier, darker tan), tufted with shrub.
 *   shrub    dark olive low cover, mottled.
 *   sparse   light ground (grass over rock) under scattered trees; the canopy shading on it is weakened.
 *
 * Each share is pushed through a threshold the ground's own mottle moves, so a class edge is a ragged line rather than
 * a straight ramp over a grid cell; 0 and 1 stay 0 and 1. Detail fades with distance (`d`) before it can shimmer.
 * Returns the blended colour, the shares' total (`share`, 0–1) and how much canopy they take off (`opening`).
 */
function naturalGround(
  w: Node,
  via: Node,
  d: Node,
  n: { n1: Node; n2: Node; n3: Node; n4: Node; n5: Node; n6: Node },
  canopy: Node,
  clumps: Node,
  /** The share of the rock that carries a stand (0–1): its crowns cover part of it, as a sparse stand's do. */
  rockStand: Node
): { color: Node; share: Node; opening: Node } {
  const { n1, n2, n3, n4, n5, n6 } = n
  const p = positionWorld.xz
  const edgeNoise = n5.mul(0.6).add(n1.mul(0.45)).add(n2.mul(0.3)).clamp(-1, 1)
  const ragged = (share: Node) => smoothstep(0.3, 0.7, share.add(edgeNoise.mul(0.3)))
  const rockW = ragged(w.x)
  const herbW = ragged(w.y)
  const shrubW = ragged(w.z)
  const sparseW = ragged(w.w)
  const total = rockW.add(herbW).add(shrubW).add(sparseW)
  const norm = max(total, 1)

  // Rock: talus streaks run down the face and ledges across it. Across is metres along the view's arc (bearing ×
  // distance: steady over the face, where a coordinate along a per-pixel fall line would jump with every facet's
  // normal), down is height.
  const view = p.sub(cameraPosition.xz)
  const across = atan(view.x, view.y).mul(length(view))
  const height = positionWorld.y
  const streaks = mx_noise_float(vec3(across.div(7), height.div(40), 3.5)).mul(float(1).sub(smoothstep(5000, 9000, d)))
  const ledge = float(1).sub(abs(mx_noise_float(vec3(across.div(40), height.div(7), 7.5))))
  const cracks = smoothstep(0.86, 0.97, ledge).mul(float(1).sub(smoothstep(2500, 6000, d)))
  // The renders' lit rock reads about #b1a187, a photograph's #74746a: warm tan against grey.
  const rockBase = mix(vec3(0.2, 0.192, 0.165), vec3(0.38, 0.315, 0.215), via)
  // Steep faces stand darker than the benches and talus aprons below them (the sky lights them less).
  const slopeShade = mix(float(0.6), float(1), smoothstep(0.3, 0.92, normalWorld.y))
  const stained = smoothstep(0.3, 0.8, n5.mul(0.7).add(n1.mul(0.5)).add(n6.mul(0.3))).mul(0.4)
  const rock = rockBase
    .mul(streaks.mul(0.2).add(n4.mul(0.14)).add(n3.mul(0.15)).add(1))
    .mul(cracks.mul(-0.5).add(1))
    .mul(slopeShade)
    .mul(stained.mul(-0.45).add(1))
  // Shrubs and stunted trees clinging in the cracks: dark olive specks at crown scale (they fade with `n5`).
  const specks = smoothstep(0.42, 0.62, n5.mul(0.8).add(n2.mul(0.4)).add(n3.mul(0.2)))
  const rockFace = mix(rock, mix(vec3(0.025, 0.032, 0.018), vec3(0.03, 0.04, 0.02), via), specks.mul(0.7))

  // Herb: yellow-green grass with tan dry patches (a photo's drier and darker), tufted with shrubs.
  const herbDry = mix(vec3(0.105, 0.092, 0.056), vec3(0.165, 0.142, 0.07), via)
  const herbGreen = mix(vec3(0.078, 0.084, 0.042), vec3(0.12, 0.138, 0.05), via)
  const shrubDark = mix(vec3(0.034, 0.04, 0.024), vec3(0.03, 0.04, 0.016), via)
  const shrubLight = mix(vec3(0.055, 0.06, 0.036), vec3(0.06, 0.072, 0.03), via)
  const tufts = smoothstep(0.45, 0.85, n5.mul(0.6).add(n2.mul(0.5)).add(n3.mul(0.3)))
  const herb = mix(mix(herbGreen, herbDry, smoothstep(-0.3, 0.5, n1.add(n6.mul(0.6)))), shrubDark, tufts.mul(0.55)).mul(
    n3.mul(0.18).add(1)
  )
  // Shrub: dark olive, low and mottled.
  const shrub = mix(shrubDark, shrubLight, smoothstep(-0.4, 0.6, n5.mul(0.7).add(n2.mul(0.5)).add(n1.mul(0.3)))).mul(
    n3.mul(0.15).add(1)
  )
  // Sparse forest: light ground (grass over rock) between scattered crowns' shade.
  const light = mix(herb, rockFace, 0.3)
  const sparse = mix(light, canopy, clumps.mul(0.35).add(0.1))

  const color = rockFace
    .mul(rockW)
    .add(herb.mul(herbW))
    .add(shrub.mul(shrubW))
    .add(sparse.mul(sparseW))
    .div(max(total, 1e-3))
  const opening = rockW.mul(rockStand.mul(-0.45).add(1)).add(herbW).add(shrubW).add(sparseW.mul(0.55)).div(norm)
  return { color, share: total.min(1), opening }
}

/** How far a lake's reflection rays travel, metres: past a far shore and the ridges behind it. */
const LAKE_REFLECTION_REACH = 20_000

/** Where the invented apron ends, metres from the origin. */
const APRON_RADIUS = 7600

/** A vertex on the edge a ring is stitched to: package x, y, its drawn height, and `side + fraction along it`. */
interface EdgePoint {
  x: number
  y: number
  h: number
  t: number
}

/**
 * The package grid's mesh edge (a `PlaneGeometry` laid flat, `n` segments a side), counter-clockwise from its
 * south-west corner as seen from above (x east, y north), closed (the first point again at t = 4).
 */
function meshEdge(mid: THREE.Mesh, n: number): EdgePoint[] {
  const position = mid.geometry.attributes.position
  const edge: EdgePoint[] = []
  const push = (ix: number, iy: number, t: number) => {
    const k = iy * (n + 1) + ix
    edge.push({ x: position.getX(k), y: -position.getZ(k), h: position.getY(k), t })
  }
  // PlaneGeometry rotated flat: column ix runs west to east, row iy runs north (iy = 0) to south (iy = n).
  for (let k = 0; k < n; k++) push(k, n, k / n) // south edge, eastwards
  for (let k = 0; k < n; k++) push(n, n - k, 1 + k / n) // east edge, northwards
  for (let k = 0; k < n; k++) push(n - k, 0, 2 + k / n) // north edge, westwards
  for (let k = 0; k < n; k++) push(0, k, 3 + k / n) // west edge, southwards
  edge.push({ ...edge[0], t: 4 })
  return edge
}

/** A ring mesh's outer edge (its grid's border posts, vertex j · columns + i), as `meshEdge` orders it. */
function gridEdge(ring: THREE.Mesh, grid: TerrainGrid): EdgePoint[] {
  const position = ring.geometry.attributes.position
  const { columns, rows } = grid
  const edge: EdgePoint[] = []
  const push = (i: number, j: number, t: number) => {
    const k = j * columns + i
    edge.push({ x: position.getX(k), y: -position.getZ(k), h: position.getY(k), t })
  }
  const w = columns - 1
  const h = rows - 1
  for (let k = 0; k < w; k++) push(k, 0, k / w)
  for (let k = 0; k < h; k++) push(w, k, 1 + k / h)
  for (let k = 0; k < w; k++) push(w - k, h, 2 + k / w)
  for (let k = 0; k < h; k++) push(0, h - k, 3 + k / h)
  edge.push({ ...edge[0], t: 4 })
  return edge
}

/**
 * A coarse grid (`farTerrain`, or `skylineTerrain`) as one mesh with a hole where the grid inside it is drawn (`hole`,
 * package metres), and a strip stitched from that one's very edge vertices (`edge`) to the hole's edge, so the two meet
 * without a crack or an overlap for the depth buffer to fight over kilometres off. Heights come from `groundAt`, which
 * reads this grid outside the ones inside it. The grid's posts are vertices `j · columns + i`.
 */
function ringMesh(
  grid: TerrainGrid,
  edge: EdgePoint[],
  hole: Extent,
  groundAt: (x: number, y: number) => number,
  openingAt: (x: number, y: number) => Area | undefined,
  coverOf: CoverOf,
  material: THREE.Material,
  /** Draw the grid's water posts (cover 5) as water: the skyline grid, which has no water polygons. */
  water = false
): THREE.Mesh | null {
  const { west, south, spacing, columns, rows } = grid
  // The hole: the grid cells touching the inner grid, and one more round them, so the strip is never degenerate.
  const i0 = Math.floor((hole.west - west) / spacing) - 1
  const i1 = Math.ceil((hole.east - west) / spacing) + 1
  const j0 = Math.floor((hole.south - south) / spacing) - 1
  const j1 = Math.ceil((hole.north - south) / spacing) + 1
  if (i0 < 0 || j0 < 0 || i1 > columns - 1 || j1 > rows - 1) return null
  const count = columns * rows + edge.length
  const positions = new Float32Array(count * 3)
  const kind = new Float32Array(count * 4)
  const coverage = new Float32Array(count * 4)
  const nature = new Float32Array(count * 4)
  const burn = new Float32Array(count * 3)
  const wet = new Float32Array(count)
  if (water && grid.cover?.length === columns * rows)
    for (let k = 0; k < columns * rows; k++) wet[k] = grid.cover[k] === SCENE_COVER.water ? 1 : 0
  const put = (v: number, x: number, h: number, y: number, hidden = false) => {
    positions.set([x, h, -y], v * 3)
    const cut = hidden ? undefined : openingAt(x, y)
    kind.set([cut ? 1 : 0, 0, 0, cut ? 1 : 0.55], v * 4)
    if (!hidden) coverOf(x, y, cut, coverage, v * 4, nature, burn)
  }
  for (let j = 0; j < rows; j++)
    for (let i = 0; i < columns; i++) {
      const x = west + i * spacing
      const y = south + j * spacing
      const inHole = i > i0 && i < i1 && j > j0 && j < j1
      put(j * columns + i, x, inHole ? 0 : groundAt(x, y), y, inHole)
    }
  const index: number[] = []
  for (let j = 0; j < rows - 1; j++)
    for (let i = 0; i < columns - 1; i++) {
      if (i >= i0 && i < i1 && j >= j0 && j < j1) continue
      const a = j * columns + i
      index.push(a, a + 1, a + columns, a + 1, a + columns + 1, a + columns)
    }
  // The inner edge's vertices copied in (closed: the last is the first again), and the hole's edge likewise.
  const inner: Array<{ v: number; t: number }> = []
  let next = columns * rows
  for (let k = 0; k < edge.length - 1; k++) {
    const e = edge[k]
    put(next, e.x, e.h, e.y)
    inner.push({ v: next++, t: e.t })
  }
  inner.push({ v: inner[0].v, t: 4 })
  const outer: Array<{ v: number; t: number }> = []
  const w = i1 - i0
  const h = j1 - j0
  for (let k = 0; k < w; k++) outer.push({ v: j0 * columns + i0 + k, t: k / w })
  for (let k = 0; k < h; k++) outer.push({ v: (j0 + k) * columns + i1, t: 1 + k / h })
  for (let k = 0; k < w; k++) outer.push({ v: j1 * columns + i1 - k, t: 2 + k / w })
  for (let k = 0; k < h; k++) outer.push({ v: (j1 - k) * columns + i0, t: 3 + k / h })
  outer.push({ v: outer[0].v, t: 4 })
  for (let a = 0, b = 0; a < inner.length - 1 || b < outer.length - 1; ) {
    if (b === outer.length - 1 || (a < inner.length - 1 && inner[a + 1].t <= outer[b + 1].t)) {
      index.push(inner[a].v, inner[a + 1].v, outer[b].v)
      a++
    } else {
      index.push(inner[a].v, outer[b + 1].v, outer[b].v)
      b++
    }
  }
  // Every triangle faces up (three.js: counter-clockwise seen from +y).
  for (let t = 0; t < index.length; t += 3) {
    const [a, b, c] = [index[t] * 3, index[t + 1] * 3, index[t + 2] * 3]
    const ux = positions[b] - positions[a]
    const uz = positions[b + 2] - positions[a + 2]
    const vx = positions[c] - positions[a]
    const vz = positions[c + 2] - positions[a + 2]
    if (uz * vx - ux * vz < 0) [index[t + 1], index[t + 2]] = [index[t + 2], index[t + 1]]
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('aGround', new THREE.BufferAttribute(kind, 4))
  geometry.setAttribute('aCover', new THREE.BufferAttribute(coverage, 4))
  geometry.setAttribute('aNature', new THREE.BufferAttribute(nature, 4))
  geometry.setAttribute('aBurn', new THREE.BufferAttribute(burn, 3))
  geometry.setAttribute('aWater', new THREE.BufferAttribute(wet, 1))
  geometry.setIndex(index)
  geometry.computeVertexNormals()
  const m = new THREE.Mesh(geometry, material)
  m.receiveShadow = true
  return m
}

/**
 * A ring of ground from the real terrain's edge (a square `size` metres across) out to `outer` metres, sloping to its
 * mean height and the earth's curve: invented ground, so the view never ends in a cliff edge.
 */
function horizonApron(
  size: number,
  groundAt: (x: number, y: number) => number,
  material: THREE.Material,
  outer = APRON_RADIUS
) {
  const half = size / 2 - 1
  const segments = 160
  const rings = [0, 0.04, 0.12, 0.25, 0.45, 0.7, 1]
  const positions: number[] = []
  const kind: number[] = []
  const coverage: number[] = []
  const edge: number[] = []
  for (let s = 0; s < segments; s++) {
    const a = (s / segments) * Math.PI * 2
    const c = Math.cos(a)
    const n = Math.sin(a)
    const k = half / Math.max(Math.abs(c), Math.abs(n))
    edge.push(groundAt(c * k, n * k))
  }
  const mean = edge.reduce((a, b) => a + b, 0) / edge.length
  for (const t of rings) {
    for (let s = 0; s < segments; s++) {
      const a = (s / segments) * Math.PI * 2
      const c = Math.cos(a)
      const n = Math.sin(a)
      const r0 = half / Math.max(Math.abs(c), Math.abs(n))
      const r = r0 + (outer - r0) * t
      const h = edge[s] + (mean - edge[s]) * Math.min(1, t * 1.6) - (r * r - r0 * r0) / (2 * 6_371_000) - 0.6
      positions.push(c * r, h, -n * r)
      kind.push(0, 0, 0, 0.55)
      coverage.push(0, 0, 0, 0) // closure unknown: the invented ground keeps the far canopy's colour
    }
  }
  const index: number[] = []
  for (let j = 0; j < rings.length - 1; j++)
    for (let s = 0; s < segments; s++) {
      const a = j * segments + s
      const b = j * segments + ((s + 1) % segments)
      index.push(a, b, a + segments, b, b + segments, a + segments)
    }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setAttribute('aGround', new THREE.Float32BufferAttribute(kind, 4))
  geometry.setAttribute('aCover', new THREE.Float32BufferAttribute(coverage, 4))
  geometry.setAttribute('aNature', new THREE.Float32BufferAttribute(new Array(coverage.length).fill(0), 4))
  geometry.setAttribute('aBurn', new THREE.Float32BufferAttribute(new Array((coverage.length / 4) * 3).fill(0), 3))
  geometry.setAttribute('aWater', new THREE.Float32BufferAttribute(new Array(coverage.length / 4).fill(0), 1))
  geometry.setIndex(index)
  geometry.computeVertexNormals()
  // Wound to face up whichever way the angles ran.
  if (geometry.attributes.normal.getY(0) < 0) {
    index.reverse()
    geometry.setIndex(index)
    geometry.computeVertexNormals()
  }
  const m = new THREE.Mesh(geometry, material)
  m.receiveShadow = false
  return m
}

const random = (seed: number) => mulberry32(hash(seed, 0x51a5))

/** Stumps and slash reach this far past the near stand, metres from the eye. */
const DEBRIS_MARGIN = 100

/**
 * Stumps across every opening within `reach` metres of the eye: a jittered grid at about 500 stems a hectare (a
 * harvested pine/spruce stand), fewer where stems were retained. Their cut faces weather from fresh-sawn to grey within
 * a few years. Each opening's stumps are seeded from its own geometry, so the list's order does not move them.
 */
function stumpsOf(openings: Area[], groundAt: (x: number, y: number) => number, ponds: Area[], reach: number) {
  const geometry = new THREE.CylinderGeometry(0.47, 0.55, 1, 10, 1, false)
  geometry.translate(0, 0.5, 0)
  // Cut face pale and fresh-sawn, sides bark; colour per vertex so the instance tint keeps both.
  const pos = geometry.attributes.position
  const normal = geometry.attributes.normal
  const colors = new Float32Array(pos.count * 3)
  const face = new THREE.Color('#bba07a')
  const bark = new THREE.Color('#6b5747')
  for (let i = 0; i < pos.count; i++) (normal.getY(i) > 0.5 ? face : bark).toArray(colors, i * 3)
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  const matrices: THREE.Matrix4[] = []
  const tints: THREE.Color[] = []
  const m = new THREE.Matrix4()
  const q = new THREE.Quaternion()
  const e = new THREE.Euler()
  const step = 4.4
  for (const a of openings) {
    if (!boxWithin(a.box, reach)) continue
    const rand = random(a.seed)
    // Fresh-sawn faces go grey in two or three years; the bark darkens with them.
    const weathered = Math.min(1, Math.max(0, (a.age - 1) / 3))
    for (let x = a.box[0]; x <= a.box[2]; x += step)
      for (let y = a.box[1]; y <= a.box[3]; y += step) {
        const px = x + (rand() - 0.5) * step * 0.9
        const py = y + (rand() - 0.5) * step * 0.9
        const r = 0.11 + rand() * 0.14
        const h = 0.18 + rand() * 0.3
        const tilt = (rand() - 0.5) * 0.12
        const yaw = rand() * Math.PI * 2
        const tone = 0.8 + rand() * 0.35
        const keep = rand() >= 0.12 + 0.88 * a.retained
        if (!keep || Math.hypot(px, py) > reach || !inArea(a, px, py) || ponds.some((p) => inArea(p, px, py))) continue
        e.set(tilt, yaw, tilt * 0.7)
        m.compose(
          new THREE.Vector3(px, groundAt(px, py) - 0.08, -py),
          q.setFromEuler(e),
          new THREE.Vector3(r * 2, h, r * 2)
        )
        matrices.push(m.clone())
        const t = tone * (1 - 0.38 * weathered)
        tints.push(new THREE.Color(t, t * (0.98 + 0.02 * weathered), t * (0.95 + 0.05 * weathered)))
        if (matrices.length >= 40_000) break
      }
  }
  const material = new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.9 })
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, matrices.length))
  mesh.count = matrices.length
  matrices.forEach((matrix, i) => {
    mesh.setMatrixAt(i, matrix)
    mesh.setColorAt(i, tints[i])
  })
  mesh.castShadow = mesh.receiveShadow = true
  mesh.name = 'stumps'
  return mesh
}

/** Whether any of a box lies within `reach` metres of the origin. */
const boxWithin = (box: Area['box'], reach: number) =>
  Math.hypot(Math.max(box[0], 0, -box[2]), Math.max(box[1], 0, -box[3])) <= reach

/**
 * Slash on the cut ground within `reach` metres of the eye. Each package debris pile becomes a heap of logs lying
 * roughly together (some propped on the others), tangled branches and rust-brown dead foliage (gone grey after a couple
 * of years); branches are also strewn thinly over the opening, fewer as it ages and they rot under the regrowth.
 */
function slashOf(
  debris: GroundInput['debris'],
  openings: Area[],
  groundAt: (x: number, y: number) => number,
  inWater: (x: number, y: number) => boolean,
  openingAt: (x: number, y: number) => Area | undefined,
  reach: number
) {
  const logs: number[] = []
  const twigs: number[] = []
  const tufts: number[] = []
  const push = (
    list: number[],
    x: number,
    y: number,
    z: number,
    yaw: number,
    pitch: number,
    roll: number,
    sx: number,
    sy: number,
    sz: number
  ) => list.push(x, y, z, yaw, pitch, roll, sx, sy, sz)
  for (let i = 0; i < debris.count; i++) {
    const [x, y, , footprint] = debris.data.slice(i * 4, i * 4 + 4)
    // Piles are burned or rotted under the regrowth by about fifteen years.
    if (inWater(x, y) || Math.hypot(x, y) > reach || (openingAt(x, y)?.age ?? 0) >= 15) continue
    const rand = random(pileSeed(x, y, footprint))
    const heading = rand() * Math.PI
    const span = Math.max(1.5, footprint)
    const count = 5 + Math.floor(rand() * 6)
    for (let k = 0; k < count; k++) {
      const px = x + (rand() - 0.5) * span * 0.6
      const py = y + (rand() - 0.5) * span * 0.6
      const length = span * (0.45 + rand() * 0.55)
      const r = 0.04 + rand() * 0.09
      const lift = r + rand() * Math.min(0.6, span * 0.12)
      const yaw = heading + (rand() - 0.5) * 0.9
      push(logs, px, groundAt(px, py) + lift, -py, yaw, (rand() - 0.5) * 0.25, 0, r * 2, length, r * 2)
    }
    for (let k = 0; k < 26; k++) {
      const px = x + (rand() - 0.5) * span * 0.8
      const py = y + (rand() - 0.5) * span * 0.8
      const length = 0.8 + rand() * span * 0.45
      push(
        twigs,
        px,
        groundAt(px, py) + 0.1 + rand() * 0.55,
        -py,
        rand() * Math.PI * 2,
        (rand() - 0.5) * 0.9,
        0,
        0.05,
        length,
        0.05
      )
    }
    for (let k = 0; k < 4; k++) {
      const px = x + (rand() - 0.5) * span * 0.6
      const py = y + (rand() - 0.5) * span * 0.6
      const w = span * (0.18 + rand() * 0.2)
      push(
        tufts,
        px,
        groundAt(px, py) + 0.05,
        -py,
        rand() * Math.PI,
        0,
        0,
        w,
        0.25 + rand() * 0.35,
        w * (0.6 + rand() * 0.5)
      )
    }
  }
  // Branches strewn over the openings, a few hundred to the hectare.
  for (const a of openings) {
    if (!boxWithin(a.box, reach) || a.age >= 10) continue
    const rand = random(a.seed ^ 0x5a5a)
    const areaM2 = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1])
    const n = Math.min(30_000, Math.round((areaM2 / 30) * (1 - a.age / 10)))
    for (let i = 0; i < n; i++) {
      const px = a.box[0] + rand() * (a.box[2] - a.box[0])
      const py = a.box[1] + rand() * (a.box[3] - a.box[1])
      const length = 0.6 + rand() * 2.2
      const yaw = rand() * Math.PI * 2
      if (Math.hypot(px, py) > reach || !inArea(a, px, py)) continue
      push(twigs, px, groundAt(px, py) + 0.04, -py, yaw, (rand() - 0.5) * 0.15, 0, 0.04 + rand() * 0.04, length, 0.04)
    }
  }
  const instanced = (
    geometry: THREE.BufferGeometry,
    color: string,
    list: number[],
    lay: boolean,
    variation: number,
    name: string
  ) => {
    const count = list.length / 9
    const material = new THREE.MeshStandardNodeMaterial({ color, roughness: 0.95 })
    const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count))
    mesh.count = count
    const m = new THREE.Matrix4()
    const q = new THREE.Quaternion()
    const e = new THREE.Euler()
    const c = new THREE.Color()
    for (let i = 0; i < count; i++) {
      const o = i * 9
      // Tint from the piece's own position, not its index: a pile or opening gone from the list recolours nothing else.
      const rand = random(hash(Math.floor(list[o] * 1000), Math.floor(list[o + 2] * 1000), 0x7a11))
      // Logs and twigs lie along their length: the cylinder's y axis turned down to horizontal, then yawed.
      e.set(0, list[o + 3], lay ? Math.PI / 2 + list[o + 4] : 0, 'YXZ')
      m.compose(
        new THREE.Vector3(list[o], list[o + 1], list[o + 2]),
        q.setFromEuler(e),
        new THREE.Vector3(list[o + 6], list[o + 7], list[o + 8])
      )
      mesh.setMatrixAt(i, m)
      const t = 1 + (rand() - 0.5) * variation
      mesh.setColorAt(i, c.setRGB(t, t * (1 - (rand() - 0.5) * 0.08), t * 0.97))
    }
    mesh.castShadow = mesh.receiveShadow = true
    mesh.name = name
    return mesh
  }
  const log = new THREE.CylinderGeometry(0.5, 0.5, 1, 7, 1)
  const twig = new THREE.CylinderGeometry(0.3, 0.5, 1, 4, 1)
  const tuft = new THREE.IcosahedronGeometry(0.5, 1)
  {
    // Lumpy, flattened clumps of dead needles and fine branches.
    const p = tuft.attributes.position
    const rand = random(77)
    for (let i = 0; i < p.count; i++) {
      const s = 0.75 + rand() * 0.5
      p.setXYZ(i, p.getX(i) * s, Math.max(-0.1, p.getY(i)) * s, p.getZ(i) * s)
    }
    tuft.computeVertexNormals()
  }
  return [
    instanced(log, '#8a7c69', logs, true, 0.35, 'slash logs'),
    instanced(twig, '#6e5d4c', twigs, true, 0.4, 'slash branches'),
    instanced(tuft, '#6a4a30', tufts, false, 0.45, 'slash foliage')
  ]
}

/** The package's water bodies as threejs-water lakes, rasterised into a lake mask at a cell size that suits them. */
async function lakesOf(
  ponds: Pond[],
  o: { renderer: THREE.WebGPURenderer; camera: THREE.Camera; sun: THREE.DirectionalLight; sky?: THREE.Texture | null }
): Promise<NonNullable<Ground['water']>> {
  const box = [Infinity, Infinity, -Infinity, -Infinity]
  for (const p of ponds) {
    box[0] = Math.min(box[0], p.box[0])
    box[1] = Math.min(box[1], p.box[1])
    box[2] = Math.max(box[2], p.box[2])
    box[3] = Math.max(box[3], p.box[3])
  }
  const biggest = Math.max(...ponds.map((p) => Math.max(p.box[2] - p.box[0], p.box[3] - p.box[1])))
  const cellSize = Math.max(1.5, biggest / 400)
  const margin = cellSize * 3
  // Grid in three.js x/z: x east, z south. Row r is z = z0 + r · cellSize.
  const x0 = box[0] - margin
  const z0 = -box[3] - margin
  const cols = Math.ceil((box[2] - box[0] + 2 * margin) / cellSize) + 1
  const rows = Math.ceil((box[3] - box[1] + 2 * margin) / cellSize) + 1
  const lakes: HydrologyResult['lakes'] = []
  const lakeIndex = new Int32Array(cols * rows).fill(-1)
  // The sea claims its cells first; a basemap water polygon lying over it (or over a lake before it) adds only the cells
  // still unclaimed, so no surface is drawn twice.
  const order = ponds.map((p, i) => i).sort((a, b) => Number(ponds[b].sea) - Number(ponds[a].sea))
  /** The pond each drawn lake is (a pond wholly under one before it draws none). */
  const drawn: Pond[] = []
  for (const pi of order) {
    const p = ponds[pi]
    const li = lakes.length
    const c0 = Math.max(0, Math.floor((p.box[0] - x0) / cellSize) - 1)
    const c1 = Math.min(cols - 1, Math.ceil((p.box[2] - x0) / cellSize) + 1)
    const r0 = Math.max(0, Math.floor((-p.box[3] - z0) / cellSize) - 1)
    const r1 = Math.min(rows - 1, Math.ceil((-p.box[1] - z0) / cellSize) + 1)
    const bw = c1 - c0 + 1
    const bh = r1 - r0 + 1
    const mask = new Uint8Array(bw * bh)
    let area = 0
    for (let r = 0; r < bh; r++)
      for (let c = 0; c < bw; c++) {
        const x = x0 + (c0 + c) * cellSize
        const y = -(z0 + (r0 + r) * cellSize)
        if (lakeIndex[(r0 + r) * cols + c0 + c] >= 0 || !inArea(p, x, y)) continue
        mask[r * bw + c] = 1
        lakeIndex[(r0 + r) * cols + c0 + c] = li
        area += cellSize * cellSize
      }
    if (!area) continue
    lakes.push({ level: p.level, area, maxDepth: 4.5, bounds: [c0, r0, bw, bh], mask })
    drawn.push(p)
  }
  const hydrology = {
    grid: { cols, rows, cellSize, x0, z0, heights: new Float64Array(0) },
    lakes,
    lakeIndex,
    rivers: []
  } as unknown as HydrologyResult
  const prepass = new WaterPrepass({ camera: o.camera, resolutionScale: 0.5 })
  // The planar reflection mirrors about the largest lake's level (the others keep the screen-space reflection). Its
  // render holds the whole scene, the far forest included, so the far shore and its cut blocks show in the lake.
  let main = drawn[0] ?? ponds[0]
  lakes.forEach((lake, i) => {
    if (lake.area > lakes[drawn.indexOf(main)].area) main = drawn[i]
  })
  const mirror = reflector({ resolutionScale: 0.5 })
  mirror.target.rotateX(-Math.PI / 2)
  mirror.target.position.set(0, main.level, 0)
  mirror.target.name = 'lake mirror'
  const lakes3 = await Lakes.create({
    hydrology,
    camera: o.camera,
    renderer: o.renderer,
    prepass,
    planar: mirror,
    sun: o.sun,
    rivers: false,
    // Each lake is extended a few cells past its shore, under the banks; the waterline itself is where the carved bed
    // rises through the surface.
    expand: 3,
    // Northern lake water: tea-brown tannins and green, darker than the tropical default.
    absorption: [0.75, 0.42, 0.32],
    scatterColor: '#0f2a26',
    waves: { amplitude: 0.02, wavelength: [0.25, 2.5] },
    roughness: 0.05,
    // Water's own Fresnel: F0 = ((1.333 − 1) / (1.333 + 1))² ≈ 0.02 (three.js's WaterMesh hard-codes 0.3).
    reflectivity: 0.02,
    // The sky reflected is the one drawn behind the scene (its clouds, its horizon), not threejs-water's stock gradient,
    // which read a lake seen across as a flat white-grey sheet.
    environment: o.sky ?? null,
    // No foam: a forest lake does not break at its shore, and the contact foam (drawn wherever geometry stands in
    // front of the surface) outlines every stem seen against the water.
    foam: 0
  })
  lakes3.object.add(mirror.target)
  lakes3.uniforms.planarHeight.value = main.level
  // A lake is seen across kilometres: its reflection rays reach the far shore and the ridges behind it (threejs-water
  // stops them at 250 m and shows the sky in their place).
  lakes3.uniforms.reflectionReach.value = LAKE_REFLECTION_REACH
  const water = new THREE.Box3()
  for (const p of ponds) {
    water.expandByPoint(new THREE.Vector3(p.box[0], p.level - 5, -p.box[3]))
    water.expandByPoint(new THREE.Vector3(p.box[2], p.level + 1, -p.box[1]))
  }
  let slopeVariance = 0
  const u = lakes3.uniforms
  return {
    lakes: lakes3,
    prepass,
    box: water,
    mirror,
    mirrorLevel: main.level,
    setLook(look, windMs) {
      const via = look === 'via'
      const wind = coxMunk(windMs)
      slopeVariance = via ? 0 : wind.slopeVariance
      const calm = Math.min(1, Math.max(0, windMs) / 4)
      lakes3.set({
        roughness: via ? 0.01 : Math.max(0.01, wind.roughness),
        detail: via ? 0 : 0.28 * calm,
        waves: { amplitude: via ? 0 : 0.02 * calm }
      })
      // VIA renders: a mirror darkened toward slate blue, a thin tan band at the shore.
      u.reflectionTintAmount.value = via ? 0.3 : 0
      u.reflectionDarken.value = via ? 0.8 : 1
      u.shoreBand.value = via ? 1 : 0
      u.shoreBandPixels.value = 2.5
      u.shoreDarkening.value = via ? 0.45 : 0.3
      u.planarDistortion.value = via ? 0.004 : 0.02
    },
    updateView(focalPx) {
      const blur = reflectionBlurPx(slopeVariance, focalPx)
      u.reflectionBlur.value.set(blur.horizontal, blur.vertical)
    }
  }
}

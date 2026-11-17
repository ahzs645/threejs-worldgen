// The PGMaps stand as threejs-trees: each species grown by space colonisation in height classes on a fixed ladder
// (pgmaps-tree-bins.ts, so a stem is scaled by little, keeps its branch and leaf-card proportions, and keeps its shape
// whatever other stems a harvest removes), drawn as
// full, simplified and coarse meshes out to a few hundred metres and as baked impostors beyond.
import * as THREE from 'three/webgpu'
import { disposeLayers, FIXED_INSTANCE_STRIDE, hash, hashString, type ScatterLayer } from 'threejs-scatter'
import { createVegetationLayers, generateTreeLevels, mergeTreeGeometry } from 'threejs-trees'
import { binCrownWidth, binSeed, heightBin, TREE_HEIGHT_LADDER } from './pgmaps-tree-bins'
import { cappedWidthScale } from './pgmaps-far-forest-math'
import { snagForm, standingSnagShare } from './pgmaps-burn'
import { keepStem, stemHash } from './pgmaps-nature'
import { recipe } from './pgmaps-tree-recipes'
import { morphology, stemMetadata, type StandTable } from './pgmaps-stand-fields'
import { farStandTiles, setFarCulling } from './pgmaps-far-cards'

/**
 * The grown kind every stem on burnt ground is drawn as (cover 7, pgmaps-burn.ts): a fire-killed snag, bare grey-black
 * wood with dead branch stubs, whatever its species was.
 */
export const SNAG = 'snag'
/** A snag class's crown (stub reach) as a share of its height, before `binCrownWidth`'s widening. */
const SNAG_CROWN_RATIO = 0.07
/** A snag's clear stem below its stubs, as a share of its height. */
const SNAG_CROWN_BASE = 0.32

/** The stems of a package (`trees` block) and its species table. */
export interface StandInput {
  trees: StandTable & { fields: string[]; species: string[]; data: number[] }
  species: Record<string, { crownBase: number; crownRatio?: number; broadleaf: boolean } | undefined>
}

/**
 * Where each level ends, metres from the eye. Full detail (every twig, every leaf card, shadows) to 55 m; simplified
 * (no shadows) to 120 m; a coarse mesh to 190 m, about where a 20–30 m stem shrinks to the impostor's own resolution
 * (146 px a view); impostors beyond.
 */
export const TREE_LOD_METRES = { full: 55, simplified: 120, coarse: 190 }
/** Detail of the simplified and coarse meshes (see `generateTreeLevels`). */
const DETAIL = { simplified: 0.35, coarse: 0.16 }

/**
 * One species' grown height classes, kept between packages while its crown base, crown ratio and draw distance stay
 * the same. Classes are added as stems need them and kept while the species is in use, so a drive that steps into a
 * taller or shorter stand grows only the new class, and a class that comes back is the same tree as before.
 */
interface SpeciesEntry {
  key: string
  /** Built classes by ladder index (`TREE_HEIGHT_LADDER`). */
  bins: Map<number, BinEntry>
  /** Bark and leaf materials every class of the species shares (one pipeline). */
  material: THREE.Material | THREE.Material[] | null
}
interface BinEntry {
  layer: ScatterLayer
  /** The layer's own levels (a layer with no stems loses all but its first, so they are restored on reuse). */
  lods: ScatterLayer['lods']
  impostor: ScatterLayer['impostor']
  /** The grown tree's measured size, for scaling stems to it. */
  measured: { height: number; width: number }
}

const cache = new Map<string, SpeciesEntry>()
/** Layers dropped from the cache while the previous stand may still be drawing them; freed by `disposeRetiredStands`. */
const retired: ScatterLayer[][] = []

/**
 * Builds (or reuses) the layers for a package's stand and fills them with its stems. `renderer` bakes impostors;
 * `null` draws meshes only. Layers no longer used are disposed.
 *
 * Each stem is drawn with the class of its own height on a fixed ladder (`heightBin`), grown at a crown width fixed
 * by the class and the species (`binCrownWidth`) from a seed fixed by them (`binSeed`), then scaled to the stem's own
 * height and crown width. Nothing depends on the other stems, so a harvest leaves every retained stem's shape alone.
 */
export async function standLayers(
  pkg: StandInput,
  renderer: THREE.WebGPURenderer | null,
  maxDistance: number,
  onStatus: (text: string) => void,
  /** Package points where no stem is drawn (water). */
  exclude: (x: number, y: number) => boolean = () => false,
  /**
   * The far stand: stems farther than `radius` from the package origin are drawn as a fixed set of their class's
   * impostors standing on `groundAt` (package metres), spatially tiled and optionally culled, without distance streaming, so a capture's
   * stems kilometres off are all drawn. Without it (or without impostors) every stem goes to the scatter.
   */
  far: { radius: number; groundAt: (x: number, y: number) => number; frustumCulling?: boolean } | null = null,
  /**
   * Years since the fire where a stem stands on burnt ground (cover 7), else NaN: the stem is drawn as a snag
   * ({@link SNAG}, its form from `snagForm`), and left out once fallen (`standingSnagShare`).
   */
  burntAge: (x: number, y: number) => number = () => NaN
): Promise<{ layers: ScatterLayer[]; stems: number; snags: number; far: FarStand | null }> {
  const stride = pkg.trees.fields.length
  const data = pkg.trees.data
  /** Stems by species (or {@link SNAG}), then by ladder class. */
  const bySpecies = new Map<string, Map<number, number[]>>()
  const shapes = new Map<string, ReturnType<typeof morphology> & { name: string }>()
  const rich = pkg.trees.fields.includes('crownBase')
  /** A snag's drawn height and crown width, by stem. */
  const snagged = new Map<number, { height: number; crownWidth: number }>()
  for (let i = 0; i < pkg.trees.count; i++) {
    const o = i * stride
    const x = data[o]
    const y = data[o + 1]
    if (exclude(x, y)) continue
    let s = pkg.trees.species[data[o + 3]]
    let height = data[o + 4]
    const age = burntAge(x, y)
    if (Number.isFinite(age)) {
      if (!keepStem(y, x, standingSnagShare(age))) continue
      const form = snagForm(height, data[o + 5], stemHash(x, y))
      snagged.set(i, form)
      s = SNAG
      height = form.height
    }
    const meta = stemMetadata(pkg.trees, i)
    const shape = morphology(meta, height, pkg.species[s]?.crownBase ?? 0.3)
    if (s === SNAG) {
      shape.crownBase = SNAG_CROWN_BASE
      shape.trunkRatio = 0.0095
    }
    const group = rich ? `${s}|${shape.key}` : s
    shapes.set(group, { ...shape, name: s })
    let bins = bySpecies.get(group)
    if (!bins) bySpecies.set(group, (bins = new Map()))
    const bin = heightBin(height)
    let list = bins.get(bin)
    if (!list) bins.set(bin, (list = []))
    list.push(i)
  }

  const used = new Set<string>()
  const layers: ScatterLayer[] = []
  let stems = 0
  const farMeshes: THREE.Group[] = []
  let farStems = 0
  for (const [group, bins] of bySpecies) {
    const shape = shapes.get(group)!
    const name = shape.name
    const snag = name === SNAG
    const crownBase = rich ? shape.crownBase : snag ? SNAG_CROWN_BASE : (pkg.species[name]?.crownBase ?? 0.3)
    const crownRatio = snag ? SNAG_CROWN_RATIO : pkg.species[name]?.crownRatio
    const key = JSON.stringify([crownBase, crownRatio ?? null, maxDistance])
    let entry = cache.get(group)
    if (entry && entry.key !== key) {
      retired.push([...entry.bins.values()].map((b) => b.layer))
      cache.delete(group)
      entry = undefined
    }
    if (!entry) {
      entry = { key, bins: new Map(), material: null }
      cache.set(group, entry)
    }
    used.add(group)
    const missing = [...bins.keys()].filter((b) => !entry.bins.has(b)).sort((a, b) => a - b)
    if (missing.length) onStatus(`growing ${name} (${missing.map((b) => `${TREE_HEIGHT_LADDER[b]} m`).join(', ')})…`)
    for (const bin of missing)
      entry.bins.set(
        bin,
        await buildBin(
          name,
          bin,
          crownBase,
          binCrownWidth(name, bin, crownRatio),
          entry,
          renderer,
          maxDistance,
          rich ? shape.trunkRatio : undefined,
          rich ? shape.form : 0
        )
      )

    // Classes in ladder order, each stem scaled from its class's measured size to its own.
    for (const [bin, members] of [...bins].sort((a, b) => a[0] - b[0])) {
      const built = entry.bins.get(bin)
      if (!built) continue
      const m = built.measured
      const list: number[] = []
      const farList: number[] = []
      const farRadius2 = far && built.impostor ? far.radius * far.radius : Infinity
      for (const i of members) {
        const o = i * stride
        const yaw = (data[o + 6] * 6.283185 + data[o] * 0.37) % 6.283185
        const into = data[o] * data[o] + data[o + 1] * data[o + 1] > farRadius2 ? farList : list
        const form = snagged.get(i)
        // Rich stands retain each measured crown width; legacy packages keep their earlier shape limit.
        const tall = (form ? form.height : data[o + 4]) / m.height
        const width = (form ? form.crownWidth : data[o + 5]) / m.width
        into.push(data[o], -data[o + 1], yaw, rich ? width : cappedWidthScale(width, tall), tall)
      }
      if (farList.length && far && built.impostor) {
        farMeshes.push(farStandTiles(built.impostor, farList, far.groundAt, `${name}@${TREE_HEIGHT_LADDER[bin]}`))
        farStems += farList.length / FIXED_INSTANCE_STRIDE
      }
      const layer = built.layer
      layer.lods = built.lods
      layer.impostor = built.impostor
      layer.instances = new Float32Array(list)
      stems += list.length / FIXED_INSTANCE_STRIDE
      layers.push(layer)
    }
  }
  for (const [group, entry] of cache)
    if (!used.has(group)) {
      retired.push([...entry.bins.values()].map((b) => b.layer))
      cache.delete(group)
    }
  let farStand: FarStand | null = null
  if (farMeshes.length) {
    const object = new THREE.Group()
    object.name = 'far stand'
    object.add(...farMeshes)
    setFarCulling(object, far?.frustumCulling ?? true)
    farStand = {
      object,
      stems: farStems,
      setCulling(enabled) {
        setFarCulling(object, enabled)
      },
      dispose() {
        object.removeFromParent()
        object.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) (child as THREE.Mesh).geometry.dispose()
        })
      }
    }
  }
  return { layers, stems: stems + farStems, snags: snagged.size, far: farStand }
}

/** The far stand: one impostor mesh per grown class, every stem in it drawn wherever the camera is. */
export interface FarStand {
  object: THREE.Group
  stems: number
  setCulling(enabled: boolean): void
  dispose(): void
}

/** How far the scatter's colour variation reaches (`layer.colorVariation`, set in `buildBin`). */
const COLOR_VARIATION = 0.12
/** How far a stem is sunk into the ground (`layer.sink`, set in `buildBin`). */
const SINK = 0.05

/** Grows one height class of a species. The first class grown sets the materials the species' later classes share. */
async function buildBin(
  name: string,
  bin: number,
  crownBase: number,
  width: number,
  entry: SpeciesEntry,
  renderer: THREE.WebGPURenderer | null,
  maxDistance: number,
  trunkRatio?: number,
  form = 0
): Promise<BinEntry> {
  const height = TREE_HEIGHT_LADDER[bin]
  const params = recipe(name, height, width, crownBase, trunkRatio, form)
  const id = `${name}@${height}`
  const seed = binSeed(name, bin)
  // createVegetationLayers scales its distances by size / 15 m; undo that so the levels end where we say.
  const k = Math.min(3, Math.max(0.12, height / 15))
  const [layer] = await createVegetationLayers({
    species: [{ name: id, params, density: 0, variants: 1 }],
    lodDistances: [TREE_LOD_METRES.full / k, TREE_LOD_METRES.simplified / k, maxDistance / k],
    simplifiedDetail: DETAIL.simplified,
    seed,
    renderer,
    // 7 × 7 views of 128 px: power-of-two cells, so each view keeps its own coverage-preserving mipmaps.
    impostor: { size: 896, frames: 7, mipmaps: 'coverage' },
    castShadow: true
  })
  // A third, coarse mesh between the simplified one and the impostor: the same tree (the seed
  // createVegetationLayers gives its one variant) with fewer cards.
  const [coarse] = generateTreeLevels(params, hash(seed, hashString(id), 0), [DETAIL.coarse])
  const material = layer.lods[0].material
  // Every class of a species has the same bark and leaves, so they share one set of materials (one pipeline).
  if (entry.material) {
    for (const m of Array.isArray(material) ? material : [material]) m.dispose()
    for (const lod of layer.lods) lod.material = entry.material
  } else entry.material = material
  layer.lods = [
    { ...layer.lods[0], distance: TREE_LOD_METRES.full },
    { ...layer.lods[1], distance: TREE_LOD_METRES.simplified, castShadow: false },
    {
      distance: layer.impostor ? TREE_LOD_METRES.coarse : maxDistance,
      geometry: mergeTreeGeometry(coarse),
      material: layer.lods[0].material,
      castShadow: false
    }
  ]
  if (layer.impostor) layer.impostor.distance = maxDistance
  layer.colorVariation = COLOR_VARIATION
  layer.sink = SINK
  const box =
    layer.lods[0].geometry.boundingBox ??
    new THREE.Box3().setFromBufferAttribute(layer.lods[0].geometry.attributes.position as THREE.BufferAttribute)
  return {
    layer,
    lods: layer.lods,
    impostor: layer.impostor,
    measured: { height: box.max.y, width: Math.max(box.max.x - box.min.x, box.max.z - box.min.z) }
  }
}

/** Frees the layers the last `standLayers` call dropped. Call once the stand that drew them has been disposed. */
export function disposeRetiredStands(): void {
  for (const layers of retired.splice(0)) disposeLayers(layers)
}

/** Frees every cached species (page unload). */
export function disposeStandCache(): void {
  disposeRetiredStands()
  for (const entry of cache.values()) disposeLayers([...entry.bins.values()].map((b) => b.layer))
  cache.clear()
}

// The PGMaps stand as threejs-trees: each species grown by space colonisation at the package's own stem sizes (a few
// height classes per species, so a stem is scaled by little and keeps its branch and leaf-card proportions), drawn as
// full, simplified and coarse meshes out to a few hundred metres and as baked impostors beyond.
import * as THREE from 'three/webgpu'
import { disposeLayers, FIXED_INSTANCE_STRIDE, hash, hashString, type ScatterLayer } from 'threejs-scatter'
import {
  createVegetationLayers,
  generateTreeLevels,
  mergeTreeGeometry,
  species as presets,
  type TreeParams
} from 'threejs-trees'

/** The stems of a package (`trees` block) and its species table. */
export interface StandInput {
  trees: { fields: string[]; species: string[]; count: number; data: number[] }
  species: Record<string, { crownBase: number; crownRatio: number; broadleaf: boolean } | undefined>
}

/**
 * Where each level ends, metres from the eye. Full detail (every twig, every leaf card, shadows) to 55 m; simplified
 * (no shadows) to 120 m; a coarse mesh to 190 m, about where a 20–30 m stem shrinks to the impostor's own resolution
 * (146 px a view); impostors beyond.
 */
export const TREE_LOD_METRES = { full: 55, simplified: 120, coarse: 190 }
/** Detail of the simplified and coarse meshes (see `generateTreeLevels`). */
const DETAIL = { simplified: 0.35, coarse: 0.16 }

/** A species' variant: a height class grown at that size. */
interface Variant {
  height: number
  width: number
}

/**
 * Generator settings for PGMaps' species, at a variant's height `h`, crown width `w` and crown base (fraction of the
 * height that is clear stem). Conifers grow a straight leader with needle sprays along drooping or level branches;
 * aspen and birch are broadleaves with pale bark.
 */
function recipe(name: string, h: number, w: number, crownBase: number): TreeParams {
  const r = Math.max(0.6, w / 2)
  const crownHeight = h * (1 - crownBase)
  // Growth steps of about a fiftieth of the height: tall stems get fewer, longer segments (half the tube rings of
  // the presets' 0.33 m), with leaf cards scaled up to keep the crowns as full.
  const step = Math.min(0.55, Math.max(0.32, h / 55))
  const sized = (base: TreeParams, extra: Partial<TreeParams>, shape: TreeParams['crownShape']): TreeParams => {
    // Points sampled in the crown: about five per cubic metre of crown, so narrow crowns still branch finely.
    const volume =
      shape === 'cone' ? (Math.PI * r * r * crownHeight) / 3 : (4 / 3) * Math.PI * r * r * (crownHeight / 2)
    const p: TreeParams = {
      ...base,
      height: h,
      trunkHeight: h * crownBase,
      crownHeight,
      crownRadius: r,
      crownShape: shape,
      attractionPoints: Math.round(Math.min(2000, Math.max(600, volume * 5))),
      influenceDistance: Math.min(base.influenceDistance, Math.max(1.2, r * 0.9)),
      baseRadius: Math.min(0.55, Math.max(0.12, h * 0.012)),
      ...extra
    }
    const k = step / 0.33
    return {
      ...p,
      segmentLength: step,
      killDistance: step * 2,
      iterations: Math.max(140, Math.round(h / step) + 40),
      leafDensity: p.leafDensity * Math.min(1.8, k),
      leafSize: p.leafSize * Math.min(1.25, Math.sqrt(k)),
      leafDepth: Math.max(2, Math.round(p.leafDepth / Math.min(1.6, k)))
    }
  }
  switch (name) {
    case 'pine': // lodgepole: a tall clean stem, a short rounded crown at the top, yellow-green needles
      return sized(
        presets.pine,
        {
          leader: true,
          tropism: [0, 0.05, 0],
          jitter: 0.22,
          leafColor: '#46602c',
          barkColor: '#6b4a33',
          leafSize: 0.85,
          leafDensity: 2.6,
          leafDepth: 5
        },
        'ellipsoid'
      )
    case 'spruce': // interior spruce: long dense cone, drooping branches, blue-green
      return sized(
        presets.spruce,
        { tropism: [0, -0.3, 0], leafColor: '#2f4a38', leafSize: 0.95, leafDensity: 2.5, leafDepth: 8 },
        'cone'
      )
    case 'fir': // subalpine fir: a narrow spire, level branches, dark
      return sized(
        presets.fir,
        { tropism: [0, -0.1, 0], jitter: 0.06, leafColor: '#284438', leafSize: 0.9, leafDensity: 2.6, leafDepth: 9 },
        'cone'
      )
    case 'douglasFir': // interior Douglas-fir: broader and less regular, deeply furrowed bark
      return sized(
        presets.fir,
        {
          tropism: [0, -0.18, 0],
          jitter: 0.2,
          leafColor: '#34512f',
          barkColor: '#5a4232',
          barkContrast: 0.85,
          leafSize: 1,
          leafDensity: 2.4,
          leafDepth: 7
        },
        'cone'
      )
    case 'aspen': // trembling aspen: pale green-white bark, small rounded crown, bright leaves
      return sized(
        presets.birch,
        {
          tropism: [0, 0.06, 0],
          barkColor: '#c6cab5',
          barkMarks: 0.45,
          barkContrast: 0.2,
          leafColor: '#789a3c',
          leafSize: 0.5,
          leafDensity: 2,
          leafDepth: 3
        },
        'ellipsoid'
      )
    case 'birch':
      return sized(presets.birch, { tropism: [0, -0.08, 0], barkColor: '#e0dbd0' }, 'ellipsoid')
    default:
      return sized(presets.spruce, {}, 'cone')
  }
}

/** One species' layers, kept between packages while its variants' sizes and crown base stay the same. */
interface SpeciesEntry {
  key: string
  layers: ScatterLayer[]
  /** Each layer's own levels (a layer with no stems loses all but its first, so they are restored on reuse). */
  lods: ScatterLayer['lods'][]
  impostors: ScatterLayer['impostor'][]
  variants: Variant[]
  /** Each variant's measured size, for scaling stems to it. */
  measured: { height: number; width: number }[]
}

const cache = new Map<string, SpeciesEntry>()
/** Layers dropped from the cache while the previous stand may still be drawing them; freed by `disposeRetiredStands`. */
const retired: ScatterLayer[][] = []

const quantile = (sorted: number[], q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]
const median = (values: number[]) =>
  quantile(
    values.slice().sort((a, b) => a - b),
    0.5
  )

/** Height classes of a species from its stems: three for a big population, fewer for a handful of stems. */
function variantsOf(heights: number[], widths: number[]): Variant[] {
  const sorted = heights.slice().sort((a, b) => a - b)
  const qs = heights.length >= 40 ? [0.2, 0.55, 0.85] : heights.length >= 8 ? [0.3, 0.75] : [0.5]
  // Sizes are rounded (2 m, 0.5 m) so that the next package on a drive usually finds the same variants cached.
  const classes = [...new Set(qs.map((q) => Math.max(4, Math.round(quantile(sorted, q) / 2) * 2)))]
  return classes.map((height) => {
    const near = widths.filter((_, i) => Math.abs(heights[i] - height) <= 2.5)
    const width = near.length ? median(near) : median(widths)
    return { height, width: Math.max(1, Math.round(width * 2) / 2) }
  })
}

/**
 * Builds (or reuses) the layers for a package's stand and fills them with its stems. `renderer` bakes impostors;
 * `null` draws meshes only. Layers no longer used are disposed.
 */
export async function standLayers(
  pkg: StandInput,
  renderer: THREE.WebGPURenderer | null,
  maxDistance: number,
  onStatus: (text: string) => void,
  /** Package points where no stem is drawn (water). */
  exclude: (x: number, y: number) => boolean = () => false
): Promise<{ layers: ScatterLayer[]; stems: number }> {
  const stride = pkg.trees.fields.length
  const data = pkg.trees.data
  const bySpecies = new Map<number, number[]>()
  for (let i = 0; i < pkg.trees.count; i++) {
    if (exclude(data[i * stride], data[i * stride + 1])) continue
    const s = data[i * stride + 3]
    let list = bySpecies.get(s)
    if (!list) bySpecies.set(s, (list = []))
    list.push(i)
  }

  const used = new Set<string>()
  const layers: ScatterLayer[] = []
  let stems = 0
  for (const [speciesIndex, members] of bySpecies) {
    const name = pkg.trees.species[speciesIndex]
    const crownBase = pkg.species[name]?.crownBase ?? 0.3
    const heights = members.map((i) => data[i * stride + 4])
    const widths = members.map((i) => data[i * stride + 5])
    const variants = variantsOf(heights, widths)
    const key = JSON.stringify([crownBase, variants, maxDistance])
    let entry = cache.get(name)
    if (entry && entry.key !== key) {
      retired.push(entry.layers)
      cache.delete(name)
      entry = undefined
    }
    if (!entry) {
      onStatus(`growing ${name} (${variants.length} height classes)…`)
      entry = await buildSpecies(name, crownBase, variants, renderer, maxDistance)
      entry.key = key
      cache.set(name, entry)
    }
    used.add(name)

    // Stems into their nearest height class, each scaled from that variant's measured size to its own.
    const lists = entry.variants.map(() => [] as number[])
    for (const i of members) {
      const o = i * stride
      const h = data[o + 4]
      let best = 0
      for (let v = 1; v < entry.variants.length; v++)
        if (Math.abs(Math.log(entry.variants[v].height / h)) < Math.abs(Math.log(entry.variants[best].height / h)))
          best = v
      const m = entry.measured[best]
      const yaw = (data[o + 6] * 6.283185 + data[o] * 0.37) % 6.283185
      lists[best].push(data[o], -data[o + 1], yaw, data[o + 5] / m.width, h / m.height)
    }
    entry.layers.forEach((layer, v) => {
      layer.lods = entry.lods[v]
      layer.impostor = entry.impostors[v]
      layer.instances = new Float32Array(lists[v])
      if (lists[v].length === 0) layer.lods = layer.lods.slice(0, 1)
      stems += lists[v].length / FIXED_INSTANCE_STRIDE
      layers.push(layer)
    })
  }
  for (const [name, entry] of cache)
    if (!used.has(name)) {
      retired.push(entry.layers)
      cache.delete(name)
    }
  return { layers, stems }
}

async function buildSpecies(
  name: string,
  crownBase: number,
  variants: Variant[],
  renderer: THREE.WebGPURenderer | null,
  maxDistance: number
): Promise<SpeciesEntry> {
  const layers: ScatterLayer[] = []
  const measured: SpeciesEntry['measured'] = []
  let shared: THREE.Material | THREE.Material[] | null = null
  for (let v = 0; v < variants.length; v++) {
    const { height, width } = variants[v]
    const params = recipe(name, height, width, crownBase)
    const id = `${name}@${height}`
    // createVegetationLayers scales its distances by size / 15 m; undo that so the levels end where we say.
    const k = Math.min(3, Math.max(0.12, height / 15))
    const [layer] = await createVegetationLayers({
      species: [{ name: id, params, density: 0, variants: 1 }],
      lodDistances: [TREE_LOD_METRES.full / k, TREE_LOD_METRES.simplified / k, maxDistance / k],
      simplifiedDetail: DETAIL.simplified,
      seed: 7,
      renderer,
      impostor: { size: 1024, frames: 7 },
      castShadow: true
    })
    // A third, coarse mesh between the simplified one and the impostor: the same tree (same seed) with fewer cards.
    const [coarse] = generateTreeLevels(params, hash(7, hashString(id), 0), [DETAIL.coarse])
    const material = layer.lods[0].material
    // Every variant of a species has the same bark and leaves, so they share one set of materials (one pipeline).
    if (shared) {
      for (const m of Array.isArray(material) ? material : [material]) m.dispose()
      for (const lod of layer.lods) lod.material = shared
    } else shared = material
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
    layer.colorVariation = 0.12
    layer.sink = 0.05
    const box =
      layer.lods[0].geometry.boundingBox ??
      new THREE.Box3().setFromBufferAttribute(layer.lods[0].geometry.attributes.position as THREE.BufferAttribute)
    measured.push({ height: box.max.y, width: Math.max(box.max.x - box.min.x, box.max.z - box.min.z) })
    layers.push(layer)
  }
  return {
    key: '',
    layers,
    lods: layers.map((l) => l.lods),
    impostors: layers.map((l) => l.impostor),
    variants,
    measured
  }
}

/** Frees the layers the last `standLayers` call dropped. Call once the stand that drew them has been disposed. */
export function disposeRetiredStands(): void {
  for (const layers of retired.splice(0)) disposeLayers(layers)
}

/** Frees every cached species (page unload). */
export function disposeStandCache(): void {
  disposeRetiredStands()
  for (const entry of cache.values()) disposeLayers(entry.layers)
  cache.clear()
}

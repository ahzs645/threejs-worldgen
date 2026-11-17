import { species as presets, type TreeParams } from 'threejs-trees'
const SNAG = 'snag'

/** Adapter-specific species and bounded mature/young/shrub morphology. */
export function recipe(
  name: string,
  h: number,
  w: number,
  crownBase: number,
  trunkRatio?: number,
  form = 0
): TreeParams {
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
      baseRadius:
        trunkRatio === undefined ? Math.min(0.55, Math.max(0.12, h * 0.012)) : Math.max(0.025, h * trunkRatio),
      ...extra,
      ...(form === 2 ? { leader: false, trunkHeight: h * crownBase, leafDensity: Math.max(2.5, base.leafDensity) } : {})
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
    case 'hemlock':
      return sized(
        presets.fir,
        { tropism: [0, -0.42, 0], jitter: 0.15, leafColor: '#354f32', leafDensity: 2.8 },
        'cone'
      )
    case 'blackSpruce':
      return sized(
        presets.spruce,
        { tropism: [0, -0.22, 0], jitter: 0.12, leafColor: '#30483c', leafDensity: 2.7 },
        'cone'
      )
    case 'cottonwood':
      return sized(
        presets.birch,
        { tropism: [0, 0.08, 0], barkColor: '#716c53', leafColor: '#69873e', leafSize: 0.8 },
        'ellipsoid'
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
    case SNAG: // a fire-killed stem: a bare charred trunk, a few dead branch stubs, no foliage
      return sized(
        presets.deadSnag,
        {
          leader: true,
          tropism: [0, -0.25, 0],
          jitter: 0.45,
          attractionPoints: Math.round(Math.min(300, Math.max(80, h * 8))),
          leafShape: 'none',
          leafDensity: 0,
          barkColor: '#463c44',
          barkContrast: 0.5,
          barkMarks: 0,
          baseRadius: Math.min(0.32, Math.max(0.08, h * 0.0095)),
          tipRadius: 0.025
        },
        'cone'
      )
    default:
      return sized(presets.spruce, {}, 'cone')
  }
}

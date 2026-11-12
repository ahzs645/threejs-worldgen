// A PGMaps forest scene package (https://github.com/ahzs645/PGMaps, src/pages/dev-forestry/scenePackage.ts) drawn
// with threejs-scatter + threejs-trees: the inventory's own stems, on the package's own ground, from its eye.
//   pgmaps-scene.html?scene=<url of a package JSON>   (or pick a file)
//   add &webgl to force the WebGL2 backend, &noimpostors to skip impostor baking
//
// The package is local metres: x east, y north, z up (heights absolute). three.js here is y up with north at −z,
// so a package point (x, y, z) is drawn at (x, z, −y).
import * as THREE from 'three/webgpu'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { FIXED_INSTANCE_STRIDE, Scatter, disposeLayers, type ScatterLayer } from 'threejs-scatter'
import { createVegetationLayers, setTreeEnvironment, species as presets, type TreeParams } from 'threejs-trees'
import { createEnvironment } from './shared'

type ScenePackage = {
  format: 'pgmaps-forest-scene'
  version: 1
  terrain: {
    west: number
    south: number
    spacing: number
    columns: number
    rows: number
    heights: Array<number | null>
  }
  trees: { fields: string[]; species: string[]; count: number; data: number[] }
  species: Record<string, { label: string; crownRatio: number; crownBase: number; broadleaf: boolean }>
  openings: Array<{ rings: Array<Array<[number, number]>> }>
  debris: { count: number; data: number[] }
  roadScars: Array<{ kind: 'cut' | 'fill'; ring: Array<[number, number]> }>
  camera: {
    position: [number, number, number]
    bearingDegrees: number
    pitchDegrees: number
    verticalFovDegrees: number
  }
  treeRadiusMeters: number
}

const params = new URLSearchParams(location.search)
const hud = document.getElementById('hud')!
const status = (text: string) => {
  hud.textContent = text
}

async function loadScene(): Promise<ScenePackage> {
  const url = params.get('scene')
  if (url) return (await fetch(url)).json()
  status('choose a PGMaps scene package (forestry-scene-package.json)')
  const input = document.getElementById('file') as HTMLInputElement
  input.hidden = false
  return new Promise((resolve) => {
    input.onchange = async () => resolve(JSON.parse(await input.files![0].text()))
  })
}

const scenePackage = await loadScene()
if (scenePackage.format !== 'pgmaps-forest-scene' || scenePackage.version !== 1)
  throw new Error('Not a PGMaps forest scene package (v1).')
status('building the ground…')

// ---------------------------------------------------------------- ground
const { west, south, spacing, columns, rows } = scenePackage.terrain
// Gaps (no source DEM there) take the nearest known height along their row, then their column, so the mesh stays
// whole; the trees and the eye are all inside the known area.
const heights = scenePackage.terrain.heights.slice()
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
// A loop, not Math.min(...): spreading a 160,000-cell grid overflows the call stack.
let base = Infinity
for (const h of heights) if (h !== null && h < base) base = h
const at = (column: number, row: number) => heights[row * columns + column] ?? base

/** Ground height at a package point (x east, y north), bilinear. */
function groundAt(x: number, y: number): number {
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
/** The scatter's terrain: three.js (x, z) is package (x, −z). */
const height = (x: number, z: number) => groundAt(x, -z)

function inRing(ring: Array<[number, number]>, x: number, y: number) {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i]
    const [xj, yj] = ring[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}
const cut = (x: number, y: number) =>
  scenePackage.openings.some((o) => inRing(o.rings[0], x, y) && !o.rings.slice(1).some((r) => inRing(r, x, y)))
const scar = (x: number, y: number) => scenePackage.roadScars.some((s) => inRing(s.ring, x, y))

// A finer mesh near the eye, where the ground is read most closely.
function groundMesh(size: number, step: number) {
  const n = Math.round(size / step)
  const geometry = new THREE.PlaneGeometry(size, size, n, n).rotateX(-Math.PI / 2)
  const position = geometry.attributes.position
  const colors = new Float32Array(position.count * 3)
  const floor = new THREE.Color('#3f4f33')
  const soil = new THREE.Color('#9c8664')
  const bank = new THREE.Color('#b5a585')
  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i)
    const y = -position.getZ(i)
    position.setY(i, groundAt(x, y))
    ;(cut(x, y) ? soil : scar(x, y) ? bank : floor).toArray(colors, i * 3)
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geometry.computeVertexNormals()
  return new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 }))
}

const renderer = new THREE.WebGPURenderer({ antialias: true, forceWebGL: params.has('webgl') })
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5))
renderer.setSize(innerWidth, innerHeight)
document.body.appendChild(renderer.domElement)
await renderer.init()

const scene = new THREE.Scene()
const { camera: eye } = scenePackage
const camera = new THREE.PerspectiveCamera(eye.verticalFovDegrees, innerWidth / innerHeight, 0.3, 8000)
camera.position.set(eye.position[0], eye.position[2], -eye.position[1])
const bearing = THREE.MathUtils.degToRad(eye.bearingDegrees)
const pitch = THREE.MathUtils.degToRad(eye.pitchDegrees)
const target = camera.position
  .clone()
  .add(
    new THREE.Vector3(
      Math.sin(bearing) * Math.cos(pitch),
      Math.sin(pitch),
      -Math.cos(bearing) * Math.cos(pitch)
    ).multiplyScalar(100)
  )
const controls = new OrbitControls(camera, renderer.domElement)
controls.target.copy(target)
controls.update()
const env = createEnvironment(renderer, scene, camera)
// The demo's haze is a scenic choice (about 70% at 400 m). A visual impact assessment wants the haze of a clear
// day, which is what PGMaps draws (Koschmieder's law at 60 km visibility); `&haze=` overrides the density.
scene.fog = new THREE.FogExp2(
  scene.fog instanceof THREE.FogExp2 ? scene.fog.color : '#aebfcc',
  Number(params.get('haze') ?? 0.00025)
)

const near = groundMesh(1400, 5)
const far = groundMesh(Math.min(columns, rows) * spacing - spacing, spacing * 2)
far.position.y = -0.3 // under the near mesh where they overlap
near.receiveShadow = far.receiveShadow = true
scene.add(near, far)

// ---------------------------------------------------------------- trees
// PGMaps' species on worldgen's presets: the nearest preset, with the crown base PGMaps draws, so a stem's live
// crown starts where it does in the road view. Width and height are then each stem's own, per instance.
const recipe: Record<string, { preset: keyof typeof presets; params?: Partial<TreeParams> }> = {
  pine: { preset: 'pine' },
  spruce: { preset: 'spruce' },
  fir: { preset: 'fir' },
  douglasFir: {
    preset: 'fir',
    params: { crownShape: 'ellipsoid', tropism: [0, 0.02, 0], jitter: 0.25, leafColor: '#30513a' }
  },
  aspen: { preset: 'birch', params: { barkColor: '#cfd2c3', leafColor: '#6d8f3c' } },
  birch: { preset: 'birch' }
}
const treeSpecies = scenePackage.trees.species
const used = treeSpecies.filter((_, index) => {
  for (let i = 0; i < scenePackage.trees.count; i++) if (scenePackage.trees.data[i * 7 + 3] === index) return true
  return false
})
const variants = 3
status(`generating ${used.length} species × ${variants} variants…`)
const layers: ScatterLayer[] = (await createVegetationLayers({
  species: used.map((name) => {
    const { preset, params: extra } = recipe[name] ?? { preset: 'spruce' as const }
    const p = presets[preset] as TreeParams
    const crownBase = scenePackage.species[name]?.crownBase ?? 0.3
    return {
      name,
      density: 0,
      params: { ...p, ...extra, trunkHeight: p.height * crownBase, crownHeight: p.height * (1 - crownBase) }
    }
  }),
  variants,
  seed: 7,
  renderer: params.has('noimpostors') ? null : renderer,
  castShadow: true
})) as ScatterLayer[]

// Each species' stems, shared out among its variants and scaled from each variant's own size to the stem's height
// and crown width: the same stems, heights and crowns the road view draws.
const stride = scenePackage.trees.fields.length
for (const layer of layers) {
  const [name, variant] = layer.name.split('#')
  const speciesIndex = treeSpecies.indexOf(name)
  const box = new THREE.Box3().setFromBufferAttribute(
    layer.lods[0].geometry.attributes.position as THREE.BufferAttribute
  )
  const tall = box.max.y
  const wide = Math.max(box.max.x - box.min.x, box.max.z - box.min.z)
  const list: number[] = []
  for (let i = 0; i < scenePackage.trees.count; i++) {
    const o = i * stride
    if (scenePackage.trees.data[o + 3] !== speciesIndex) continue
    // A stable variant per stem, from its position.
    const pick =
      Math.abs(Math.floor(scenePackage.trees.data[o] * 7.3 + scenePackage.trees.data[o + 1] * 3.1)) % variants
    if (pick !== Number(variant)) continue
    const yaw = (scenePackage.trees.data[o + 6] * 6.283185) % 6.283185
    list.push(
      scenePackage.trees.data[o],
      -scenePackage.trees.data[o + 1],
      yaw,
      scenePackage.trees.data[o + 5] / wide,
      scenePackage.trees.data[o + 4] / tall
    )
  }
  layer.instances = new Float32Array(list)
  layer.colorVariation = 0.1
  if (list.length === 0) layer.lods = layer.lods.slice(0, 1)
}
console.log(`fixed instances: ${layers.reduce((n, l) => n + (l.instances?.length ?? 0) / FIXED_INSTANCE_STRIDE, 0)}`)

const scatter = Scatter.create({
  camera,
  scene,
  terrain: height,
  layers,
  seed: 1,
  maxDistance: scenePackage.treeRadiusMeters + 50
})

// ---------------------------------------------------------------- slash
// Logs lying on the cut ground, as PGMaps draws its piles: a few per pile, any direction.
const logs: THREE.Matrix4[] = []
const m = new THREE.Matrix4()
for (let i = 0; i < scenePackage.debris.count; i++) {
  const [x, y, , footprint] = scenePackage.debris.data.slice(i * 4, i * 4 + 4)
  for (let k = 0; k < 6; k++) {
    const a = (i * 7 + k * 13) % 17
    const px = x + ((a / 17 - 0.5) * footprint) / 2
    const py = y + ((((a * 5) % 17) / 17 - 0.5) * footprint) / 2
    const length = footprint * (0.25 + (((a * 3) % 17) / 17) * 0.35)
    m.compose(
      new THREE.Vector3(px, groundAt(px, py) + 0.1, -py),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(0, (a / 17) * Math.PI * 2, Math.PI / 2)),
      new THREE.Vector3(1, length, 1)
    )
    logs.push(m.clone())
  }
}
const logMesh = new THREE.InstancedMesh(
  new THREE.CylinderGeometry(0.12, 0.15, 1, 6),
  new THREE.MeshStandardMaterial({ color: '#6f6150', roughness: 1 }),
  logs.length
)
logs.forEach((matrix, i) => {
  logMesh.setMatrixAt(i, matrix)
})
logMesh.castShadow = logMesh.receiveShadow = true
scene.add(logMesh)

setTreeEnvironment({ windStrength: 0.15 })
let frames = 0
let last = performance.now()
let fps = 30
Object.assign(window, { pgmaps: { scene, camera, scatter, ready: false } })
renderer.setAnimationLoop(() => {
  const now = performance.now()
  fps = fps * 0.9 + (1000 / Math.max(1, now - last)) * 0.1
  last = now
  controls.update()
  env.follow(controls.target)
  setTreeEnvironment({ sun: env.sun })
  scatter.update()
  env.render()
  const s = scatter.stats
  status(
    `${fps.toFixed(0)} fps · ${scenePackage.trees.count} stems in the package · instances ${s.instances} · drawn ${s.visible} · meshes ${s.meshes}`
  )
  // `&still`: once the stand has streamed in, stop on that frame (for screenshots and comparisons).
  frames += 1
  if (params.has('still') && frames > 4 && s.instances >= scenePackage.trees.count) {
    renderer.setAnimationLoop(null)
    ;(window as unknown as { pgmaps: { ready: boolean } }).pgmaps.ready = true
  } else if (!params.has('still') && frames === 60)
    (window as unknown as { pgmaps: { ready: boolean } }).pgmaps.ready = true
})
addEventListener('beforeunload', () => {
  scatter.dispose()
  disposeLayers(layers)
})
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(innerWidth, innerHeight)
})

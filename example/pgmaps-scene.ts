// A PGMaps forest scene package (https://github.com/ahzs645/PGMaps, src/pages/dev-forestry/scenePackage.ts) drawn
// with threejs-trees, threejs-scatter, threejs-sky, threejs-water and threejs-weathering: the inventory's own stems,
// on the package's own ground, from its eye, under the sky of a given date and place.
//   pgmaps-scene.html?scene=<url of a package JSON>   (or pick a file)
//   add &webgl to force the WebGL2 backend, &noimpostors to skip impostor baking, &still to stop on a full frame
//   environment for standalone tests: &dateTime=2026-01-15T13:00:00-08:00&latitude=54&longitude=-122.6
//                                      &ground=bare|snow&weather=clear|overcast|rain|snow
//
// Embedded (`?embed`, no `?scene`): a parent page drives it by postMessage.
//   in:  { type: 'pgmaps-scene', scene }            build (or rebuild, in place) from a package
//        { type: 'pgmaps-camera', position: [x, y, z], bearingDegrees, pitchDegrees, verticalFovDegrees }
//                                                    package local metres, z up; kept until a scene is ready
//        { type: 'pgmaps-environment', dateTime, utc, timeZone, sunPosition, latitude, longitude, ground: 'bare' | 'snow',
//          weather: 'clear' | 'overcast' | 'rain' | 'snow', sun: 'true' | 'daylight' }
//                                                    sun from the instant (`utc`, else `dateTime`) and place; with
//                                                    sun 'daylight' (the default) a sun below the horizon lights the
//                                                    scene as at dusk, with 'true' the real sun (night is dark); ground
//                                                    snow, weather; any field may be left out. Kept and applied to
//                                                    every later build; never rebuilds.
//   out: { type: 'pgmaps-engine-hello', engine }    listening, send a scene
//        { type: 'pgmaps-engine-ready', engine }    first full frame of a (new) scene drawn
//        { type: 'pgmaps-engine-error', engine, message }
//        { type: 'pgmaps-engine-stats', engine, drawn, instances, meshes, fps, stems, seatAdjustMaxM, seatAdjustMeanM,
//          sun: { elevation, azimuth, drawnElevation, drawnAzimuth, clockShiftMinutes, exposure, mode } }
//                                                    about once a second. seatAdjust*: |package z − drawn ground| over
//                                                    the drawn stems, m. sun: degrees, the asked and the drawn sun.
// With ?embed the status overlay is hidden and the orbit controls take no pointer input: the parent owns the camera.
//
// The package is local metres: x east, y north, z up (heights absolute). three.js here is y up with north at −z,
// so a package point (x, y, z) is drawn at (x, z, −y). An optional `water` block lists water bodies:
// `[{ rings: [[[x, y], …], …holes], elevationMeters: number | null }]` (null = the lowest ground on the shore).
import * as THREE from 'three/webgpu'
import { pass } from 'three/tsl'
import { bloom } from 'three/addons/tsl/display/BloomNode.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { Scatter } from 'threejs-scatter'
import {
  createAtmosphere,
  DEFAULT_ENVIRONMENT,
  environmentFromParams,
  mergeEnvironment,
  type EnvironmentInput,
  type EnvironmentState
} from './pgmaps-environment'
import { buildGround, type Ground, type GroundInput } from './pgmaps-ground'
import { disposeRetiredStands, disposeStandCache, standLayers, type StandInput } from './pgmaps-trees'

type ScenePackage = GroundInput &
  StandInput & {
    format: 'pgmaps-forest-scene'
    version: 1
    origin?: { lng: number; lat: number; elevationMeters?: number }
    camera: PackageCamera
    treeRadiusMeters: number
  }
type PackageCamera = {
  position: [number, number, number]
  bearingDegrees: number
  pitchDegrees: number
  verticalFovDegrees: number
}

const ENGINE = 'worldgen'
const params = new URLSearchParams(location.search)
const embedded = params.has('embed')
const hud = document.getElementById('hud')!
if (embedded) hud.hidden = true
const status = (text: string) => {
  hud.textContent = text
}
const toParent = (message: Record<string, unknown>) => {
  if (window.parent !== window) window.parent.postMessage({ engine: ENGINE, ...message }, '*')
}

// ---------------------------------------------------------------- the renderer, camera and sky: made once
const renderer = new THREE.WebGPURenderer({ antialias: true, forceWebGL: params.has('webgl') })
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5))
renderer.setSize(innerWidth, innerHeight)
document.body.appendChild(renderer.domElement)
await renderer.init()

const scene = new THREE.Scene()
const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.3, 8000)
const controls = new OrbitControls(camera, renderer.domElement)
if (embedded) controls.enabled = false
const atmosphere = await createAtmosphere(renderer, scene, camera)

const pipeline = new THREE.RenderPipeline(renderer)
const color = pass(scene, camera).getTextureNode('output')
pipeline.outputNode = color.add(bloom(color, 0.08, 0.3, 4)) // subtle: the HDR threshold keeps it to the sun and glints

/** The environment in force: URL parameters, then every `pgmaps-environment` message on top. */
let environment: EnvironmentState = mergeEnvironment(DEFAULT_ENVIRONMENT, environmentFromParams(params))
let origin: { lat: number; lng: number } | null = null
const applyEnvironment = () => {
  atmosphere.apply(environment, origin)
  const state = (window as unknown as { pgmaps?: Record<string, unknown> }).pgmaps
  if (state) state.environment = { ...atmosphere.applied }
}
applyEnvironment()

/** Stands the camera at a package eye: position in package metres, bearing clockwise from north, pitch up. */
function setCamera(eye: PackageCamera) {
  camera.fov = eye.verticalFovDegrees
  camera.updateProjectionMatrix()
  camera.position.set(eye.position[0], eye.position[2], -eye.position[1])
  const bearing = THREE.MathUtils.degToRad(eye.bearingDegrees)
  const pitch = THREE.MathUtils.degToRad(eye.pitchDegrees)
  controls.target
    .copy(camera.position)
    .add(
      new THREE.Vector3(
        Math.sin(bearing) * Math.cos(pitch),
        Math.sin(pitch),
        -Math.cos(bearing) * Math.cos(pitch)
      ).multiplyScalar(100)
    )
  controls.update()
}

// ---------------------------------------------------------------- one package's ground, slash, water and stand
type Built = { scatter: Scatter; ground: Ground; stems: number; seat: SeatRecord; dispose(): void }

/** How far seating moved the stems: |package z − drawn ground| at each drawn stem, metres. */
type SeatRecord = { maxM: number; meanM: number }

/**
 * Stems stand on this engine's own ground (`groundAt`, bilinear on the package's terrain grid), not at the package's
 * `z`: a stem placed at its own `z` floats or sinks wherever the 15 m grid's facets leave the drawn ground. Seating
 * stays; this measures what it moved, over the stems that are drawn (not those standing in water).
 */
function seatRecord(pkg: ScenePackage, ground: Ground): SeatRecord {
  const stride = pkg.trees.fields.length
  const zField = pkg.trees.fields.indexOf('z')
  const data = pkg.trees.data
  let max = 0
  let sum = 0
  let n = 0
  if (zField >= 0)
    for (let i = 0; i < pkg.trees.count; i++) {
      const o = i * stride
      const x = data[o]
      const y = data[o + 1]
      const z = data[o + zField]
      if (!Number.isFinite(z) || ground.inWater(x, y)) continue
      const d = Math.abs(z - ground.groundAt(x, y))
      if (!Number.isFinite(d)) continue
      max = Math.max(max, d)
      sum += d
      n++
    }
  const round = (v: number) => Math.round(v * 100) / 100
  return { maxM: round(max), meanM: round(n ? sum / n : 0) }
}

/**
 * True while impostors bake. The bake binds its atlas as the render target and awaits shader compiles between
 * passes; a frame drawn in that gap lands in the atlas (the sky and haze baked over every far tree), so the frame
 * loop draws nothing until it is done.
 */
let baking = false

async function build(pkg: ScenePackage): Promise<Built> {
  if (pkg?.format !== 'pgmaps-forest-scene' || pkg.version !== 1)
    throw new Error('Not a PGMaps forest scene package (v1).')
  status('building the ground…')
  const maxDistance = pkg.treeRadiusMeters + 50
  const ground = await buildGround(pkg, atmosphere.states, {
    renderer,
    camera,
    sun: atmosphere.sky.sun,
    treeRadius: pkg.treeRadiusMeters
  })
  baking = true
  const stand = await standLayers(
    pkg,
    params.has('noimpostors') ? null : renderer,
    maxDistance,
    status,
    ground.inWater
  ).finally(() => {
    baking = false
  })
  console.log(`fixed instances: ${stand.stems}`)
  const seat = seatRecord(pkg, ground)
  const scatter = Scatter.create({
    camera,
    scene,
    terrain: (x: number, z: number) => ground.groundAt(x, -z),
    layers: stand.layers,
    seed: 1,
    maxDistance,
    // Drawn whole for its first frames, so every level of every species gets its pipeline compiled at load rather than
    // on the first turn of the head (a 0.6 s stall); then culled to the view (CULL_MARGIN, in the frame loop).
    cullMargin: 0,
    cullNear: 35
  })
  return {
    scatter,
    ground,
    stems: stand.stems,
    seat,
    dispose() {
      scatter.dispose()
      ground.dispose()
    }
  }
}

/**
 * Degrees added to each side of the view before stems are left out. A road view sees about a sixth of the circle
 * around it, and the stand is drawn as meshes out to 190 m; the margin covers a turn of the head between re-sorts.
 */
const CULL_MARGIN = 25

// ---------------------------------------------------------------- packages in, one build at a time
let current: Built | null = null
let pending: ScenePackage | null = null
let building = false
/** The latest camera from the parent, applied once a scene is ready (a package carries its own eye otherwise). */
let parentCamera: PackageCamera | null = null
let framesSinceBuild = 0
let announced = false

async function load(pkg: ScenePackage) {
  pending = pkg
  if (building) return
  building = true
  try {
    while (pending) {
      const next = pending
      pending = null
      const built = await build(next)
      current?.dispose()
      disposeRetiredStands() // species the new package no longer uses, now nothing draws them
      current = built
      scene.add(built.ground.group)
      atmosphere.setGround((x, z) => built.ground.groundAt(x, -z))
      const at = next.origin
      origin = at && Number.isFinite(at.lat) && Number.isFinite(at.lng) ? { lat: at.lat, lng: at.lng } : null
      setCamera(parentCamera ?? next.camera)
      framesSinceBuild = 0
      announced = false
      Object.assign(window, {
        pgmaps: {
          scene,
          camera,
          controls,
          scatter: built.scatter,
          ground: built.ground,
          renderer,
          atmosphere,
          ready: false
        }
      })
      applyEnvironment()
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    status(`failed: ${message}`)
    toParent({ type: 'pgmaps-engine-error', message })
    Object.assign(window, { pgmaps: { ready: true, error: message } })
  } finally {
    building = false
  }
}

addEventListener('message', (event: MessageEvent) => {
  const data = event.data as { type?: string; scene?: ScenePackage } & Partial<PackageCamera> & EnvironmentInput
  if (data?.type === 'pgmaps-scene' && data.scene) void load(data.scene)
  else if (data?.type === 'pgmaps-camera' && data.position) {
    parentCamera = {
      position: data.position,
      bearingDegrees: data.bearingDegrees ?? 0,
      pitchDegrees: data.pitchDegrees ?? 0,
      verticalFovDegrees: data.verticalFovDegrees ?? camera.fov
    }
    if (current) setCamera(parentCamera)
  } else if (data?.type === 'pgmaps-environment') {
    // Light, sky and weather only: the stand and ground stay as built.
    environment = mergeEnvironment(environment, data)
    applyEnvironment()
  }
})

async function fromUrlOrFile(): Promise<ScenePackage> {
  const url = params.get('scene')
  if (url) return (await fetch(url)).json()
  status('choose a PGMaps scene package (forestry-scene-package.json)')
  const input = document.getElementById('file') as HTMLInputElement
  input.hidden = false
  return new Promise((resolve) => {
    input.onchange = async () => resolve(JSON.parse(await input.files![0].text()))
  })
}

Object.assign(window, {
  pgmaps: { scene, camera, controls, renderer, atmosphere, ready: false, environment: { ...atmosphere.applied } }
})
if (embedded && !params.get('scene')) {
  status('waiting for a scene from the parent page…')
  toParent({ type: 'pgmaps-engine-hello' })
} else {
  fromUrlOrFile()
    .then(load)
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error)
      status(`failed: ${message}`)
      toParent({ type: 'pgmaps-engine-error', message })
    })
}

// ---------------------------------------------------------------- the frame
let last = performance.now()
let fps = 30
let lastStats = 0
const start = performance.now()
const follow = new THREE.Vector3()
const ahead = new THREE.Vector3()
const frustum = new THREE.Frustum()
const viewProjection = new THREE.Matrix4()
renderer.setAnimationLoop(() => {
  const now = performance.now()
  fps = fps * 0.9 + (1000 / Math.max(1, now - last)) * 0.1
  last = now
  controls.update()
  // Shadows centre a little ahead of the eye, along the ground: the eye's own surroundings, not the far target.
  camera.getWorldDirection(ahead).setY(0)
  if (ahead.lengthSq() < 1e-6) ahead.set(0, 0, -1)
  follow.copy(camera.position).addScaledVector(ahead.normalize(), 30)
  if (baking) return
  atmosphere.update((now - start) / 1000, follow)
  if (current) {
    current.scatter.update()
    const water = current.ground.water
    if (water) {
      water.lakes.update()
      // The water's reflections need the scene's depth from a prepass: only drawn while the water is in view.
      camera.updateMatrixWorld()
      viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
      frustum.setFromProjectionMatrix(viewProjection, camera.coordinateSystem)
      if (frustum.intersectsBox(water.box)) water.prepass.update(renderer, scene)
    }
  }
  pipeline.render()
  if (!current) return
  const s = current.scatter.stats
  status(
    `${fps.toFixed(0)} fps · ${current.stems} stems in the package · instances ${s.instances} · drawn ${s.visible} · meshes ${s.meshes}`
  )
  framesSinceBuild += 1
  if (framesSinceBuild === 3) current.scatter.set({ cullMargin: CULL_MARGIN })
  const state = (window as unknown as { pgmaps: { ready: boolean } }).pgmaps
  const streamed = s.instances >= current.stems
  // A (new) scene's first full frame: every stem streamed in, or long enough that the rest are out of reach.
  if (!announced && framesSinceBuild > 2 && (streamed || framesSinceBuild > 120)) {
    announced = true
    toParent({ type: 'pgmaps-engine-ready' })
  }
  if (now - lastStats > 1000) {
    lastStats = now
    toParent({
      type: 'pgmaps-engine-stats',
      drawn: s.visible,
      instances: s.instances,
      meshes: s.meshes,
      fps: Math.round(fps),
      stems: current.stems,
      seatAdjustMaxM: current.seat.maxM,
      seatAdjustMeanM: current.seat.meanM,
      sun: { ...atmosphere.sunRecord }
    })
  }
  // `&still`: once the stand has streamed in, stop on that frame (for screenshots and comparisons).
  if (params.has('still') && framesSinceBuild > 4 && streamed) {
    renderer.setAnimationLoop(null)
    state.ready = true
  } else if (!params.has('still') && framesSinceBuild === 60) state.ready = true
})
addEventListener('beforeunload', () => {
  current?.dispose()
  disposeStandCache()
  atmosphere.dispose()
})
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(innerWidth, innerHeight)
})

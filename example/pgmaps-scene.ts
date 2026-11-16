// A PGMaps forest scene package (https://github.com/ahzs645/PGMaps, src/pages/dev-forestry/scenePackage.ts) drawn
// with threejs-trees, threejs-scatter, threejs-sky, threejs-water and threejs-weathering: the inventory's own stems,
// on the package's own ground, from its eye, under the sky of a given date and place.
//   pgmaps-scene.html?scene=<url of a package JSON>   (or pick a file)
//   add &webgl to force the WebGL2 backend, &noimpostors to skip impostor baking, &still to stop on a full frame
//   environment for standalone tests: &dateTime=2026-01-15T13:00:00-08:00&latitude=54&longitude=-122.6
//                                      &ground=bare|snow&weather=clear|overcast|rain|snow
//
// Embedded (`?embed`, no `?scene`): a parent page drives it by postMessage.
//   in:  { type: 'pgmaps-scene', scene }            build (or rebuild, in place) from a package; a package that
//                                                    builds the same scene as the one drawn (the same content, but for
//                                                    its camera, environment, origin, createdAt and notes, which are
//                                                    taken from it) keeps that scene and is ready at once
//        { type: 'pgmaps-camera', position: [x, y, z], bearingDegrees, pitchDegrees, verticalFovDegrees }
//                                                    package local metres, z up; kept until a scene is ready
//        { type: 'pgmaps-capture', id, position, frames, classes, zones? }
//                                                    the picture, the ground class pass, the screened pass (trees in
//                                                    front), depth and (with `zones`) a zone pass for each frame
//                                                    (pgmaps-capture.ts); waits for a package that is building
//        { type: 'pgmaps-environment', dateTime, utc, timeZone, sunPosition, latitude, longitude, ground: 'bare' | 'snow',
//          weather: 'clear' | 'overcast' | 'rain' | 'snow', sun: 'true' | 'daylight', cloudCover, visibilityKm }
//                                                    sun from the instant (`utc`, else `dateTime`) and place; with
//                                                    sun 'daylight' (the default) a sun below the horizon lights the
//                                                    scene as at dusk, with 'true' the real sun (night is dark); ground
//                                                    snow, weather; cloudCover (0–1) is drawn whatever the weather
//                                                    says (0: no cloud at all), visibilityKm sets the haze by
//                                                    Koschmieder's law; any field may be left out. Kept and applied to
//                                                    every later build; never rebuilds.
//   out: { type: 'pgmaps-engine-hello', engine, accepts }  listening, send a scene. `accepts`: what PGMaps may send
//                                                    beyond the plain protocol (pgmaps-packed.ts): the package's
//                                                    long numeric arrays as Float64Arrays (NaN for null; turned
//                                                    back on arrival), and a package named by its key alone
//   in:  { type: 'pgmaps-scene', key }              the package built now, when PGMaps sent it with this `key`;
//                                                    else out { type: 'pgmaps-engine-scene-needed', key } and PGMaps
//                                                    sends it whole
//        { type: 'pgmaps-engine-ready', engine }    first full frame of a (new) scene drawn
//        { type: 'pgmaps-engine-error', engine, message }
//        { type: 'pgmaps-engine-stats', engine, drawn, instances, meshes, fps, stems, seatAdjustMaxM, seatAdjustMeanM,
//          sun: { elevation, azimuth, drawnElevation, drawnAzimuth, clockShiftMinutes, exposure, mode },
//          environment: { weather, cloudCover, visibilityKm } }   the sky as drawn
//        { type: 'pgmaps-engine-capture', id, engine, version, frames, scene, environment, maxTreeHeightMeters,
//          missing, missingDetail } or { type: 'pgmaps-engine-capture', id, error }   the reply to a capture
// `&captureTest` (no parent): once the scene is ready, captures one frame at the package's eye with two test class
// layers and leaves the reply, object URLs of its images and an alignment check on `window.pgmaps.lastCapture`;
// `window.pgmaps.capture(request)` runs any request.
//                                                    about once a second. seatAdjust*: |package z − drawn ground| over
//                                                    the drawn stems, m. sun: degrees, the asked and the drawn sun.
// With ?embed the status overlay is hidden and the orbit controls take no pointer input: the parent owns the camera.
//
// A package may also carry `farTerrain` (the terrain grid's layout, coarser, out to the skyline): it is drawn where
// `terrain` has no posts, stitched to its edge, and the invented apron only starts past it. Stems past 600 m (a
// capture's package) are drawn as fixed impostor cards however far they stand.
//
// Burnt ground (`terrain.cover` 7, the fire season per post in `burntYear`, its age in `fires[].yearsSinceFire`) is drawn
// as a VIA render draws a recent fire (pgmaps-burn.ts): the stems there as bare grey-black snags, fewer as the years
// pass, over herb and shrub regrowth; past them a dark mottle streaked by the snags. In the VIA look (`look: 'via'`),
// openings a year old or younger are painted as the renders paint fresh cut ground: pinkish tan with a slash pepper.
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
import { buildFarForest, disposeFarForestAtlases, type FarForest } from './pgmaps-far-forest'
import { fitCameraViewport, type CameraProjection } from './pgmaps-camera'
import { syntheticStand } from './pgmaps-far-forest-math'
import { SCENE_UNBUILT_FIELDS, scenePackageKey } from './pgmaps-capture-math'
import { ACCEPTS, unpackArrays } from './pgmaps-packed'
import { viewYear } from './pgmaps-burn'
import {
  alignmentCheck,
  createCapturer,
  inspectReply,
  testRequest,
  transferOf,
  type CaptureReply,
  type CaptureRequest
} from './pgmaps-capture'

type ScenePackage = GroundInput &
  StandInput & {
    format: 'pgmaps-forest-scene'
    version: 1
    origin?: { lng: number; lat: number; elevationMeters?: number }
    createdAt?: string
    camera: PackageCamera
    environment?: EnvironmentInput
    treeRadiusMeters: number
  }
type PackageCamera = {
  position: [number, number, number]
  bearingDegrees: number
  pitchDegrees: number
  verticalFovDegrees: number
  projection?: CameraProjection
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
const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.3, 8000) // LIVE_FAR
const controls = new OrbitControls(camera, renderer.domElement)
if (embedded) controls.enabled = false
const atmosphere = await createAtmosphere(renderer, scene, camera)

const pipeline = new THREE.RenderPipeline(renderer)
const color = pass(scene, camera).getTextureNode('output')
pipeline.outputNode = color.add(bloom(color, 0.08, 0.3, 4)) // subtle: the HDR threshold keeps it to the sun and glints

/** The scene drawn now (one package's ground, stand, far forest and water). */
let current: Built | null = null
/** The environment in force: URL parameters, then every `pgmaps-environment` message on top. */
let environment: EnvironmentState = mergeEnvironment(DEFAULT_ENVIRONMENT, environmentFromParams(params))
let environmentOverridden = Object.values(environmentFromParams(params)).some((value) => value !== undefined)
let origin: { lat: number; lng: number } | null = null
const applyEnvironment = () => {
  atmosphere.apply(environment, origin)
  current?.farForest.setLook(environment.look)
  current?.ground.setLook(environment.look)
  current?.ground.water?.setLook(environment.look, environment.windMs)
  const state = (window as unknown as { pgmaps?: Record<string, unknown> }).pgmaps
  if (state) state.environment = { ...atmosphere.applied }
}
applyEnvironment()

let savedProjection: CameraProjection | undefined
let viewWidth = innerWidth
let viewHeight = innerHeight
const resize = () => {
  const view = fitCameraViewport(savedProjection, innerWidth, innerHeight)
  camera.aspect = view.aspect
  camera.updateProjectionMatrix()
  if (viewWidth !== view.width || viewHeight !== view.height) {
    renderer.setSize(view.width, view.height)
    viewWidth = view.width
    viewHeight = view.height
  }
  Object.assign(renderer.domElement.style, { position: 'absolute', left: `${view.left}px`, top: `${view.top}px` })
}

/** Stands the camera at a package eye: position in package metres, bearing clockwise from north, pitch up. */
function setCamera(eye: PackageCamera) {
  savedProjection = eye.projection
  camera.fov = eye.verticalFovDegrees
  resize()
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
type Built = {
  pkg: ScenePackage
  /**
   * What the package built (`scenePackageKey`): a package arriving with the same key keeps this scene. Null until a
   * second package arrives to be told from it: the first of a page is built without being hashed.
   */
  key: string | null
  scatter: Scatter
  ground: Ground
  /**
   * Everything past the scatter's stems: stand patches (slabs) to the dissolve distance, the canopy surface past it, and
   * single cards for the package's far stems on open ground (pgmaps-far-forest.ts).
   */
  farForest: FarForest
  /** Stems the scatter draws (the package's within its near stand, on dry ground). */
  stems: number
  seat: SeatRecord
  maxDistance: number
  dispose(): void
}

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

async function build(pkg: ScenePackage, key: string | null): Promise<Built> {
  if (pkg?.format !== 'pgmaps-forest-scene' || pkg.version !== 1)
    throw new Error('Not a PGMaps forest scene package (v1).')
  status('building the ground…')
  // The road view's package (stems to 600 m) streams every stem through the scatter, as it always has. A capture's
  // package reaches kilometres: its stems past NEAR_STAND go to the far stand, fixed impostor cards drawn however far.
  const nearStand = pkg.treeRadiusMeters > NEAR_STAND + 50 ? NEAR_STAND : pkg.treeRadiusMeters
  const maxDistance = nearStand + 50
  const ground = await buildGround(pkg, atmosphere.states, {
    renderer,
    camera,
    sun: atmosphere.sky.sun,
    treeRadius: nearStand,
    sky: atmosphere.skyTexture
  })
  baking = true
  // The scatter takes the stems within the near stand; the far forest stands for every one past it.
  const near2 = nearStand * nearStand
  const farReach = nearStand < pkg.treeRadiusMeters
  const farStems: number[] = []
  {
    const stride = pkg.trees.fields.length
    for (let i = 0; i < pkg.trees.count; i++) {
      const x = pkg.trees.data[i * stride]
      const y = pkg.trees.data[i * stride + 1]
      if (farReach && x * x + y * y > near2 && !ground.inWater(x, y) && !ground.onBare(x, y)) farStems.push(i)
    }
  }
  const stand = await standLayers(
    pkg,
    params.has('noimpostors') ? null : renderer,
    maxDistance,
    status,
    // Stems on rock or water posts (the package's cover codes) are not drawn: the picture shows the bluff.
    (x, y) => ground.inWater(x, y) || ground.onBare(x, y) || (farReach && x * x + y * y > near2),
    null,
    // Stems on burnt posts (cover 7) stand as snags (pgmaps-burn.ts).
    (x, y) => ground.burntAge(x, y)
  ).finally(() => {
    baking = false
  })
  status('baking the far forest…')
  const farForest = buildFarForest(pkg, ground, farReach ? nearStand : pkg.treeRadiusMeters, farStems)
  ground.group.add(farForest.object)
  console.log(`fixed instances: ${stand.stems} (${stand.snags} snags on burnt ground)`)
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
    pkg,
    key,
    scatter,
    ground,
    farForest,
    stems: stand.stems,
    seat,
    maxDistance,
    dispose() {
      scatter.dispose()
      farForest.dispose()
      ground.dispose()
    }
  }
}

/** A capture not answered by now is answered with an error (PGMaps waits 180 s). */
const CAPTURE_WATCHDOG_MS = 170_000

/** Stems nearer than this to the package's eye are streamed by the scatter (the road view's own 600 m stand). */
const NEAR_STAND = 600
/** The live camera's far plane, metres; a package with skyline terrain moves it out to the apron's edge. */
const LIVE_FAR = 8000

/**
 * Degrees added to each side of the view before stems are left out. A road view sees about a sixth of the circle
 * around it, and the stand is drawn as meshes out to 190 m; the margin covers a turn of the head between re-sorts.
 */
const CULL_MARGIN = 25

// ---------------------------------------------------------------- packages in, one build at a time
let pending: ScenePackage | null = null
let building = false
/** The latest camera from the parent, applied once a scene is ready (a package carries its own eye otherwise). */
let parentCamera: PackageCamera | null = null
let framesSinceBuild = 0
let announced = false

// ---------------------------------------------------------------- captures (pgmaps-capture.ts)
/** True while a capture holds the camera and the renderer; the frame loop draws nothing. */
let capturing = false
const capturer = createCapturer({
  renderer,
  scene,
  camera,
  pipeline,
  atmosphere,
  async settled() {
    while (building || pending) await new Promise((resolve) => setTimeout(resolve, 50))
    return current
  },
  hold(on) {
    capturing = on
  },
  prepare: prepareFrame
})

/**
 * `window.pgmaps.capture(request?)`: runs a capture (the test request at the package's eye when none is given) and
 * keeps the reply, object URLs of its images and the alignment check on `window.pgmaps.lastCapture`. Returns what
 * survives a structured clone (no blobs or buffers).
 */
async function captureForPage(request?: CaptureRequest, options: { size?: [number, number]; frames?: number } = {}) {
  if (!current) throw new Error('No scene is loaded to capture.')
  const asked = request ?? testRequest(current.pkg, options.size, options.frames)
  const started = performance.now()
  const reply: CaptureReply = await capturer.capture(asked)
  const ms = Math.round(performance.now() - started)
  const urls = await inspectReply(reply)
  const ground = current.ground
  const check = alignmentCheck(asked, reply, ground.groundAt)
  const summary = {
    ms,
    captureMs: Math.round(capturer.lastTimingMs),
    final: reply.final ?? null,
    request: asked,
    version: reply.version,
    engine: reply.engine,
    scene: reply.scene,
    environment: reply.environment,
    maxTreeHeightMeters: reply.maxTreeHeightMeters,
    missing: reply.missing,
    missingDetail: reply.missingDetail,
    frames: reply.frames.map((f, i) => ({ width: f.width, height: f.height, camera: f.camera, urls: urls[i] })),
    check
  }
  const state = (window as unknown as { pgmaps: Record<string, unknown> }).pgmaps
  state.lastCapture = { ...summary, reply }
  return summary
}

/**
 * `window.pgmaps.syntheticStand(request, density?)`: a real-density stand over a request's frames, for testing `final`
 * captures before a parent sends one: stems at `density` × closure per hectare (800 by default) on the package's own
 * closure, species mix and heights, out to where its median tree is 2 px tall in the frames.
 */
function syntheticStandFor(request: CaptureRequest, density = 800) {
  if (!current) throw new Error('No scene is loaded.')
  const { pkg, farForest, ground } = current
  const spans = request.frames.map((f) => [f.bearingDegrees - f.horizontalFovDegrees / 2 - 2, f.bearingDegrees + f.horizontalFovDegrees / 2 + 2])
  const first = spans[0][0]
  const unwrap = (b: number) => first + ((((b - first) % 360) + 540) % 360) - 180
  const from = Math.min(...spans.map(([a]) => unwrap(a)))
  const to = Math.max(...spans.map(([, b]) => unwrap(b)))
  const radius = Math.max(...request.frames.map((f) => farForest.cutoffFor(f.height, f.verticalFovDegrees, 2)))
  const fields = pkg.trees.fields
  const stride = fields.length
  const [is, ih] = ['species', 'height'].map((k) => fields.indexOf(k))
  const mix = pkg.trees.species.map((species) => ({ species, share: 0, height: 0 }))
  for (let i = 0; i < pkg.trees.count; i++) {
    const m = mix[pkg.trees.data[i * stride + is]]
    if (!m) continue
    m.share++
    m.height += pkg.trees.data[i * stride + ih]
  }
  for (const m of mix) m.height = m.share ? m.height / m.share : 20
  return syntheticStand(request.position, [from, to], radius, farForest.closureAt, ground.groundAt, mix, { density })
}

if (params.has('captureTest')) {
  void (async () => {
    // Polls from module load, before `window.pgmaps` exists.
    const state = () => (window as unknown as { pgmaps?: { ready?: boolean } }).pgmaps
    while (!state()?.ready || !current) await new Promise((resolve) => setTimeout(resolve, 200))
    try {
      let asked: CaptureRequest | undefined
      const scene = current as Built | null
      if (params.has('final') && scene) {
        asked = testRequest(scene.pkg)
        asked.quality = 'final'
        asked.samples = Number(params.get('samples') ?? 16)
        asked.stand = syntheticStandFor(asked)
      }
      const summary = await captureForPage(asked)
      status(`capture test: ${summary.ms} ms, ${(summary.check.offShare * 100).toFixed(2)}% of ground pixels off class`)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      ;(window as unknown as { pgmaps: Record<string, unknown> }).pgmaps.lastCapture = { error: message }
      status(`capture test failed: ${message}`)
    }
  })()
}

/**
 * PGMaps' own key for a package it sent (`{ type: 'pgmaps-scene', scene, key }`), by the package object. A later
 * `{ type: 'pgmaps-scene', key }` with no package means the one built now, when that is the package of that key;
 * otherwise the page asks for it whole (`pgmaps-engine-scene-needed`). A held package is never built from again.
 */
const postedKeys = new WeakMap<ScenePackage, string>()

async function load(pkg: ScenePackage) {
  pending = pkg
  if (building) return
  building = true
  try {
    while (pending) {
      const next = pending
      pending = null
      // PGMaps sends the package with every job, a structured clone each time: the same content (every array, every
      // field a build reads) keeps the scene that is built, and only what a build never reads is taken from it.
      // Nothing is hashed until there is a built scene to tell the package from (then both are, the built one once).
      const hashStarted = performance.now()
      let key: string | null = null
      // (the built package itself, named by PGMaps' key alone: nothing to hash or to copy)
      const itself = current !== null && next === current.pkg
      if (current !== null && !itself) {
        current.key ??= scenePackageKey(current.pkg, viewYear(current.pkg))
        key = scenePackageKey(next, viewYear(next))
      }
      const hashMs = performance.now() - hashStarted
      const kept = itself || (current !== null && current.key === key)
      if (current !== null && kept) {
        const into = current.pkg as unknown as Record<string, unknown>
        const from = next as unknown as Record<string, unknown>
        if (!itself) {
          for (const field of SCENE_UNBUILT_FIELDS) into[field] = from[field]
          // PGMaps' key for the package now drawn is the arriving one's (it names these fields too).
          const posted = postedKeys.get(next)
          if (posted === undefined) postedKeys.delete(current.pkg)
          else postedKeys.set(current.pkg, posted)
        }
        console.log(`[scene] the same package again: the built scene is kept (hashed in ${hashMs.toFixed(0)} ms)`)
      } else {
        const built = await build(next, key)
        current?.dispose()
        capturer.forgetMaterials() // class-pass copies of the old stand's materials
        disposeRetiredStands() // species the new package no longer uses, now nothing draws them
        current = built
        scene.add(built.ground.group)
        camera.far =
          built.ground.farExtent || built.ground.skylineExtent
            ? Math.max(LIVE_FAR, built.ground.apronRadius + 1000)
            : LIVE_FAR
        camera.updateProjectionMatrix()
        atmosphere.setGround((x, z) => built.ground.groundAt(x, -z))
      }
      const built: Built = current
      const at = next.origin
      origin = at && Number.isFinite(at.lat) && Number.isFinite(at.lng) ? { lat: at.lat, lng: at.lng } : null
      if (!environmentOverridden) environment = mergeEnvironment(DEFAULT_ENVIRONMENT, next.environment ?? {})
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
          farForest: built.farForest,
          capture: captureForPage,
          syntheticStand: syntheticStandFor,
          ready: false,
          /** Whether this package kept the scene already built, and what telling took (the content hash), ms. */
          package: { kept, hashMs: Math.round(hashMs) }
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
  const data = event.data as { type?: string; scene?: ScenePackage; key?: unknown } & Partial<PackageCamera> &
    EnvironmentInput
  if (data?.type === 'pgmaps-scene' && data.scene) {
    // PGMaps may send the long numeric arrays packed (pgmaps-packed.ts): back to the plain arrays a build reads.
    unpackArrays(data.scene)
    if (typeof data.key === 'string') postedKeys.set(data.scene, data.key)
    parentCamera = null
    void load(data.scene)
  } else if (data?.type === 'pgmaps-scene' && typeof data.key === 'string') {
    // The key alone: the package built now, if that is the one PGMaps sent under this key (else it is asked for;
    // so is one named during a build, which takes its turn whole like any other).
    if (current && !building && !pending && postedKeys.get(current.pkg) === data.key) {
      parentCamera = null
      void load(current.pkg)
    } else toParent({ type: 'pgmaps-engine-scene-needed', key: data.key })
  } else if (data?.type === 'pgmaps-camera' && data.position) {
    parentCamera = {
      position: data.position,
      bearingDegrees: data.bearingDegrees ?? 0,
      pitchDegrees: data.pitchDegrees ?? 0,
      verticalFovDegrees: data.verticalFovDegrees ?? camera.fov,
      projection: data.projection
    }
    if (current) setCamera(parentCamera)
  } else if (data?.type === 'pgmaps-capture') {
    const request = event.data as CaptureRequest
    const source = (event.source as Window | null) ?? window.parent
    const targetOrigin = event.origin && event.origin !== 'null' ? event.origin : '*'
    // Always a reply, and before PGMaps' own 180 s timeout: a capture that has not finished by the watchdog is answered
    // with an error (its late result is dropped).
    let answered = false
    const watchdog = setTimeout(() => {
      answered = true
      source.postMessage(
        { type: 'pgmaps-engine-capture', id: request.id, error: `The capture took longer than ${CAPTURE_WATCHDOG_MS / 1000} s.` },
        { targetOrigin }
      )
    }, CAPTURE_WATCHDOG_MS)
    // Steps done of the total while it renders, for PGMaps' time-left estimate (dropped once answered).
    const progress = (done: number, total: number, stage: string) => {
      if (!answered) source.postMessage({ type: 'pgmaps-capture-progress', id: request.id, done, total, stage }, { targetOrigin })
    }
    capturer.capture(request, progress).finally(() => clearTimeout(watchdog)).then(
      (reply) => {
        // What the parent got, for checking from the page (`window.pgmaps.lastCapture`); buffers are transferred.
        const state = (window as unknown as { pgmaps?: Record<string, unknown> }).pgmaps
        if (state)
          state.lastCapture = {
            ...capturer.lastSummary,
            request: { id: request.id, position: request.position, frames: request.frames, classes: request.classes }
          }
        if (!answered) source.postMessage(reply, { targetOrigin, transfer: transferOf(reply) })
        answered = true
      },
      (error) =>
        !answered &&
        source.postMessage(
          {
            type: 'pgmaps-engine-capture',
            id: request.id,
            error: error instanceof Error ? error.message : String(error)
          },
          { targetOrigin }
        )
    )
  } else if (data?.type === 'pgmaps-environment') {
    environmentOverridden = true
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
  pgmaps: {
    scene,
    camera,
    controls,
    renderer,
    atmosphere,
    capture: captureForPage,
    syntheticStand: syntheticStandFor,
    ready: false,
    environment: { ...atmosphere.applied }
  }
})
if (embedded && !params.get('scene')) {
  status('waiting for a scene from the parent page…')
  toParent({ type: 'pgmaps-engine-hello', accepts: [...ACCEPTS] })
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
const sunDirection = new THREE.Vector3()
const bufferSize = new THREE.Vector2()
/** Half the side of the sun's shadow box round the eye in the live view, metres. */
const LIVE_SHADOW_HALF = 55
/**
 * Everything drawn before the picture, at the camera as it stands: sky, light, shadows, the stand, the far forest's
 * dissolve distance and the water. `fit` (a capture's `final`) centres the sun's shadow box on the frame's own near
 * frustum and sizes it to hold it.
 */
function prepareFrame(now: number, fit?: { centre: THREE.Vector3; radius: number }) {
  if (fit) follow.copy(fit.centre)
  else {
    // Shadows centre a little ahead of the eye, along the ground: the eye's own surroundings, not the far target.
    camera.getWorldDirection(ahead).setY(0)
    if (ahead.lengthSq() < 1e-6) ahead.set(0, 0, -1)
    follow.copy(camera.position).addScaledVector(ahead.normalize(), 30)
  }
  atmosphere.update((now - start) / 1000, follow)
  const sun = atmosphere.sky.sun
  const half = fit ? fit.radius : LIVE_SHADOW_HALF
  const shadowCamera = sun.shadow.camera
  if (shadowCamera.right !== half) {
    Object.assign(shadowCamera, { left: -half, right: half, top: half, bottom: -half })
    shadowCamera.updateProjectionMatrix()
  }
  sunDirection.copy(sun.position).sub(sun.target.position).normalize()
  renderer.getDrawingBufferSize(bufferSize)
  const focal = bufferSize.y / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)
  if (current) {
    current.scatter.update()
    const far = current.farForest
    far.update(sunDirection)
    // The dissolve distance follows the frame: a median tree under 4 px tall (a capture pins its own).
    if (!far.pinned) far.setCutoff(far.cutoffFor(bufferSize.y, camera.fov))
    const water = current.ground.water
    if (water) {
      water.lakes.update()
      water.updateView(focal)
      // The water's reflections need the scene's depth from a prepass: only drawn while the water is in view.
      camera.updateMatrixWorld()
      viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
      frustum.setFromProjectionMatrix(viewProjection, camera.coordinateSystem)
      if (frustum.intersectsBox(water.box)) water.prepass.update(renderer, scene)
    }
  }
}
renderer.setAnimationLoop(() => {
  // A capture has the camera and the renderer: the canvas keeps its last frame meanwhile.
  if (capturing) return
  const now = performance.now()
  fps = fps * 0.9 + (1000 / Math.max(1, now - last)) * 0.1
  last = now
  controls.update()
  if (baking) return
  prepareFrame(now)
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
      sun: { ...atmosphere.sunRecord },
      environment: {
        weather: atmosphere.applied.state.weather,
        cloudCover: atmosphere.applied.cloudCover,
        visibilityKm: atmosphere.applied.visibilityKm,
        haze: atmosphere.applied.haze
      }
    })
  }
  // `&still`: once the stand has streamed in, stop on that frame (for screenshots and comparisons).
  if (params.has('still') && framesSinceBuild > 4 && streamed) {
    renderer.setAnimationLoop(null)
    state.ready = true
  } else if (!params.has('still') && framesSinceBuild === 60) state.ready = true
})
addEventListener('beforeunload', () => {
  capturer.dispose()
  current?.dispose()
  disposeStandCache()
  disposeFarForestAtlases()
  atmosphere.dispose()
})
addEventListener('resize', resize)

import type { RenderOptions } from './pgmaps-render-options'
// `pgmaps-capture`: PGMaps asks for the frames its analysis sheet is framed on (PGMaps
// src/pages/dev-forestry/engineCapture.ts). For each frame, from one scene and one camera:
//   image   the scene as drawn now (environment, weather, post), rendered offscreen at the frame's size, PNG;
//   classes the ground class pass: the same camera, the class layers draped flat on the terrain (textures over the
//           package's terrain grid and, coarser, over its skyline terrain; nearest), water in `water`, no trees,
//           stumps or slash at all; nearer terrain hides farther; the sky transparent. Straight RGBA bytes. Outlines
//           drawn from it follow the ground polygons over the forest, as a VIA's own render draws them;
//   screened the same with every tree in front, its alpha-cut silhouette (impostors too, every stem of the package
//           however far) in the class of the ground under its own stem, and in `vegetation` (screening) only where
//           that ground is unlisted or water; precipitation left out;
//   depth   metres along the view axis of the picture (trees included), 0 for the sky;
//   zones   when the request has `zones`: the same camera, the zone polygons draped on the bare terrain (no trees, no
//           water, no stumps), unlisted terrain `zones.terrain`, the sky transparent. Straight RGBA bytes.
// The class pass writes its colour and the view-space depth into one float target in a single render, so the two
// can never disagree. The live view and its camera are put back afterwards; the canvas is never resized.
import * as THREE from 'three/webgpu'
import {
  mrt,
  positionView,
  positionWorld,
  select,
  sRGBTransferEOTF,
  texture,
  uniform,
  uniformArray,
  vec2,
  vec4
} from 'three/tsl'
import { scatterParamsNode, type Scatter } from 'threejs-scatter'
import type { Atmosphere } from './pgmaps-environment'
import type { Ground } from './pgmaps-ground'
import {
  captureFar,
  clippedShare,
  colourCounts,
  cropColumns,
  encodeStemClass,
  frameCamera,
  guardedAspect,
  km,
  percent,
  pictureGuard,
  pixelShare,
  projectPinhole,
  putYaws,
  rasterizeClassLayers,
  readbackRows,
  splitClassDepth,
  stemClassCode,
  stemClassPalette,
  takeYaws,
  unprojectPixel,
  type CaptureFrame,
  type ClassLayer,
  type ClassRaster,
  type Extent,
  type Rgb
} from './pgmaps-capture-math'
import { inRing } from './pgmaps-ground'
import { TREE_LOD_METRES } from './pgmaps-trees'
import {
  CARD_BAND,
  CARD_BAND_DENSITY,
  DEFAULT_CUTOFF_PX,
  SLAB_MAX,
  type FarForest,
  type FinalStand
} from './pgmaps-far-forest'
import {
  jitterOffsets,
  linearToSrgbByte,
  meanSrgbBytes,
  PATCH_RADIUS,
  PATCH_SPACING,
  SRGB_TO_LINEAR
} from './pgmaps-far-forest-math'

declare const __WORLDGEN_VERSION__: string | undefined

export const CAPTURE_ENGINE = 'threejs-worldgen'
export const CAPTURE_VERSION = typeof __WORLDGEN_VERSION__ === 'string' ? __WORLDGEN_VERSION__ : 'unknown'

export interface CaptureRequest {
  type: 'pgmaps-capture'
  /** Frozen adapter options for this capture; live scale and frame pacing never size or throttle it. */
  renderOptions?: Partial<RenderOptions>
  id: string
  position: [number, number, number]
  frames: CaptureFrame[]
  classes: { terrain: Rgb; layers: ClassLayer[]; vegetation: Rgb; water: Rgb }
  /** A second flat pass of the same cameras: these polygons on the bare terrain, the rest `terrain`. */
  zones?: { terrain: Rgb; layers: ClassLayer[] }
  /**
   * `final`: the report's still. Waits for every stream, LOD and shader, fits the sun's shadows to each frame, draws
   * `stand` in place of the package's stems where it reaches, the water's reflection at full resolution, and
   * accumulates `samples` jittered renders per frame. `active` (or absent): as the live view draws.
   */
  quality?: 'active' | 'final'
  /** Jittered samples accumulated per frame in `final` (default 24, at most 64). */
  samples?: number
  /** The frame's forest at real density (`final`): the package's tree fields, radius and bearings from the eye. */
  stand?: FinalStand
  /** Pixels a median tree is tall where the forest becomes the canopy surface (VNS's dissolve; default 4). */
  canopyCutoffPx?: number
}

export interface CaptureFrameReply {
  width: number
  height: number
  image: Blob
  /** The ground class pass: class layers on the terrain, water; no trees. */
  classes: Uint8ClampedArray
  /** The screened pass: `classes` with every tree drawn in front, in the class of the ground under its stem. */
  screened: Uint8ClampedArray
  /** Metres along the view axis of the picture (trees included), 0 for the sky. */
  depth: Float32Array
  zones?: Uint8ClampedArray
  camera: {
    position: [number, number, number]
    bearingDegrees: number
    pitchDegrees: number
    horizontalFovDegrees: number
    verticalFovDegrees: number
    near: number
    far: number
  }
}

export interface CaptureReply {
  type: 'pgmaps-engine-capture'
  id: string
  engine: string
  version: string
  frames: CaptureFrameReply[]
  scene: { createdAt: string | null; origin: { lng: number; lat: number; elevationMeters?: number } | null }
  environment: Record<string, unknown> | null
  maxTreeHeightMeters: number | null
  renderOptions: RenderOptions
  dissolve: {
    requestedPixels: number
    mode: RenderOptions['dissolveMode']
    referenceHeight: number
    cutoffMeters: number
    patchRadiusMeters: number
    eyeOffsetMeters: number
    guardMeters: number
  }
  /** What was drawn without data, as short phrases the sheet prints (at most about 60 characters each). */
  missing: string[]
  /** The same and every other limit of the capture, at length with figures. */
  missingDetail: string[]
  /** What a `final` request drew (absent for `active`): samples, stems drawn, the canopy cut-off, milliseconds. */
  final?: { samples: number; stems: number; canopyCutoffMeters: number | null; millis: number } | null
}

/** The package fields the capture reads. */
export interface CapturePackage {
  createdAt?: string
  origin?: { lng: number; lat: number; elevationMeters?: number }
  camera: { position: [number, number, number]; bearingDegrees: number; pitchDegrees: number }
  trees: { fields: string[]; count: number; data: number[] }
  treeRadiusMeters: number
  treated?: { count: number }
}

/** What the page lends the capture. */
export interface CaptureHost {
  renderer: THREE.WebGPURenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  pipeline: THREE.RenderPipeline
  atmosphere: Atmosphere
  /** Waits out any build in progress; the scene to capture, or null when there is none. */
  settled(): Promise<{
    pkg: CapturePackage
    scatter: Scatter
    /** Everything past the scatter's stems: stand patches, single cards and the canopy surface. */
    farForest: FarForest
    ground: Ground
    /** How far the scatter draws stems from the camera, metres. */
    maxDistance: number
    /** Stems the scatter draws (the package's within its near stand, on dry ground). */
    stems: number
  } | null>
  /** Holds (true) or releases the live frame loop: the capture borrows the camera and the renderer. */
  hold(on: boolean): void
  /**
   * What the frame loop does before drawing (sky, light, shadows, stand, water), at the camera as it stands; `fit`
   * centres and sizes the sun's shadow box on a frame's near frustum.
   */
  renderOptions(): RenderOptions
  captureOptions(options?: Partial<RenderOptions>): Promise<() => void>
  prepare(now: number, fit?: { centre: THREE.Vector3; radius: number }): void
}

/** Renders the picture this many times at the capture camera before keeping it (shaders, streaming, the sky bake). */
const SETTLE_RENDERS = 3
/** In `final`, warm-up renders before the samples are kept (MRQ's warm-up frames). */
const FINAL_SETTLE_RENDERS = 4
/** Samples a `final` frame accumulates by default, and at most. */
const FINAL_SAMPLES = 24
const MAX_SAMPLES = 64
/** In `final`, the sun's shadow box holds the frame's frustum from the eye out to this many metres. */
const FINAL_SHADOW_REACH = 160
/**
 * The capture camera's near plane, metres. Not the live view's 0.3 m: the far plane now reaches the skyline's apron
 * (30 km), and a 24-bit depth buffer resolves about z² / (near · 2²⁴) — 5 m at 5 km with 0.3 m, where a lake's surface
 * stands under a metre over its bed near the shore. At 1 m it is 1.5 m; the eye stands 1.5 m above the ground.
 */
const NEAR = 1
const MAX_SIZE = 8192
/** Classes a tree can take: `vegetation` and up to this many less one listed colours. */
const STEM_CLASSES = 32
/** A note only goes on the sheet when this share of a capture's drawn pixels shows it. */
const NOTE_SHARE = 0.005

/**
 * The next animation frame, or 50 ms, whichever comes first: a hidden or throttled frame (an embedded view scrolled
 * away, a background tab) never runs `requestAnimationFrame`, and a capture must never wait on it.
 */
const nextFrame = () =>
  new Promise<void>((resolve) => {
    const done = () => {
      clearTimeout(timer)
      resolve()
    }
    const timer = setTimeout(done, 50)
    requestAnimationFrame(done)
  })

/**
 * A capture's time budget, milliseconds from the scene being settled: PGMaps gives up at 180 s. A `final` frame stops
 * accumulating samples once its share of the budget is spent (keeping those it has, and saying how many in `final`).
 */
export const CAPTURE_BUDGET_MS = 120_000

/** Render steps done of the capture's total, with what is being drawn ("frame 2 of 3"). */
export type CaptureProgress = (done: number, total: number, stage: string) => void
/** Progress goes to the parent at most this often (and once at the end). */
const PROGRESS_EVERY_MS = 250

const isRenderable = (o: THREE.Object3D) =>
  (o as THREE.Mesh).isMesh || (o as THREE.Points).isPoints || (o as THREE.Line).isLine || (o as THREE.Sprite).isSprite

function check(request: CaptureRequest) {
  const finite = (v: unknown) => typeof v === 'number' && Number.isFinite(v)
  if (!Array.isArray(request.position) || request.position.length !== 3 || !request.position.every(finite))
    throw new Error('The capture position is not [x, y, z] in metres.')
  if (!Array.isArray(request.frames) || !request.frames.length) throw new Error('The capture asks for no frames.')
  request.frames.forEach((f, i) => {
    if (!Number.isInteger(f.width) || !Number.isInteger(f.height) || f.width < 1 || f.height < 1)
      throw new Error(`Frame ${i + 1}'s size is not whole pixels.`)
    if (f.width > MAX_SIZE || f.height > MAX_SIZE) throw new Error(`Frame ${i + 1} is larger than ${MAX_SIZE} px.`)
    for (const k of ['horizontalFovDegrees', 'verticalFovDegrees'] as const)
      if (!finite(f[k]) || f[k] <= 0 || f[k] >= 179) throw new Error(`Frame ${i + 1}'s ${k} is out of range.`)
    if (!finite(f.bearingDegrees) || !finite(f.pitchDegrees) || Math.abs(f.pitchDegrees) >= 89.9)
      throw new Error(`Frame ${i + 1}'s bearing or pitch is out of range.`)
  })
  const c = request.classes
  if (!c || !Array.isArray(c.layers)) throw new Error('The capture has no class layers.')
  if (request.zones && !Array.isArray(request.zones.layers)) throw new Error('The capture’s zones have no layers.')
}

/**
 * The sphere round a camera's frustum from its near plane out to `reach` metres (world space): what the sun's shadow
 * box must hold for the frame's own trees and ground to shadow each other.
 */
function frustumSphere(camera: THREE.PerspectiveCamera, reach: number): { centre: THREE.Vector3; radius: number } {
  const t = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)
  const points: THREE.Vector3[] = []
  for (const d of [camera.near, reach])
    for (const sx of [-1, 1])
      for (const sy of [-1, 1])
        points.push(new THREE.Vector3(sx * d * t * camera.aspect, sy * d * t, -d).applyMatrix4(camera.matrixWorld))
  const centre = points.reduce((c, p) => c.add(p), new THREE.Vector3()).multiplyScalar(1 / points.length)
  const radius = Math.max(...points.map((p) => p.distanceTo(centre)))
  return { centre, radius: Math.ceil(radius) }
}

/** Straight RGBA bytes → PNG. */
async function png(rgba: Uint8ClampedArray, width: number, height: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(width, height)
  const context = canvas.getContext('2d')!
  context.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, width, height), 0, 0)
  return canvas.convertToBlob({ type: 'image/png' })
}

/** A layer id short enough for a sheet note. */
const shortId = (id: string) => (id.length > 22 ? `${id.slice(0, 21)}…` : id)

export function createCapturer(host: CaptureHost) {
  const { renderer, scene, camera, pipeline } = host
  const webgl = (renderer.backend as { isWebGLBackend?: boolean }).isWebGLBackend === true

  // ---- the flat passes: every drawn surface keeps its own material (so its displacement and alpha cut, the very
  // silhouette the picture has), and a multiple-render-target output replaces what it writes with one flat colour
  // (rgb) and its view-axis depth (alpha). Each object's colour is a per-object uniform: `classOf` is filled for the
  // render. Role 1 is the package's terrain grid, coloured from the near drape; role 2 the skyline terrain, from the
  // far drape; role 4 the real skyline's terrain (`skylineTerrain`), from its own coarser drape; role 0 takes its flat
  // colour.
  const classOf = new Map<THREE.Object3D, { role: 0 | 1 | 2 | 3 | 4 | 5; rgb: THREE.Vector3 }>()
  const none = { role: 0 as const, rgb: new THREE.Vector3() }
  const classRole = uniform(0).onObjectUpdate(({ object }) => (object ? (classOf.get(object) ?? none).role : 0))
  const classColour = uniform(new THREE.Vector3()).onObjectUpdate(({ object }) =>
    object ? (classOf.get(object) ?? none).rgb : none.rgb
  )
  const vegetation = new THREE.Vector3()
  const water = new THREE.Vector3()
  const terrain = new THREE.Vector3()
  const zoneTerrain = new THREE.Vector3()

  /** A raster draped over the terrain (package x, y = world x, −z), nearest; `inside` where it covers. */
  const placeholder = new THREE.DataTexture(new Uint8Array(4), 1, 1)
  placeholder.needsUpdate = true
  const drape = () => {
    const map = texture(placeholder)
    const origin = uniform(new THREE.Vector2())
    const span = uniform(new THREE.Vector2(1, 1))
    const uv = vec2(positionWorld.x, positionWorld.z.negate()).sub(origin).div(span)
    const texel = map.sample(uv)
    const inside = uv.x
      .greaterThanEqual(0)
      .and(uv.x.lessThan(1))
      .and(uv.y.greaterThanEqual(0))
      .and(uv.y.lessThan(1))
      .and(texel.a.greaterThan(0.5))
    return {
      texel,
      inside,
      /** Shows a raster, or nothing. */
      set(grid: ClassRaster | null) {
        if (!grid) {
          map.value = placeholder
          origin.value.set(0, 0)
          span.value.set(1e-6, 1e-6)
          return
        }
        map.value = textureOf(grid)
        origin.value.set(grid.west, grid.south)
        span.value.set(grid.width * grid.cell, grid.height * grid.cell)
      }
    }
  }
  /** One texture per raster for the capture's frames; freed when the capture ends. */
  const textures = new Map<ClassRaster, THREE.DataTexture>()
  const textureOf = (grid: ClassRaster) => {
    let t = textures.get(grid)
    if (!t) {
      t = new THREE.DataTexture(grid.data, grid.width, grid.height, THREE.RGBAFormat, THREE.UnsignedByteType)
      t.magFilter = t.minFilter = THREE.NearestFilter
      t.generateMipmaps = false
      t.needsUpdate = true
      textures.set(grid, t)
    }
    return t
  }
  const freeTextures = () => {
    for (const t of textures.values()) t.dispose()
    textures.clear()
  }
  const nearDrape = drape()
  const farDrape = drape()
  const skylineDrape = drape()
  // Role 3 is a tree: the colour of the class its stem stands in, by the code its yaw carries in whole turns
  // (`encodeStemClass`, set once a capture per stem): 0 is `vegetation`, k the k-th listed colour.
  const stemPalette = uniformArray(
    Array.from({ length: STEM_CLASSES }, () => new THREE.Vector3()),
    'vec3'
  )
  const stemCode = scatterParamsNode()
    .x.div(Math.PI * 2)
    .floor()
    .clamp(0, STEM_CLASSES - 1)
    .toInt()
  // Role 5 is the canopy surface: trees standing on the ground under it, so the class of that ground where a layer
  // lists it, else `vegetation` (the stem palette's first entry), as a stem standing there would take.
  const canopyClass = select(
    nearDrape.inside,
    nearDrape.texel.rgb,
    select(
      farDrape.inside,
      farDrape.texel.rgb,
      select(skylineDrape.inside, skylineDrape.texel.rgb, stemPalette.element(0) as unknown as THREE.Node<'vec3'>)
    )
  )
  const flatOutput = mrt({
    classes: vec4(
      select(
        classRole.equal(3),
        stemPalette.element(stemCode) as unknown as THREE.Node<'vec3'>,
        select(
          classRole.equal(5),
          canopyClass,
          select(
            classRole.equal(1).and(nearDrape.inside),
            nearDrape.texel.rgb,
            select(
              classRole.equal(2).and(farDrape.inside),
              farDrape.texel.rgb,
              select(classRole.equal(4).and(skylineDrape.inside), skylineDrape.texel.rgb, classColour)
            )
          )
        )
      ),
      positionView.z.negate()
    )
  })

  /**
   * Water draws through a stand-in with its own surface displacement only: its material reads the picture behind it
   * (a copy of the framebuffer), which a float class target cannot give it.
   */
  const waterStandIns = new Map<THREE.Material, THREE.NodeMaterial>()
  const waterStandIn = (source: THREE.Material) => {
    let m = waterStandIns.get(source)
    if (!m) {
      m = new THREE.MeshBasicNodeMaterial({ side: source.side })
      m.lights = false
      m.fog = false
      const node = source as THREE.NodeMaterial
      if (node.isNodeMaterial) m.positionNode = node.positionNode
      waterStandIns.set(source, m)
    }
    return m
  }

  const imageTarget = new THREE.RenderTarget(1, 1, { type: THREE.UnsignedByteType, depthBuffer: false, samples: 0 })
  const classTarget = new THREE.RenderTarget(1, 1, { type: THREE.FloatType, depthBuffer: true, samples: 0 })
  const screenedTarget = new THREE.RenderTarget(1, 1, { type: THREE.FloatType, depthBuffer: true, samples: 0 })
  const zoneTarget = new THREE.RenderTarget(1, 1, { type: THREE.FloatType, depthBuffer: true, samples: 0 })
  for (const t of [classTarget, screenedTarget, zoneTarget])
    t.texture.minFilter = t.texture.magFilter = THREE.NearestFilter
  // The flat output is the MRT output `classes`: a target receives it only from a texture of that very name.
  for (const t of [classTarget, screenedTarget, zoneTarget]) t.texture.name = 'classes'

  // ---- a `final` frame's samples, added up on the GPU as three's own SSAAPassNode does: each jittered render goes
  // into `sampleTarget` as the pipeline writes it (tone-mapped, sRGB-encoded, in half floats: not rounded to bytes),
  // and a quad adds it, turned back to linear light, into `sumTarget` (floats) with an additive blend. The sum is read
  // back once a frame. It needs a float target that blends (`float32-blendable`) and the WebGPU queue; without them
  // each sample is read back and added here, as before.
  const gpuQueue = (
    renderer.backend as unknown as { device?: { queue?: { onSubmittedWorkDone?: () => Promise<unknown> } } }
  ).device?.queue
  /**
   * three's node frame. A pass (the pipeline's scene pass, the bloom) redraws once per frame id, and the id only moves
   * on with the animation loop: samples drawn faster than the display's frames would all be the first one's picture.
   * Each sample added on the GPU starts a frame of its own.
   */
  // This reaches into three (`renderer._nodes.nodeFrame`, r186: there is no public call that starts a frame outside
  // the animation loop), so it is checked by what it does, not only that it is there: `newFrame` says whether the
  // frame id moved on by one. If a later three keeps its frames elsewhere, the samples go back to being read back
  // one by one (slower, the same picture), and pgmaps-three-internals.test.ts fails on the upgrade.
  const newFrame = (): boolean => {
    const nodeFrame = (renderer as unknown as { _nodes?: { nodeFrame?: { update?: unknown; frameId?: unknown } } })
      ._nodes?.nodeFrame
    if (!nodeFrame || typeof nodeFrame.update !== 'function' || typeof nodeFrame.frameId !== 'number') return false
    const before = nodeFrame.frameId
    ;(nodeFrame.update as () => void).call(nodeFrame)
    return nodeFrame.frameId === before + 1
  }
  /** Resolves once the GPU has drawn everything sent to it so far. */
  const gpuDone = gpuQueue?.onSubmittedWorkDone ? () => gpuQueue.onSubmittedWorkDone?.() ?? null : null
  /** Whether samples are added up on the GPU: decided at the first capture (the renderer is up by then), once. */
  let sumOnGpuChecked: boolean | null = null
  const sumOnGpu = (): boolean => {
    if (sumOnGpuChecked === null) {
      const able = !webgl && !!gpuDone && renderer.hasFeature('float32-blendable')
      sumOnGpuChecked = able && newFrame()
      if (able && !sumOnGpuChecked)
        console.warn('[capture] three no longer starts a node frame this way: samples are read back one by one')
    }
    return sumOnGpuChecked
  }
  const sampleTarget = new THREE.RenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false, samples: 0 })
  sampleTarget.texture.minFilter = sampleTarget.texture.magFilter = THREE.NearestFilter
  const sumTarget = new THREE.RenderTarget(1, 1, { type: THREE.FloatType, depthBuffer: false, samples: 0 })
  sumTarget.texture.minFilter = sumTarget.texture.magFilter = THREE.NearestFilter
  const addMaterial = new THREE.NodeMaterial()
  // What the 8-bit target kept of a sample, less its rounding: clamped to 0…1, then sRGB → linear.
  addMaterial.fragmentNode = vec4(
    sRGBTransferEOTF(texture(sampleTarget.texture).rgb.saturate()) as unknown as THREE.Node<'vec3'>,
    1
  )
  addMaterial.transparent = true
  addMaterial.depthTest = false
  addMaterial.depthWrite = false
  addMaterial.premultipliedAlpha = true
  addMaterial.blending = THREE.AdditiveBlending
  addMaterial.name = 'capture: add a sample'
  const addQuad = new THREE.QuadMesh(addMaterial)

  /**
   * The own yaws of every list a capture has coded (a scatter layer's instances, a far-forest mesh's `scatterParams`),
   * taken the first time it is coded (`takeYaws`). Each capture codes from these and puts them back when it ends.
   */
  const ownYaws = new WeakMap<object, Float32Array>()

  let queue: Promise<unknown> = Promise.resolve()
  let lastTimingMs = 0
  let lastSummary: Record<string, unknown> | null = null

  /** The near and far drapes of a set of layers over the drawn terrain. */
  const rasters = (layers: readonly ClassLayer[], ground: Ground, nearSize: number) => ({
    near: rasterizeClassLayers(layers, ground.drawnExtent, { maxSize: nearSize }),
    far: ground.farExtent ? rasterizeClassLayers(layers, ground.farExtent, { maxSize: 2048, minCell: 4 }) : null,
    skyline: ground.skylineExtent
      ? rasterizeClassLayers(layers, ground.skylineExtent, { maxSize: 2048, minCell: 8 })
      : null
  })

  async function run(request: CaptureRequest, onProgress?: CaptureProgress): Promise<CaptureReply> {
    check(request)
    const built = await host.settled()
    if (!built) throw new Error('No scene is loaded to capture.')
    const restoreOptions = await host.captureOptions(request.renderOptions)
    const mirror = built.ground.water?.mirror
    const mirrorScale = mirror?.reflector.resolutionScale
    try {
      return await captureBuilt(request, built, onProgress)
    } finally {
      built.farForest.clearFinal()
      built.farForest.pin(null)
      if (mirror && mirrorScale !== undefined) mirror.reflector.resolutionScale = mirrorScale
      freeTextures()
      restoreOptions()
    }
  }

  async function captureBuilt(
    request: CaptureRequest,
    built: NonNullable<Awaited<ReturnType<CaptureHost['settled']>>>,
    onProgress?: CaptureProgress
  ): Promise<CaptureReply> {
    const appliedOptions = host.renderOptions()
    const { pkg, scatter, ground, farForest } = built
    const started = performance.now()
    /** Milliseconds spent in each stage of the capture, over all its frames (`lastSummary.timings`). */
    const spent: Record<string, number> = {}
    let lapAt = started
    const lap = (stage: string) => {
      const now = performance.now()
      spent[stage] = (spent[stage] ?? 0) + now - lapAt
      lapAt = now
    }
    const final = request.quality === 'final'
    const samples = final
      ? Math.min(
          MAX_SAMPLES,
          Math.max(1, Math.round(Number.isFinite(request.samples) ? request.samples! : FINAL_SAMPLES))
        )
      : 1
    // The final stand, before the stems are coded: its cards are coded with them.
    const nearRadius = built.maxDistance - 50
    let finalStems = 0
    let standHeight = farForest.stats.medianHeight
    /** Why the final stand could not be drawn (the capture goes on with the stand patches), or null. */
    let standFailed: string | null = null
    if (final && request.stand && request.stand.count > 0) {
      try {
        const placed = farForest.showFinal(request.stand, request.position, nearRadius, ground.groundAt)
        finalStems = placed.stems
        standHeight = placed.medianHeight
      } catch (error) {
        farForest.clearFinal()
        standFailed = error instanceof Error ? error.message : String(error)
      }
    }
    lap('stand')
    /** Samples each frame actually accumulated (fewer than asked when the budget ran out). */
    let samplesDrawn = samples
    // Progress for PGMaps' time-left estimate (`pgmaps-capture-progress`): one step per warm-up render and sample.
    const settleRenders = final ? FINAL_SETTLE_RENDERS : SETTLE_RENDERS
    const totalSteps = request.frames.length * (settleRenders + (samples > 1 ? samples : 0))
    let stepsDone = 0
    let sentAt = 0
    const step = (stage: string, count = 1) => {
      stepsDone = Math.min(totalSteps, stepsDone + count)
      const now = performance.now()
      if (!onProgress || (now - sentAt < PROGRESS_EVERY_MS && stepsDone < totalSteps)) return
      sentAt = now
      onProgress(stepsDone, totalSteps, stage)
    }
    const deadline = started + CAPTURE_BUDGET_MS
    // The dissolve distance for these frames: where the stand's median tree is under `canopyCutoffPx` pixels tall.
    const cutoffPx =
      Number.isFinite(request.canopyCutoffPx) && request.canopyCutoffPx! > 0
        ? request.canopyCutoffPx!
        : appliedOptions.dissolvePixels
    const cutoff = Math.max(
      ...request.frames.map((f) =>
        farForest.cutoffFor(f.height, f.verticalFovDegrees, cutoffPx, standHeight, request.position)
      )
    )
    farForest.pin(cutoff)
    const mirror = ground.water?.mirror ?? null
    const mirrorScale = mirror?.reflector.resolutionScale ?? 0.5
    if (final && mirror) mirror.reflector.resolutionScale = 1
    /** Short phrases for the sheet, and every limit at length. */
    const missing: string[] = []
    const detail: string[] = []

    // Class colours and the drapes of the class layers over the terrain grid and the skyline terrain.
    const toUnit = (rgb: Rgb) => new THREE.Vector3(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    vegetation.copy(toUnit(request.classes.vegetation))
    water.copy(toUnit(request.classes.water))
    terrain.copy(toUnit(request.classes.terrain))
    const drawn = ground.drawnExtent
    const real: Extent = ground.skylineExtent ?? ground.farExtent ?? drawn
    const layers = request.classes.layers
    const classRasters = rasters(layers, ground, 4096)
    const zoneLayers = request.zones?.layers ?? null
    const zoneRasters = zoneLayers ? rasters(zoneLayers, ground, 2048) : null
    if (request.zones) zoneTerrain.copy(toUnit(request.zones.terrain))
    const extentText = (e: Extent) =>
      `x ${Math.round(e.west)}…${Math.round(e.east)} m, y ${Math.round(e.south)}…${Math.round(e.north)} m`
    for (const layer of layers) {
      const share = clippedShare(layer.polygons, real)
      if (share > 0.001) {
        missing.push(`${percent(share)} of ${shortId(layer.id)} past terrain, not drawn`)
        detail.push(
          `Class layer "${layer.id}": ${(share * 100).toFixed(1)}% of its area lies past the drawn terrain (${extentText(real)}) and was cut from the class pass.`
        )
      }
    }
    const describe = (what: string, r: { near: ClassRaster; far: ClassRaster | null; skyline: ClassRaster | null }) =>
      `${what} are draped as a ${r.near.width} × ${r.near.height} texture of ${r.near.cell.toFixed(2)} m cells over the package's terrain grid` +
      (r.far
        ? ` and a ${r.far.width} × ${r.far.height} one of ${r.far.cell.toFixed(1)} m cells over the skyline terrain`
        : '') +
      (r.skyline
        ? ` and a ${r.skyline.width} × ${r.skyline.height} one of ${r.skyline.cell.toFixed(1)} m cells over the real skyline's terrain`
        : '') +
      ' (cell centre in the polygon; nearest).'
    if (layers.length) detail.push(describe('Class layers', classRasters))

    // Each stem's class, once, from the ground under it: carried by its yaw (whole turns) into the class pass.
    const palette = stemClassPalette(layers).slice(0, STEM_CLASSES - 1)
    const paletteNames = [
      'vegetation',
      ...palette.map((c) => layers.find((l) => l.rgb.every((v, k) => v === c[k]))?.id ?? '?')
    ]
    stemPalette.array.forEach((v, k) => {
      ;(v as THREE.Vector3).copy(k > 0 && k <= palette.length ? toUnit(palette[k - 1]) : vegetation)
    })
    const stemCounts = new Array(palette.length + 1).fill(0)
    const codeAt = (x: number, y: number) => stemClassCode(classRasters, drawn, palette, x, y)
    /** Every list coded here with its own yaws, to put back when the capture ends (`uncode`). */
    const coded: Array<{
      values: Float32Array
      stride: number
      offset: number
      yaws: Float32Array
      attribute?: THREE.BufferAttribute
    }> = []
    const yawsOf = (values: Float32Array, stride: number, offset: number, attribute?: THREE.BufferAttribute) => {
      let yaws = ownYaws.get(values)
      if (!yaws) ownYaws.set(values, (yaws = takeYaws(values, stride, offset, Math.floor(values.length / stride))))
      coded.push({ values, stride, offset, yaws, attribute })
      return yaws
    }
    for (const layer of scatter.settings.layers) {
      const list = layer.instances
      if (!list) continue
      const yaws = yawsOf(list, 5, 2)
      for (let k = 0, n = 0; k + 5 <= list.length; k += 5, n++) {
        const code = codeAt(list[k], -list[k + 1])
        list[k + 2] = encodeStemClass(yaws[n], code)
        stemCounts[code]++
      }
    }
    scatter.regenerate() // the tiles take the new yaws; the capture streams them all in below
    // The far forest's slabs and cards: each by the ground under its centre or stem (instanced: `instanceCount`).
    for (const o of farForest.coded()) {
      const g = (o as THREE.Mesh<THREE.InstancedBufferGeometry>).geometry
      const at = g?.getAttribute('scatterPosition') as THREE.InstancedBufferAttribute | undefined
      const params = g?.getAttribute('scatterParams') as THREE.InstancedBufferAttribute | undefined
      if (!at || !params) continue
      const count = Math.min(params.count, g.instanceCount ?? params.count)
      const yaws = yawsOf(params.array as Float32Array, 4, 0, params)
      for (let n = 0; n < count; n++) {
        const code = codeAt(at.array[n * 4], -at.array[n * 4 + 2])
        params.array[n * 4] = encodeStemClass(yaws[n], code)
        stemCounts[code]++
      }
      params.needsUpdate = true
    }
    /** Puts every coded yaw back as it was built: the live view and the next capture start from the scene as built. */
    const uncode = () => {
      for (const { values, stride, offset, yaws, attribute } of coded) {
        putYaws(values, stride, offset, yaws)
        if (attribute) attribute.needsUpdate = true
      }
      coded.length = 0
      scatter.regenerate()
    }
    const stemClasses = Object.fromEntries(paletteNames.map((name, k) => [name, stemCounts[k]]))
    detail.push(
      'Each drawn tree takes the class of the ground under its own stem; only trees on unlisted ground or water count ' +
        `as screening. Stems by class: ${paletteNames.map((name, k) => `${name} ${stemCounts[k]}`).join(', ')}.`
    )
    if (zoneRasters && zoneLayers?.length) detail.push(describe('Zone layers', zoneRasters))

    // The terrain: what had no data.
    const filled = ground.filled
    if (filled.invented >= NOTE_SHARE) missing.push(`${percent(filled.invented)} of near terrain posts gap-filled`)
    if (filled.invented > 0)
      detail.push(
        `${(filled.invented * 100).toFixed(1)}% of the package's terrain grid had no height and took its neighbours' along rows and columns.`
      )
    if (filled.fromFar >= 0.01) missing.push(`${percent(filled.fromFar)} of near terrain from coarser grid`)
    if (filled.fromFar > 0)
      detail.push(
        `${(filled.fromFar * 100).toFixed(1)}% of the package's terrain grid had no height and took the skyline terrain's (coarser) height there.`
      )
    if (filled.farInvented >= NOTE_SHARE)
      missing.push(`${percent(filled.farInvented)} of skyline terrain posts gap-filled`)
    if (filled.farInvented > 0)
      detail.push(
        `${(filled.farInvented * 100).toFixed(1)}% of the skyline terrain's posts had no height and took their neighbours'.`
      )
    if (filled.skylineInvented >= NOTE_SHARE)
      missing.push(`${percent(filled.skylineInvented)} of far skyline terrain posts gap-filled`)
    if (filled.skylineInvented > 0)
      detail.push(
        `${(filled.skylineInvented * 100).toFixed(1)}% of the real skyline's terrain posts had no height and took their neighbours'.`
      )
    if (ground.skylineExtent)
      detail.push(
        `Terrain past the skyline grid's inner edge is the package's \`skylineTerrain\` (${extentText(ground.skylineExtent)}), drawn with the earth's curve already taken off its heights, as given.`
      )

    // The stand: what is drawn from this eye.
    const fields = pkg.trees.fields
    const stride = fields.length
    const [ix, iy, ih] = ['x', 'y', 'height'].map((f) => fields.indexOf(f))
    let tallest = -Infinity
    let wet = 0
    let bare = 0
    let burnt = 0
    let reach = 0
    for (let i = 0; i < pkg.trees.count; i++) {
      const o = i * stride
      const x = pkg.trees.data[o + ix]
      const y = pkg.trees.data[o + iy]
      if (ground.inWater(x, y)) {
        wet++
        continue
      }
      if (ground.onBare(x, y)) {
        bare++
        continue
      }
      if (Number.isFinite(ground.burntAge(x, y))) burnt++
      reach = Math.max(reach, Math.hypot(x - request.position[0], y - request.position[1]))
      if (ih >= 0) tallest = Math.max(tallest, pkg.trees.data[o + ih])
    }
    const drawnStems = built.stems
    const forest = farForest.stats
    // Past the scatter's stems the far forest stands to the terrain's edge (slabs, then the canopy surface).
    const stemReach = farForest.canopy ? Infinity : Math.max(reach, pkg.treeRadiusMeters, SLAB_MAX)
    detail.push(
      `${drawnStems} stems of the package within ${Math.round(nearRadius)} m of its origin are drawn as trees: ` +
        `meshes to ${TREE_LOD_METRES.coarse} m, the scatter's impostor cards (coverage-preserving mipmaps) to ${Math.round(built.maxDistance)} m. ` +
        `Past them the forest is drawn from the package's crown closure, species mix and stem heights: single trees at real density ` +
        `(about ${CARD_BAND_DENSITY} stems/ha × closure, real crowns) to ${CARD_BAND / 1000} km, then ${forest.patches} stand patches ` +
        `(a ${2 * PATCH_RADIUS} m disc of real-density trees baked from 7 angles to its ground, one per ${PATCH_SPACING} m cell of forest) out to ` +
        `${Math.round(cutoff)} m, using ${appliedOptions.dissolveMode} dissolve at requested ${cutoffPx} px ` +
        `(reference height ${appliedOptions.dissolveReferenceHeight} px; median tree ${standHeight.toFixed(1)} m), bounded by sampled patch reach ` +
        `minus eye offset and the patch edge guard; the canopy ` +
        `surface past it (the ground raised by the stand's height, coloured by its species and closure). ` +
        `The package's ${forest.packageStems} far stems stand for that forest (those on open ground are drawn as single trees); ${forest.cards} single trees in all. ` +
        (forest.coverCodes
          ? `The package's ground cover codes thin the far forest: no trees on rock or water, fewer on sparse forest, shrub and herb ground ` +
            `(${forest.thinned} single trees left out), and the canopy surface is not raised over rock, herb or shrub; the ground there is drawn as that cover. `
          : 'The package carries no ground cover codes. ') +
        (pkg.trees.fields.includes('crownBase') || (final && request.stand?.fields.includes('crownBase'))
          ? 'Additive stands retain supplied crown widths. Near meshes use crown-base classes of 0.2 and six trunk slenderness classes where DBH is supplied; their trunk diameter is approximate after crown scaling. Far cards remap their crown base/form and retain their silhouette in the class pass. Expansion and source IDs are metadata; every supplied visual record is drawn at most once. '
          : 'Legacy trees are at most 1.5 times wider than their grown shape. Impostors, patches and cards keep their silhouettes in the class pass. ') +
        "their depth is the card's plane; the canopy surface takes the class of the ground under it."
    )
    if (forest.paletteRecords)
      detail.push(
        `${forest.paletteRecords} local stand palette records supply far-forest heights and species mixtures; canopy and cover grids still determine closure.`
      )
    if (pkg.treated?.count)
      detail.push(
        `${pkg.treated.count} prescribed treated records are drawn with instanced stumps, logs, snags and felled stems; felled crowns are approximate retained crowns. Other clearcuts retain their seeded stumps.`
      )
    if (final)
      detail.push(
        request.stand
          ? `Final render: the frame's real-density stand (${request.stand.count} stems to ${Math.round(request.stand.radiusMeters)} m, ` +
              `bearings ${request.stand.bearings.map((b) => Math.round(b)).join('–')}°) replaces the stand patches there; ${finalStems} of its stems past ` +
              `${Math.round(nearRadius)} m are drawn as cards (those nearer are the package's own, as meshes), dissolving into the canopy past ${Math.round(cutoff)} m` +
              (forest.coverCodes
                ? `; ${forest.finalThinned} stood on rock, water or open ground and were left out by its cover codes.`
                : '.')
          : 'Final render with no stand sent: the stand patches are drawn.'
      )
    if (wet) detail.push(`${wet} stems standing in water are not drawn.`)
    if (bare)
      detail.push(
        `${bare} stems standing on rock or water by the package's cover codes are not drawn (the picture shows the ground there).`
      )
    if (burnt)
      detail.push(
        `${burnt} stems stand on burnt ground (cover 7) and are drawn as snags: bare trunks and branch stubs, no foliage, ` +
          `over herb and shrub regrowth by the years since the fire; in the screened pass each screens only its own thin ` +
          `silhouette, and past the drawn snags the burnt ground is not raised into a canopy.`
      )
    detail.push('Stumps and slash are drawn in the picture but left out of the class pass (they do not screen).')
    const weather = host.atmosphere.applied.state.weather
    if (weather === 'rain' || weather === 'snow')
      detail.push(`Falling ${weather} is drawn in the picture and left out of the class pass and depth.`)

    const far = captureFar(request.position, real, ground.apronRadius, 8000)
    const live = {
      position: camera.position.clone(),
      quaternion: camera.quaternion.clone(),
      fov: camera.fov,
      aspect: camera.aspect,
      near: camera.near,
      far: camera.far,
      zoom: camera.zoom
    }
    const clear = { color: renderer.getClearColor(new THREE.Color()), alpha: renderer.getClearAlpha() }
    const autoClear = renderer.autoClear
    const target = renderer.getRenderTarget()
    lap('classes')
    host.hold(true)
    const frames: CaptureFrameReply[] = []
    /** Drawn-pixel shares, over all frames: ground past the stems' reach, the apron. */
    let pastStems = 0
    let invented = 0
    const counts: Array<{ ground: Record<string, number>; screened: Record<string, number> }> = []

    /**
     * One flat render of the scene into `into`: `paint` fills `classOf`, everything not in it is hidden; the sky is
     * transparent. Puts every material, visibility and setting back.
     */
    const flatPass = (into: THREE.RenderTarget, paint: (swap: (root: THREE.Object3D) => void) => void) => {
      const swapped: Array<[THREE.Mesh, THREE.Material | THREE.Material[]]> = []
      const hidden: THREE.Object3D[] = []
      classOf.clear()
      paint((root) =>
        root.traverse((o) => {
          const mesh = o as THREE.Mesh
          if (!mesh.isMesh) return
          swapped.push([mesh, mesh.material])
          mesh.material = Array.isArray(mesh.material) ? mesh.material.map(waterStandIn) : waterStandIn(mesh.material)
        })
      )
      scene.traverse((o) => {
        if (isRenderable(o) && o.visible && !classOf.has(o)) hidden.push(o)
      })
      for (const o of hidden) o.visible = false
      const background = scene.background
      const mrtBefore = renderer.getMRT()
      scene.background = null
      renderer.setClearColor(0x000000, 0)
      renderer.autoClear = true
      try {
        renderer.setRenderTarget(into)
        renderer.setMRT(flatOutput)
        renderer.render(scene, camera)
      } finally {
        renderer.setMRT(mrtBefore)
        scene.background = background
        for (const [mesh, material] of swapped) mesh.material = material
        for (const o of hidden) o.visible = true
        classOf.clear()
        renderer.setClearColor(clear.color, clear.alpha)
        renderer.autoClear = autoClear
        renderer.setRenderTarget(target)
      }
    }
    /** The terrain meshes in `classOf`: the grid from the near drape, the skyline from the far, the apron flat. */
    const paintTerrain = (rgb: THREE.Vector3) => {
      for (const mesh of ground.parts.terrain) classOf.set(mesh, { role: 1, rgb })
      if (ground.parts.far) classOf.set(ground.parts.far, { role: 2, rgb })
      if (ground.parts.skyline) classOf.set(ground.parts.skyline, { role: 4, rgb })
      classOf.set(ground.parts.apron, { role: 0, rgb })
    }

    try {
      for (const frame of request.frames) {
        const pose = frameCamera(request.position, frame)
        const place = () => {
          camera.position.set(...pose.position)
          camera.quaternion.set(...pose.quaternion)
          camera.fov = pose.fov
          camera.aspect = pose.aspect
          camera.near = NEAR
          camera.far = far
          camera.zoom = 1
          camera.clearViewOffset()
          camera.updateProjectionMatrix()
          camera.updateMatrixWorld()
        }
        const { width, height } = frame
        /** The averaged samples of a `final` frame (rows from the top), or null for the single render. */
        let accumulated: Uint8Array | null = null
        /** Whether this frame's samples are added up on the GPU (`sumTarget`), and how many were. */
        const summed = samples > 1 && sumOnGpu()
        let summedSamples = 0
        /** What the picture is rendered into: the sample target when the GPU adds the samples up. */
        const drawTarget = summed ? sampleTarget : imageTarget
        // The picture is drawn wider by a guard band each side and cropped: screen-space effects (the water's
        // reflection above all) fade out at a frame's border, which left a pale strip of sky in the lake where two
        // frames meet. The same pinhole, so the cropped pixels are exactly the frame's.
        const guard = pictureGuard(width)
        const wide = width + 2 * guard
        const placeWide = () => {
          place()
          camera.aspect = guardedAspect(pose.aspect, width, guard)
          camera.updateProjectionMatrix()
        }
        // The post passes size themselves from the drawing buffer: for the capture it is the frame's size, so the
        // canvas itself is never resized (nor cleared on screen).
        const patched = renderer as unknown as { getDrawingBufferSize: (t: THREE.Vector2) => THREE.Vector2 }
        let bufferWidth = wide
        patched.getDrawingBufferSize = (t) => t.set(bufferWidth, height)
        try {
          drawTarget.setSize(wide, height)
          if (summed) sumTarget.setSize(wide, height)
          classTarget.setSize(width, height)
          screenedTarget.setSize(width, height)
          if (request.zones) zoneTarget.setSize(width, height)
          placeWide()
          scatter.update(Infinity) // every tile in reach streamed in and sorted into levels for this camera
          // `final`: the sun's shadow box on this frame's near frustum, every pipeline compiled before the warm-up.
          const fit = final ? frustumSphere(camera, FINAL_SHADOW_REACH) : undefined
          if (final) await renderer.compileAsync(scene, camera)
          lap('compile')
          const settle = settleRenders
          const stage = `frame ${frames.length + 1} of ${request.frames.length}`
          for (let k = 0; k < settle; k++) {
            placeWide()
            host.prepare(performance.now(), fit)
            renderer.setRenderTarget(drawTarget)
            pipeline.render()
            step(stage)
            if (k < settle - 1) await nextFrame()
          }
          lap('settle')
          // `final`: jittered sub-pixel samples averaged in linear light (a still camera's TAA, by hand).
          if (samples > 1) {
            const sum = summed ? null : new Float32Array(wide * height * 3)
            // This frame's share of what is left of the budget.
            const framesLeft = request.frames.length - frames.length
            const frameDeadline = performance.now() + Math.max(0, deadline - performance.now()) / framesLeft
            let taken = 0
            /** The GPU finishing the sample before the one just sent (`summed`). */
            let drawn: Promise<unknown> | null = null
            for (const [jx, jy] of jitterOffsets(samples)) {
              if (taken > 0 && performance.now() > frameDeadline) break
              taken++
              placeWide()
              const m = camera.projectionMatrix.elements
              m[8] += (2 * jx) / wide
              m[9] += (2 * jy) / height
              camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert()
              host.prepare(performance.now(), fit)
              if (summed) newFrame()
              renderer.setRenderTarget(drawTarget)
              pipeline.render()
              if (sum) {
                const raw = await renderer.readRenderTargetPixelsAsync(imageTarget, 0, 0, wide, height)
                const rows = readbackRows(new Uint8Array((raw as Uint8Array).buffer), wide, height, 4, webgl)
                for (let i = 0, j = 0; j < sum.length; i += 4, j += 3) {
                  sum[j] += SRGB_TO_LINEAR[rows[i]]
                  sum[j + 1] += SRGB_TO_LINEAR[rows[i + 1]]
                  sum[j + 2] += SRGB_TO_LINEAR[rows[i + 2]]
                }
              } else {
                // Added into the sum, the first sample over a clear to zero.
                renderer.setRenderTarget(sumTarget)
                if (taken === 1) {
                  renderer.setClearColor(0x000000, 0)
                  renderer.clear(true, false, false)
                  renderer.setClearColor(clear.color, clear.alpha)
                }
                renderer.autoClear = false
                addQuad.render(renderer)
                renderer.autoClear = autoClear
                // The next sample is drawn while the GPU works on this one, and no further ahead: the progress and
                // the budget follow what is drawn, not what is queued.
                const before = drawn
                drawn = gpuDone?.() ?? null
                if (before) await before
              }
              step(stage)
            }
            if (drawn) await drawn
            // Samples the budget left out still count as done: the frame is finished.
            if (taken < samples) step(stage, samples - taken)
            samplesDrawn = Math.min(samplesDrawn, taken)
            summedSamples = taken
            if (sum) {
              accumulated = new Uint8Array(wide * height * 4)
              for (let i = 0, j = 0; j < sum.length; i += 4, j += 3) {
                accumulated[i] = linearToSrgbByte(sum[j] / taken)
                accumulated[i + 1] = linearToSrgbByte(sum[j + 1] / taken)
                accumulated[i + 2] = linearToSrgbByte(sum[j + 2] / taken)
                accumulated[i + 3] = 255
              }
            }
          }
          lap('samples')
          // The flat passes, straight after the picture's last render (the same wind), at the frame's own size.
          bufferWidth = width
          place()
          nearDrape.set(classRasters.near)
          farDrape.set(classRasters.far)
          skylineDrape.set(classRasters.skyline)
          const paintWater = (swap: (root: THREE.Object3D) => void) => {
            const lakes = ground.water?.lakes.object
            if (!lakes) return
            lakes.traverse((o) => classOf.set(o, { role: 0, rgb: water }))
            swap(lakes)
          }
          // The ground class pass: the class layers on the terrain and the water, no trees, stumps or slash, so
          // outlines follow the ground polygons as the VIA's own render draws them over the forest.
          flatPass(classTarget, (swap) => {
            paintWater(swap)
            paintTerrain(terrain)
          })
          // The screened pass: the same with every tree in front, in the class of the ground under its stem.
          flatPass(screenedTarget, (swap) => {
            scatter.object.traverse((o) => classOf.set(o, { role: 3, rgb: vegetation }))
            for (const o of farForest.coded()) classOf.set(o, { role: 3, rgb: vegetation })
            if (farForest.canopy) classOf.set(farForest.canopy, { role: 5, rgb: vegetation })
            paintWater(swap)
            paintTerrain(terrain)
          })
          // The zone pass: the bare terrain only.
          if (zoneRasters) {
            nearDrape.set(zoneRasters.near)
            farDrape.set(zoneRasters.far)
            skylineDrape.set(zoneRasters.skyline)
            flatPass(zoneTarget, () => paintTerrain(zoneTerrain))
          }
        } finally {
          delete (patched as Partial<typeof patched>).getDrawingBufferSize
          nearDrape.set(null)
          farDrape.set(null)
          skylineDrape.set(null)
        }
        const [pictureRaw, classRaw, screenedRaw, zoneRaw] = await Promise.all([
          // The picture: the frame's own columns of the sum (one readback for all its samples), else the last render.
          summed
            ? renderer.readRenderTargetPixelsAsync(sumTarget, guard, 0, width, height)
            : renderer.readRenderTargetPixelsAsync(imageTarget, 0, 0, wide, height),
          renderer.readRenderTargetPixelsAsync(classTarget, 0, 0, width, height),
          renderer.readRenderTargetPixelsAsync(screenedTarget, 0, 0, width, height),
          request.zones ? renderer.readRenderTargetPixelsAsync(zoneTarget, 0, 0, width, height) : null
        ])
        lap('passes')
        const picture = summed
          ? meanSrgbBytes(readbackRows(pictureRaw as Float32Array, width, height, 4, webgl), summedSamples)
          : new Uint8ClampedArray(
              cropColumns(
                accumulated ?? readbackRows(new Uint8Array((pictureRaw as Uint8Array).buffer), wide, height, 4, webgl),
                wide,
                height,
                4,
                guard,
                width
              )
            )
        for (let i = 3; i < picture.length; i += 4) picture[i] = 255
        const groundPass = splitClassDepth(
          readbackRows(classRaw as Float32Array, width, height, 4, webgl),
          width,
          height
        ).classes
        // Depth is the picture's: the screened pass, trees in front.
        const split = splitClassDepth(readbackRows(screenedRaw as Float32Array, width, height, 4, webgl), width, height)
        const zones = zoneRaw
          ? splitClassDepth(readbackRows(zoneRaw as Float32Array, width, height, 4, webgl), width, height).classes
          : undefined

        // What the frame shows without data: ground past the stems' reach (not under water), and the apron.
        const c = split.classes
        const v = request.classes.vegetation
        const w = request.classes.water
        const onGround = (o: number) =>
          !(c[o] === v[0] && c[o + 1] === v[1] && c[o + 2] === v[2]) &&
          !(c[o] === w[0] && c[o + 1] === w[1] && c[o + 2] === w[2])
        const [ex, ey] = request.position
        pastStems += pixelShare(
          split.depth,
          frame,
          request.position,
          (x, y) => Math.hypot(x - ex, y - ey) > stemReach + 50 && !ground.inWater(x, y),
          onGround
        )
        invented += pixelShare(
          split.depth,
          frame,
          request.position,
          (x, y) => x < real.west || x > real.east || y < real.south || y > real.north
        )
        const palette = {
          vegetation: v,
          water: w,
          terrain: request.classes.terrain,
          ...Object.fromEntries(layers.map((l) => [l.id, l.rgb]))
        }
        counts.push({ ground: colourCounts(groundPass, palette), screened: colourCounts(c, palette) })
        lap('figures')
        const image = await png(picture, width, height)
        lap('png')
        frames.push({
          width,
          height,
          image,
          classes: groundPass,
          screened: split.classes,
          depth: split.depth,
          ...(zones ? { zones } : {}),
          camera: {
            position: [...request.position] as [number, number, number],
            bearingDegrees: frame.bearingDegrees,
            pitchDegrees: frame.pitchDegrees,
            horizontalFovDegrees:
              (2 * Math.atan(Math.tan(((pose.fov / 2) * Math.PI) / 180) * pose.aspect) * 180) / Math.PI,
            verticalFovDegrees: pose.fov,
            near: NEAR,
            far
          }
        })
      }
    } finally {
      camera.position.copy(live.position)
      camera.quaternion.copy(live.quaternion)
      Object.assign(camera, { fov: live.fov, aspect: live.aspect, near: live.near, far: live.far, zoom: live.zoom })
      camera.updateProjectionMatrix()
      camera.updateMatrixWorld()
      renderer.setRenderTarget(target)
      renderer.autoClear = autoClear
      renderer.setClearColor(clear.color, clear.alpha)
      uncode()
      farForest.clearFinal()
      farForest.pin(null)
      if (mirror) mirror.reflector.resolutionScale = mirrorScale
      host.hold(false)
      freeTextures()
    }
    pastStems /= request.frames.length
    invented /= request.frames.length
    if (pastStems >= NOTE_SHARE) missing.push(`trees past ${km(stemReach)} not drawn`)
    if (pastStems > 0)
      detail.push(
        `${(pastStems * 100).toFixed(1)}% of the drawn pixels are ground farther than ${Math.round(stemReach)} m, past the package's last stem: ` +
          'drawn in the picture as canopy-coloured ground, and the terrain class in the class pass.'
      )
    if (invented >= NOTE_SHARE) missing.push(`terrain past ${km(ground.apronInner)} invented`)
    if (invented > 0)
      detail.push(
        `${(invented * 100).toFixed(1)}% of the drawn pixels are ground past the package's ${ground.skylineExtent || ground.farExtent ? 'skyline' : ''} terrain (${extentText(real)}): ` +
          `an apron the engine invents out to ${km(ground.apronRadius)}, easing to the terrain edge's mean height and the earth's curve; it takes the terrain class.`
      )
    const atmosphere = host.atmosphere
    lastTimingMs = performance.now() - started
    if (final && samplesDrawn < samples)
      detail.push(
        `Final render: ${samplesDrawn} of the ${samples} samples asked were accumulated per frame before the capture's ${CAPTURE_BUDGET_MS / 1000} s budget ran out.`
      )
    if (standFailed)
      detail.push(
        `Final render: the stand sent could not be drawn (${standFailed}); the stand patches were drawn instead.`
      )
    const dissolve = {
      requestedPixels: cutoffPx,
      mode: appliedOptions.dissolveMode,
      referenceHeight: appliedOptions.dissolveReferenceHeight,
      cutoffMeters: Math.round(cutoff),
      patchRadiusMeters: farForest.stats.patchRadiusMeters,
      eyeOffsetMeters: Math.round(Math.hypot(request.position[0], request.position[1])),
      guardMeters: farForest.stats.dissolveGuardMeters
    }
    const finalRecord = final
      ? {
          samples: samplesDrawn,
          stems: finalStems,
          canopyCutoffMeters: Number.isFinite(cutoff) ? Math.round(cutoff) : null,
          millis: Math.round(lastTimingMs)
        }
      : null
    lastSummary = {
      ms: Math.round(lastTimingMs),
      timings: Object.fromEntries(Object.entries(spent).map(([stage, ms]) => [stage, Math.round(ms)])),
      final: finalRecord,
      renderOptions: { ...appliedOptions },
      dissolve,
      canopyCutoffMeters: Math.round(cutoff),
      stems: drawnStems,
      farForest: { ...farForest.stats },
      stemReachM: Math.round(stemReach),
      stemClasses,
      pastStems,
      invented,
      classPixels: counts,
      missing,
      missingDetail: detail
    }
    return {
      type: 'pgmaps-engine-capture',
      id: request.id,
      engine: CAPTURE_ENGINE,
      version: CAPTURE_VERSION,
      frames,
      scene: { createdAt: pkg.createdAt ?? null, origin: pkg.origin ?? null },
      environment: {
        ...structuredClone(atmosphere.applied),
        sun: { ...atmosphere.sunRecord }
      } as Record<string, unknown>,
      renderOptions: { ...appliedOptions },
      dissolve,
      maxTreeHeightMeters: Number.isFinite(tallest) ? Math.round(tallest * 10) / 10 : null,
      missing,
      missingDetail: detail,
      ...(final ? { final: finalRecord } : {})
    }
  }

  return {
    /** Captures one request (queued behind any capture already running). */
    capture(request: CaptureRequest, onProgress?: CaptureProgress): Promise<CaptureReply> {
      const result = queue.then(() => run(request, onProgress))
      queue = result.catch(() => undefined)
      return result
    },
    /** Milliseconds the last capture took, from the scene being settled to the reply. */
    get lastTimingMs() {
      return lastTimingMs
    },
    /** The last capture's figures: time, stems, the notes and each frame's class pixels by colour. */
    get lastSummary() {
      return lastSummary
    },
    /** Frees the water stand-ins (call when the ground is rebuilt; they are made again on demand). */
    forgetMaterials() {
      for (const m of waterStandIns.values()) m.dispose()
      waterStandIns.clear()
    },
    dispose() {
      this.forgetMaterials()
      nearDrape.set(null)
      farDrape.set(null)
      freeTextures()
      placeholder.dispose()
      addMaterial.dispose()
      sampleTarget.dispose()
      sumTarget.dispose()
      imageTarget.dispose()
      classTarget.dispose()
      screenedTarget.dispose()
      zoneTarget.dispose()
    }
  }
}

export type Capturer = ReturnType<typeof createCapturer>

/** The buffers a reply moves rather than copies. */
export const transferOf = (reply: CaptureReply): Transferable[] =>
  reply.frames.flatMap((f) => [
    f.classes.buffer as ArrayBuffer,
    f.screened.buffer as ArrayBuffer,
    f.depth.buffer as ArrayBuffer,
    ...(f.zones ? [f.zones.buffer as ArrayBuffer] : [])
  ])

// ---------------------------------------------------------------- `?captureTest`: a capture with no parent

/** A rectangle `size` metres a side centred `ahead` metres from `eye` along `bearing`, turned to face it. */
function squareAhead(eye: readonly number[], bearing: number, ahead: number, size: number): Array<[number, number]> {
  const b = (bearing * Math.PI) / 180
  const f = [Math.sin(b), Math.cos(b)]
  const r = [Math.cos(b), -Math.sin(b)]
  const c = [eye[0] + f[0] * ahead, eye[1] + f[1] * ahead]
  const h = size / 2
  const corner = (u: number, v: number): [number, number] => [c[0] + r[0] * u + f[0] * v, c[1] + r[1] * u + f[1] * v]
  return [corner(-h, -h), corner(h, -h), corner(h, h), corner(-h, h), corner(-h, -h)]
}

/** A visualisation of depth: near white, far dark (log scale), sky transparent. */
function depthImage(depth: Float32Array): Uint8ClampedArray {
  const out = new Uint8ClampedArray(depth.length * 4)
  for (let i = 0; i < depth.length; i++) {
    const d = depth[i]
    if (!(d > 0)) continue
    const t = Math.min(1, Math.max(0, Math.log(d) / Math.log(10_000)))
    const v = Math.round(255 * (1 - t))
    out.set([v, v, Math.min(255, v + 30), 255], i * 4)
  }
  return out
}

/** The picture with the class pass laid over it at half strength. */
async function overlayImage(image: Blob, classes: Uint8ClampedArray, width: number, height: number) {
  const canvas = new OffscreenCanvas(width, height)
  const context = canvas.getContext('2d')!
  context.drawImage(await createImageBitmap(image), 0, 0)
  const layer = new OffscreenCanvas(width, height)
  layer.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(classes), width, height), 0, 0)
  context.globalAlpha = 0.5
  context.drawImage(layer, 0, 0)
  return canvas.convertToBlob({ type: 'image/png' })
}

/** Object URLs of a reply's frames for inspection (picture, class pass, depth, overlay). */
export async function inspectReply(reply: CaptureReply) {
  return Promise.all(
    reply.frames.map(async (f) => ({
      image: URL.createObjectURL(f.image),
      classes: URL.createObjectURL(await png(new Uint8ClampedArray(f.classes), f.width, f.height)),
      screened: URL.createObjectURL(await png(new Uint8ClampedArray(f.screened), f.width, f.height)),
      depth: URL.createObjectURL(await png(depthImage(f.depth), f.width, f.height)),
      overlay: URL.createObjectURL(await overlayImage(f.image, f.screened, f.width, f.height))
    }))
  )
}

/**
 * The test request: the package's own eye, one 960 × 540 frame with a 60° horizontal field, a red 100 m square 120 m
 * ahead and a blue 50 m square 70 m ahead and 25° to the right (inside the clearing of the sample package, so both
 * are in sight; the stand's edge is about 190 m out).
 */
export function testRequest(pkg: CapturePackage, size: [number, number] = [960, 540], frames = 1): CaptureRequest {
  const eye = pkg.camera.position
  const bearing = pkg.camera.bearingDegrees
  const [width, height] = size
  const hfov = 60
  const vfov = (2 * Math.atan((Math.tan((hfov * Math.PI) / 360) * height) / width) * 180) / Math.PI
  return {
    type: 'pgmaps-capture',
    id: `capture-test-${Date.now().toString(36)}`,
    position: [eye[0], eye[1], eye[2]],
    frames: Array.from({ length: frames }, (_, k) => ({
      bearingDegrees: (((bearing + (k - (frames - 1) / 2) * hfov) % 360) + 360) % 360,
      pitchDegrees: pkg.camera.pitchDegrees,
      horizontalFovDegrees: hfov,
      verticalFovDegrees: vfov,
      width,
      height
    })),
    classes: {
      terrain: [64, 64, 64],
      layers: [
        { id: 'test-square', rgb: [255, 0, 0], polygons: [[squareAhead(eye, bearing, 120, 100)]] },
        { id: 'test-right', rgb: [0, 0, 255], polygons: [[squareAhead(eye, bearing + 25, 70, 50)]] }
      ],
      vegetation: [255, 255, 255],
      water: [255, 128, 0]
    }
  }
}

/**
 * How well the class pass lines up with the pinhole: every ground pixel (not sky, not a tree) unprojected through its
 * depth with PGMaps' own pinhole, and its class compared with the layer polygons at that point; and each layer's
 * centre, projected with the pinhole onto the drawn ground, with the class found at that pixel.
 */
export function alignmentCheck(
  request: CaptureRequest,
  reply: CaptureReply,
  groundAt: (x: number, y: number) => number
) {
  const frame = request.frames[0]
  const f = reply.frames[0]
  const layers = request.classes.layers
  const inLayer = (x: number, y: number) => {
    let hit: Rgb | null = null
    for (const l of layers)
      for (const polygon of l.polygons)
        if (inRing(polygon[0], x, y) && !polygon.slice(1).some((r) => inRing(r, x, y))) hit = l.rgb
    return hit
  }
  const same = (o: number, rgb: readonly number[]) =>
    f.screened[o] === rgb[0] && f.screened[o + 1] === rgb[1] && f.screened[o + 2] === rgb[2]
  let ground = 0
  let wrong = 0
  for (let row = 0; row < f.height; row++)
    for (let column = 0; column < f.width; column++) {
      const i = row * f.width + column
      const d = f.depth[i]
      if (!(d > 0) || same(i * 4, request.classes.vegetation)) continue
      const [x, y] = unprojectPixel(frame, request.position, column, row, d)
      const expected = inLayer(x, y) ?? request.classes.terrain
      ground++
      if (!same(i * 4, expected)) wrong++
    }
  const centres = layers.map((l) => {
    const ring = l.polygons[0][0]
    const n = ring.length - 1
    const cx = ring.slice(0, n).reduce((s, p) => s + p[0], 0) / n
    const cy = ring.slice(0, n).reduce((s, p) => s + p[1], 0) / n
    const pixel = projectPinhole(frame, request.position, [cx, cy, groundAt(cx, cy)])
    const o = pixel ? (Math.floor(pixel[1]) * f.width + Math.floor(pixel[0])) * 4 : -1
    return {
      id: l.id,
      pixel,
      class: o >= 0 ? [...f.screened.subarray(o, o + 4)] : null,
      depth: o >= 0 ? f.depth[o / 4] : null
    }
  })
  let skyRows = 0
  for (let column = 0; column < f.width; column++)
    if (f.depth[column] === 0 && f.screened[column * 4 + 3] === 0) skyRows++
  return {
    groundPixels: ground,
    groundPixelsOffClass: wrong,
    offShare: ground ? wrong / ground : 0,
    centres,
    topRowSkyShare: skyRows / f.width
  }
}

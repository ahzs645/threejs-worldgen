// The DOM-free parts of the PGMaps capture (`pgmaps-capture`, see pgmaps-scene.ts): a frame's pinhole camera as a
// three.js pose, the class layers rasterised into a texture over the drawn terrain, and the readback buffers turned
// into the class pass and depth the parent expects. No three.js, no DOM: the tests run them in node.
//
// Package metres: x east, y north, z up (absolute). three.js here: y up, north at −z, so (x, y, z) → (x, z, −y).

/** One frame of a capture request (PGMaps `EngineCaptureFrame`). */
export interface CaptureFrame {
  /** Clockwise from north, degrees. */
  bearingDegrees: number
  /** Up from level, degrees. */
  pitchDegrees: number
  horizontalFovDegrees: number
  verticalFovDegrees: number
  /** Pixels. */
  width: number
  height: number
}

export type Rgb = [number, number, number]
/** Package metres: polygons of rings (outer first, then holes) of [x, y] points. */
export type LocalMultiPolygon = Array<Array<Array<[number, number]>>>
export interface ClassLayer {
  id: string
  rgb: Rgb
  polygons: LocalMultiPolygon
}

/** An axis-aligned rectangle in package metres. */
export interface Extent {
  west: number
  south: number
  east: number
  north: number
}

const DEG = Math.PI / 180
type Vec3 = [number, number, number]

/** The frame's forward, right and up axes in package metres (exactly PGMaps' `projectToFrame`). */
export function frameAxes(frame: Pick<CaptureFrame, 'bearingDegrees' | 'pitchDegrees'>): {
  forward: Vec3
  right: Vec3
  up: Vec3
} {
  const b = frame.bearingDegrees * DEG
  const p = frame.pitchDegrees * DEG
  return {
    forward: [Math.sin(b) * Math.cos(p), Math.cos(b) * Math.cos(p), Math.sin(p)],
    right: [Math.cos(b), -Math.sin(b), 0],
    up: [-Math.sin(b) * Math.sin(p), -Math.cos(b) * Math.sin(p), Math.cos(p)]
  }
}

/** A package vector in three.js axes. */
export const toThree = (v: readonly number[]): Vec3 => [v[0], v[2], -v[1]]

/**
 * The three.js camera for a frame: position, orientation (quaternion x, y, z, w), vertical field (degrees, three's
 * `fov`) and aspect. The aspect comes from the fields, `tan(h/2) / tan(v/2)`, never from the pixel size, so the
 * horizontal field is exactly the one asked for even when `width / height` was rounded.
 */
export function frameCamera(
  position: readonly number[],
  frame: CaptureFrame
): { position: Vec3; quaternion: [number, number, number, number]; fov: number; aspect: number } {
  const { forward, right, up } = frameAxes(frame)
  // Camera axes in three.js: +x right, +y up, looking down −z.
  const x = toThree(right)
  const y = toThree(up)
  const z = toThree(forward).map((v) => -v)
  // Rotation matrix with those columns → quaternion (Shepperd's method, as three's setFromRotationMatrix).
  const m11 = x[0],
    m12 = y[0],
    m13 = z[0]
  const m21 = x[1],
    m22 = y[1],
    m23 = z[1]
  const m31 = x[2],
    m32 = y[2],
    m33 = z[2]
  const trace = m11 + m22 + m33
  let q: [number, number, number, number]
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1)
    q = [(m32 - m23) * s, (m13 - m31) * s, (m21 - m12) * s, 0.25 / s]
  } else if (m11 > m22 && m11 > m33) {
    const s = 2 * Math.sqrt(1 + m11 - m22 - m33)
    q = [0.25 * s, (m12 + m21) / s, (m13 + m31) / s, (m32 - m23) / s]
  } else if (m22 > m33) {
    const s = 2 * Math.sqrt(1 + m22 - m11 - m33)
    q = [(m12 + m21) / s, 0.25 * s, (m23 + m32) / s, (m13 - m31) / s]
  } else {
    const s = 2 * Math.sqrt(1 + m33 - m11 - m22)
    q = [(m13 + m31) / s, (m23 + m32) / s, 0.25 * s, (m21 - m12) / s]
  }
  const tanH = Math.tan((frame.horizontalFovDegrees * DEG) / 2)
  const tanV = Math.tan((frame.verticalFovDegrees * DEG) / 2)
  return { position: toThree(position), quaternion: q, fov: frame.verticalFovDegrees, aspect: tanH / tanV }
}

/**
 * PGMaps' pinhole (`figureFraming.ts` `projectToFrame`) in package metres: the pixel a point lands on, or null behind
 * the eye or outside the frame.
 */
export function projectPinhole(
  frame: CaptureFrame,
  eye: readonly number[],
  point: readonly number[]
): [number, number] | null {
  const { forward, right, up } = frameAxes(frame)
  const tanH = Math.tan((frame.horizontalFovDegrees * DEG) / 2)
  const tanV = Math.tan((frame.verticalFovDegrees * DEG) / 2)
  const d = [point[0] - eye[0], point[1] - eye[1], point[2] - eye[2]]
  const dot = (v: Vec3) => v[0] * d[0] + v[1] * d[1] + v[2] * d[2]
  const z = dot(forward)
  if (z <= 1e-6) return null
  const x = frame.width / 2 + (dot(right) / z / tanH) * (frame.width / 2)
  const y = frame.height / 2 - (dot(up) / z / tanV) * (frame.height / 2)
  return x >= 0 && y >= 0 && x <= frame.width && y <= frame.height ? [x, y] : null
}

/** The package point a pixel's centre sees at `depth` metres along the view axis. */
export function unprojectPixel(
  frame: CaptureFrame,
  eye: readonly number[],
  column: number,
  row: number,
  depth: number
): Vec3 {
  const { forward, right, up } = frameAxes(frame)
  const tanH = Math.tan((frame.horizontalFovDegrees * DEG) / 2)
  const tanV = Math.tan((frame.verticalFovDegrees * DEG) / 2)
  const sx = ((column + 0.5 - frame.width / 2) / (frame.width / 2)) * tanH
  const sy = (-(row + 0.5 - frame.height / 2) / (frame.height / 2)) * tanV
  return [0, 1, 2].map((k) => eye[k] + depth * (forward[k] + right[k] * sx + up[k] * sy)) as Vec3
}

/**
 * Share of a frame's drawn pixels (depth > 0) that land outside `extent`: ground the engine invented past the package
 * grid. Every `step`-th pixel each way is tested.
 */
export function shareOutside(
  depth: ArrayLike<number>,
  frame: CaptureFrame,
  eye: readonly number[],
  extent: Extent,
  step = 2
): number {
  let drawn = 0
  let outside = 0
  for (let row = 0; row < frame.height; row += step)
    for (let column = 0; column < frame.width; column += step) {
      const d = depth[row * frame.width + column]
      if (!(d > 0)) continue
      drawn++
      const [x, y] = unprojectPixel(frame, eye, column, row, d)
      if (x < extent.west || x > extent.east || y < extent.south || y > extent.north) outside++
    }
  return drawn ? outside / drawn : 0
}

// ---------------------------------------------------------------- class layers → texture

export interface ClassRaster {
  /** RGBA, row 0 at the south edge; alpha 255 where a layer covers the cell, 0 for unlisted ground. */
  data: Uint8Array
  width: number
  height: number
  /** South-west corner of cell (0, 0), package metres. */
  west: number
  south: number
  /** Cell size, metres (square cells). */
  cell: number
}

const intersect = (a: Extent, b: Extent): Extent | null => {
  const e = {
    west: Math.max(a.west, b.west),
    south: Math.max(a.south, b.south),
    east: Math.min(a.east, b.east),
    north: Math.min(a.north, b.north)
  }
  return e.east > e.west && e.north > e.south ? e : null
}

/** Bounding box of every ring point of the layers, or null when there are none. */
export function layersExtent(layers: readonly ClassLayer[]): Extent | null {
  let west = Infinity
  let south = Infinity
  let east = -Infinity
  let north = -Infinity
  for (const layer of layers)
    for (const polygon of layer.polygons)
      for (const ring of polygon)
        for (const [x, y] of ring) {
          if (x < west) west = x
          if (x > east) east = x
          if (y < south) south = y
          if (y > north) north = y
        }
  return Number.isFinite(west) && east > west && north > south ? { west, south, east, north } : null
}

/**
 * Rasterises the class layers, bottom first (a later layer paints over an earlier one), into cells over the part of
 * `drawn` (the drawn terrain) the polygons cover, at most `maxSize` cells a side and never finer than `minCell`
 * metres. A cell takes a layer's colour when its centre is inside the polygon (even-odd over its rings, so holes stay
 * out). Polygons past `drawn` are cut: see {@link clippedShare}.
 */
export function rasterizeClassLayers(
  layers: readonly ClassLayer[],
  drawn: Extent,
  options: { maxSize?: number; minCell?: number } = {}
): ClassRaster {
  const maxSize = options.maxSize ?? 4096
  const minCell = options.minCell ?? 0.5
  const bounds = layersExtent(layers)
  const area = bounds && intersect(bounds, drawn)
  if (!area) return { data: new Uint8Array(4), width: 1, height: 1, west: drawn.west, south: drawn.south, cell: 1e-6 }
  const cell = Math.max(minCell, (area.east - area.west) / maxSize, (area.north - area.south) / maxSize)
  const width = Math.max(1, Math.ceil((area.east - area.west) / cell))
  const height = Math.max(1, Math.ceil((area.north - area.south) / cell))
  const data = new Uint8Array(width * height * 4)
  const crossings: number[] = []
  for (const layer of layers) {
    const [r, g, b] = layer.rgb
    for (const polygon of layer.polygons) {
      let top = -Infinity
      let bottom = Infinity
      for (const [, y] of polygon[0] ?? []) {
        top = Math.max(top, y)
        bottom = Math.min(bottom, y)
      }
      const j0 = Math.max(0, Math.floor((bottom - area.south) / cell - 0.5))
      const j1 = Math.min(height - 1, Math.ceil((top - area.south) / cell - 0.5))
      for (let j = j0; j <= j1; j++) {
        const y = area.south + (j + 0.5) * cell
        crossings.length = 0
        for (const ring of polygon)
          for (let i = 0, k = ring.length - 1; i < ring.length; k = i++) {
            const [xa, ya] = ring[k]
            const [xb, yb] = ring[i]
            if (ya <= y !== yb <= y) crossings.push(xa + ((y - ya) * (xb - xa)) / (yb - ya))
          }
        crossings.sort((a, b) => a - b)
        for (let c = 0; c + 1 < crossings.length; c += 2) {
          const i0 = Math.max(0, Math.ceil((crossings[c] - area.west) / cell - 0.5))
          const i1 = Math.min(width, Math.ceil((crossings[c + 1] - area.west) / cell - 0.5))
          for (let i = i0; i < i1; i++) {
            const o = (j * width + i) * 4
            data[o] = r
            data[o + 1] = g
            data[o + 2] = b
            data[o + 3] = 255
          }
        }
      }
    }
  }
  return { data, width, height, west: area.west, south: area.south, cell }
}

const ringArea = (ring: ReadonlyArray<readonly number[]>) => {
  let sum = 0
  for (let i = 0, k = ring.length - 1; i < ring.length; k = i++)
    sum += ring[k][0] * ring[i][1] - ring[i][0] * ring[k][1]
  return Math.abs(sum) / 2
}

/** A ring clipped to a rectangle (Sutherland–Hodgman). */
function clipRing(ring: ReadonlyArray<readonly number[]>, e: Extent): number[][] {
  let out: number[][] = ring.map((p) => [p[0], p[1]])
  const edges: Array<[(p: number[]) => boolean, (a: number[], b: number[]) => number[]]> = [
    [(p) => p[0] >= e.west, (a, b) => [e.west, a[1] + ((e.west - a[0]) * (b[1] - a[1])) / (b[0] - a[0])]],
    [(p) => p[0] <= e.east, (a, b) => [e.east, a[1] + ((e.east - a[0]) * (b[1] - a[1])) / (b[0] - a[0])]],
    [(p) => p[1] >= e.south, (a, b) => [a[0] + ((e.south - a[1]) * (b[0] - a[0])) / (b[1] - a[1]), e.south]],
    [(p) => p[1] <= e.north, (a, b) => [a[0] + ((e.north - a[1]) * (b[0] - a[0])) / (b[1] - a[1]), e.north]]
  ]
  for (const [inside, cross] of edges) {
    const input = out
    out = []
    for (let i = 0; i < input.length; i++) {
      const a = input[(i + input.length - 1) % input.length]
      const b = input[i]
      if (inside(b)) {
        if (!inside(a)) out.push(cross(a, b))
        out.push(b)
      } else if (inside(a)) out.push(cross(a, b))
    }
    if (!out.length) break
  }
  return out
}

/** Share of a layer's area (holes left out) that lies outside `extent`: 0 when it is all inside. */
export function clippedShare(polygons: LocalMultiPolygon, extent: Extent): number {
  let total = 0
  let inside = 0
  for (const polygon of polygons)
    polygon.forEach((ring, k) => {
      const sign = k === 0 ? 1 : -1
      total += sign * ringArea(ring)
      inside += sign * ringArea(clipRing(ring, extent))
    })
  return total > 0 ? Math.min(1, Math.max(0, 1 - inside / total)) : 0
}

// ---------------------------------------------------------------- readback

/**
 * A readback as rows from the top, `width × channels` values each: drops the 256-byte row padding a WebGPU copy adds
 * (inferred from the length) and turns WebGL's bottom-up rows over (`bottomUp`).
 */
export function readbackRows<T extends Uint8Array | Uint8ClampedArray | Float32Array>(
  data: T,
  width: number,
  height: number,
  channels: number,
  bottomUp: boolean
): T {
  const row = width * channels
  const stride = height > 1 ? (data.length - row) / (height - 1) : row
  if (!Number.isInteger(stride) || stride < row)
    throw new Error(`A ${width} × ${height} readback of ${data.length} values.`)
  if (stride === row && !bottomUp) return data
  const out = new (data.constructor as new (n: number) => T)(row * height)
  for (let y = 0; y < height; y++) {
    const from = (bottomUp ? height - 1 - y : y) * stride
    out.set(data.subarray(from, from + row) as unknown as ArrayLike<number>, y * row)
  }
  return out
}

/**
 * The class pass and depth from one float RGBA readback (rows from the top) in which the class pass wrote its flat
 * colour as rgb (0..1) and the view-axis distance in metres as alpha, over a clear of 0: classes as straight RGBA
 * bytes (alpha 0 for the sky), depth as metres (0 for the sky).
 */
export function splitClassDepth(
  pixels: Float32Array,
  width: number,
  height: number
): { classes: Uint8ClampedArray; depth: Float32Array } {
  const n = width * height
  const classes = new Uint8ClampedArray(n * 4)
  const depth = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const d = pixels[i * 4 + 3]
    if (!(d > 0) || !Number.isFinite(d)) continue
    depth[i] = d
    classes[i * 4] = Math.round(pixels[i * 4] * 255)
    classes[i * 4 + 1] = Math.round(pixels[i * 4 + 1] * 255)
    classes[i * 4 + 2] = Math.round(pixels[i * 4 + 2] * 255)
    classes[i * 4 + 3] = 255
  }
  return { classes, depth }
}

/**
 * Far plane for a capture: past the farthest corner of the drawn ground (terrain grid and apron) as seen from the eye,
 * never nearer than `minimum`.
 */
export function captureFar(eye: readonly number[], drawn: Extent, apronRadius: number, minimum: number): number {
  let far = minimum
  for (const x of [drawn.west, drawn.east])
    for (const y of [drawn.south, drawn.north]) far = Math.max(far, Math.hypot(x - eye[0], y - eye[1]) + 100)
  far = Math.max(far, apronRadius + Math.hypot(eye[0], eye[1]) + 100)
  return Math.ceil(far / 100) * 100
}

/**
 * Where a frame's drawn pixels land: unprojects every `step`-th pixel each way through its depth and counts, among the
 * pixels `counted` keeps (by RGBA offset into `classes`), those `test` accepts at their package point. Returns the
 * share of all drawn pixels sampled.
 */
export function pixelShare(
  depth: ArrayLike<number>,
  frame: CaptureFrame,
  eye: readonly number[],
  test: (x: number, y: number) => boolean,
  counted: (offset: number) => boolean = () => true,
  step = 2
): number {
  let drawn = 0
  let hits = 0
  for (let row = 0; row < frame.height; row += step)
    for (let column = 0; column < frame.width; column += step) {
      const i = row * frame.width + column
      const d = depth[i]
      if (!(d > 0)) continue
      drawn++
      if (!counted(i * 4)) continue
      const [x, y] = unprojectPixel(frame, eye, column, row, d)
      if (test(x, y)) hits++
    }
  return drawn ? hits / drawn : 0
}

/** Pixels of each listed colour in a straight-RGBA class pass (alpha > 0 only), keyed by the colour's name. */
export function colourCounts(
  classes: ArrayLike<number>,
  colours: Record<string, readonly number[]>
): Record<string, number> & { other: number; sky: number } {
  const out: Record<string, number> = Object.fromEntries(Object.keys(colours).map((k) => [k, 0]))
  const entries = Object.entries(colours)
  let other = 0
  let sky = 0
  for (let o = 0; o < classes.length; o += 4) {
    if (!classes[o + 3]) {
      sky++
      continue
    }
    const hit = entries.find(([, c]) => classes[o] === c[0] && classes[o + 1] === c[1] && classes[o + 2] === c[2])
    if (hit) out[hit[0]]++
    else other++
  }
  return Object.assign(out, { other, sky })
}

/** Kilometres for a short note: "0.6 km", "5.6 km", "15 km". */
export function km(metres: number): string {
  const k = metres / 1000
  return `${k >= 10 ? Math.round(k) : Math.round(k * 10) / 10} km`
}

/** A percentage for a short note: "48%", "2.5%", "0.4%". */
export function percent(share: number): string {
  const p = share * 100
  return `${p >= 10 ? Math.round(p) : Math.round(p * 10) / 10}%`
}

// ---------------------------------------------------------------- the class of each stem

/**
 * A stem takes the class of the ground under its own stem: a tree standing on the landform is the landform's (a VIA
 * counts a landform's own forest as part of it), and only a tree standing on unlisted ground (or water) screens.
 * The listed classes' colours, each once in layer order; a stem's code is 1 + its colour's index, 0 for unlisted.
 */
export function stemClassPalette(layers: readonly ClassLayer[]): Rgb[] {
  const out: Rgb[] = []
  for (const l of layers)
    if (!out.some((c) => c[0] === l.rgb[0] && c[1] === l.rgb[1] && c[2] === l.rgb[2])) out.push(l.rgb)
  return out
}

/** The raster's colour in the cell holding a point (as the drape samples it, nearest), or null where no layer is. */
export function rasterColourAt(raster: ClassRaster, x: number, y: number): Rgb | null {
  const i = Math.floor((x - raster.west) / raster.cell)
  const j = Math.floor((y - raster.south) / raster.cell)
  if (i < 0 || j < 0 || i >= raster.width || j >= raster.height) return null
  const o = (j * raster.width + i) * 4
  return raster.data[o + 3] ? [raster.data[o], raster.data[o + 1], raster.data[o + 2]] : null
}

/**
 * A stem's class code at its base (x, y), package metres: the near raster inside `drawn` (the package's terrain grid),
 * the far raster past it, as the class pass drapes them. 0 for unlisted ground, else 1 + the colour's index in
 * `palette`.
 */
export function stemClassCode(
  rasters: { near: ClassRaster; far: ClassRaster | null },
  drawn: Extent,
  palette: readonly Rgb[],
  x: number,
  y: number
): number {
  const inNear = x >= drawn.west && x <= drawn.east && y >= drawn.south && y <= drawn.north
  const raster = inNear ? rasters.near : rasters.far
  const c = raster && rasterColourAt(raster, x, y)
  if (!c) return 0
  const k = palette.findIndex((p) => p[0] === c[0] && p[1] === c[1] && p[2] === c[2])
  return k < 0 ? 0 : k + 1
}

const TURN = Math.PI * 2
/** Keeps a yaw's whole turns from ever being misread: the base yaw stays this far inside [0, 2π). */
const YAW_MARGIN = 1e-3

/**
 * A stem's yaw carrying its class code as whole turns: the same rotation (every use of the yaw goes through its sine
 * and cosine), and `floor(yaw / 2π)` is the code. Base yaws of any sign are folded into [0, 2π) first.
 */
export function encodeStemClass(yaw: number, code: number): number {
  const base = ((yaw % TURN) + TURN) % TURN
  return Math.min(TURN - YAW_MARGIN, Math.max(YAW_MARGIN, base)) + TURN * code
}

/** The class code a yaw carries ({@link encodeStemClass}). */
export const stemClassOf = (yaw: number): number => Math.floor(yaw / TURN)

/**
 * A copy of the yaws of `count` records, each `stride` values long with its yaw at `offset`: what a capture codes
 * from, and what it puts back ({@link putYaws}) so that no class code stays in the scene after it. A capture never
 * codes a yaw an earlier capture coded: a float32 yaw loses its last bits under whole turns.
 */
export function takeYaws(values: ArrayLike<number>, stride: number, offset: number, count: number): Float32Array {
  const yaws = new Float32Array(count)
  for (let n = 0; n < count; n++) yaws[n] = values[n * stride + offset]
  return yaws
}

/** Writes yaws taken by {@link takeYaws} back into their records. */
export function putYaws(
  values: Float32Array | number[],
  stride: number,
  offset: number,
  yaws: ArrayLike<number>
): void {
  for (let n = 0; n < yaws.length; n++) values[n * stride + offset] = yaws[n]
}

// ---------------------------------------------------------------- the picture's guard band

/** Columns drawn past each side of a frame's picture and cropped away: a tenth of its width, at least 16. */
export const pictureGuard = (width: number): number => Math.max(16, Math.ceil(width * 0.1))

/**
 * The aspect of a frame's camera widened by `guard` columns each side: the same pinhole (vertical field, focal
 * length in pixels), so the middle `width` columns of the wide picture are exactly the frame's pixels.
 */
export const guardedAspect = (aspect: number, width: number, guard: number): number =>
  (aspect * (width + 2 * guard)) / width

/** Columns `from … from + width` of rows `wide` pixels across, `channels` values a pixel. */
export function cropColumns<T extends Uint8Array | Uint8ClampedArray | Float32Array>(
  data: T,
  wide: number,
  height: number,
  channels: number,
  from: number,
  width: number
): T {
  if (from === 0 && width === wide) return data
  const out = new (data.constructor as new (n: number) => T)(width * height * channels)
  for (let y = 0; y < height; y++)
    out.set(
      data.subarray((y * wide + from) * channels, (y * wide + from + width) * channels) as unknown as ArrayLike<number>,
      y * width * channels
    )
  return out
}

// ---------------------------------------------------------------- content hashes

const HASH_F64 = new Float64Array(1)
const HASH_U32 = new Uint32Array(HASH_F64.buffer)
/** Counts values {@link contentHash} cannot read, so that each hashes unlike anything else. */
let unhashable = 0
const HASH_TAG = {
  array: 1,
  typed: 2,
  object: 3,
  string: 4,
  number: 5,
  yes: 6,
  no: 7,
  null: 8,
  undefined: 9,
  date: 10,
  other: 11
}
const TYPED_NAMES = [
  'Float32Array',
  'Float64Array',
  'Int8Array',
  'Uint8Array',
  'Uint8ClampedArray',
  'Int16Array',
  'Uint16Array',
  'Int32Array',
  'Uint32Array',
  'BigInt64Array',
  'BigUint64Array',
  'DataView',
  'ArrayBuffer'
]

/**
 * A hash of a value's whole content, to tell whether two messages (each a structured clone, so never the same
 * object) say the same thing: every element of every array, plain or typed, every key, string, number and flag, object
 * keys in sorted order. Nothing is sampled. `skip` leaves out top-level keys. 16 hex digits: two 32-bit lanes
 * (MurmurHash3's body and xxHash32's round), each a bijection of its state for a given word and of the word for a
 * given state, so one changed 32-bit word always changes both lanes; any other difference collides about once in
 * 2⁶⁴. A number is its 64 bits (so 0 and −0 differ, and 1 in a plain array is not 1 in a Float32Array). A value it
 * cannot read (a Map, a Set, a function, a class instance) hashes unlike anything, itself included.
 */
export function contentHash(value: unknown, skip?: ReadonlySet<string>): string {
  let a = 0x2f0b4a13
  let b = 0x7ed55d16
  let words = 0
  const word = (w: number) => {
    let k = Math.imul(w, 0xcc9e2d51)
    k = (k << 15) | (k >>> 17)
    a ^= Math.imul(k, 0x1b873593)
    a = (a << 13) | (a >>> 19)
    a = (Math.imul(a, 5) + 0xe6546b64) | 0
    b = (b + Math.imul(w, 0x85ebca77)) | 0
    b = (b << 13) | (b >>> 19)
    b = Math.imul(b, 0x9e3779b1)
    words++
  }
  const text = (s: string) => {
    word(s.length)
    for (let i = 0; i < s.length; i++) word(s.charCodeAt(i))
  }
  /** The 32-bit words of a run of bytes (the last word short of four bytes is padded with zeros). */
  const bytes = (buffer: ArrayBufferLike, offset: number, length: number) => {
    word(length)
    const whole = length >>> 2
    const view =
      offset % 4 === 0
        ? new Uint32Array(buffer, offset, whole)
        : new Uint32Array(new Uint8Array(buffer, offset, whole * 4).slice().buffer)
    // The lanes as locals: this loop reads tens of megabytes.
    let la = a
    let lb = b
    for (let i = 0; i < whole; i++) {
      const w = view[i]
      let k = Math.imul(w, 0xcc9e2d51)
      k = (k << 15) | (k >>> 17)
      la ^= Math.imul(k, 0x1b873593)
      la = (la << 13) | (la >>> 19)
      la = (Math.imul(la, 5) + 0xe6546b64) | 0
      lb = (lb + Math.imul(w, 0x85ebca77)) | 0
      lb = (lb << 13) | (lb >>> 19)
      lb = Math.imul(lb, 0x9e3779b1)
    }
    a = la
    b = lb
    words += whole
    if (length > whole * 4) {
      const tail = new Uint8Array(buffer, offset + whole * 4, length - whole * 4)
      let w = 0
      for (let i = 0; i < tail.length; i++) w |= tail[i] << (8 * i)
      word(w)
    }
  }
  /**
   * An array: its length, each element's 64 bits where it is a number (zeros where it is not), then how many are not
   * numbers and each of those behind its index.
   */
  const list = (values: readonly unknown[]) => {
    word(HASH_TAG.array)
    word(values.length)
    let others = 0
    let la = a
    let lb = b
    for (let i = 0; i < values.length; i++) {
      const v = values[i]
      let lo = 0
      let hi = 0
      if (typeof v === 'number') {
        HASH_F64[0] = v
        lo = HASH_U32[0]
        hi = HASH_U32[1]
      } else others++
      let k = Math.imul(lo, 0xcc9e2d51)
      k = (k << 15) | (k >>> 17)
      la ^= Math.imul(k, 0x1b873593)
      la = (la << 13) | (la >>> 19)
      la = (Math.imul(la, 5) + 0xe6546b64) | 0
      lb = (lb + Math.imul(lo, 0x85ebca77)) | 0
      lb = (lb << 13) | (lb >>> 19)
      lb = Math.imul(lb, 0x9e3779b1)
      k = Math.imul(hi, 0xcc9e2d51)
      k = (k << 15) | (k >>> 17)
      la ^= Math.imul(k, 0x1b873593)
      la = (la << 13) | (la >>> 19)
      la = (Math.imul(la, 5) + 0xe6546b64) | 0
      lb = (lb + Math.imul(hi, 0x85ebca77)) | 0
      lb = (lb << 13) | (lb >>> 19)
      lb = Math.imul(lb, 0x9e3779b1)
    }
    a = la
    b = lb
    words += values.length * 2
    word(others)
    if (!others) return
    for (let i = 0; i < values.length; i++) {
      const v = values[i]
      if (typeof v === 'number') continue
      word(i)
      any(v, null)
    }
  }
  const any = (v: unknown, skipKeys: ReadonlySet<string> | null) => {
    if (v === null) return word(HASH_TAG.null)
    if (v === undefined) return word(HASH_TAG.undefined)
    if (typeof v === 'number') {
      word(HASH_TAG.number)
      HASH_F64[0] = v
      word(HASH_U32[0])
      return word(HASH_U32[1])
    }
    if (typeof v === 'string') {
      word(HASH_TAG.string)
      return text(v)
    }
    if (typeof v === 'boolean') return word(v ? HASH_TAG.yes : HASH_TAG.no)
    if (typeof v === 'object') {
      if (Array.isArray(v)) return list(v)
      if (ArrayBuffer.isView(v) || v instanceof ArrayBuffer) {
        word(HASH_TAG.typed)
        word(TYPED_NAMES.indexOf(Object.prototype.toString.call(v).slice(8, -1)))
        if (v instanceof ArrayBuffer) return bytes(v, 0, v.byteLength)
        return bytes(v.buffer, v.byteOffset, v.byteLength)
      }
      if (v instanceof Date) {
        word(HASH_TAG.date)
        HASH_F64[0] = v.getTime()
        word(HASH_U32[0])
        return word(HASH_U32[1])
      }
      const prototype = Object.getPrototypeOf(v)
      if (prototype === Object.prototype || prototype === null) {
        const keys = Object.keys(v).filter((key) => !skipKeys?.has(key)).sort()
        word(HASH_TAG.object)
        word(keys.length)
        for (const key of keys) {
          text(key)
          any((v as Record<string, unknown>)[key], null)
        }
        return
      }
    }
    // Not something a message's content is read from here: never equal to anything.
    word(HASH_TAG.other)
    word(++unhashable)
  }
  any(value, skip ?? null)
  const mix = (h: number) => {
    h ^= h >>> 16
    h = Math.imul(h, 0x85ebca6b)
    h ^= h >>> 13
    h = Math.imul(h, 0xc2b2ae35)
    return (h ^ (h >>> 16)) >>> 0
  }
  const hex = (h: number) => h.toString(16).padStart(8, '0')
  return hex(mix(a ^ words)) + hex(mix(b ^ Math.imul(words, 0x27d4eb2f)))
}

/**
 * The fields of a scene package a build reads nothing from but the year its view is set in (`viewYear`,
 * pgmaps-burn.ts): they are taken from each package as it arrives, built again or not.
 */
export const SCENE_UNBUILT_FIELDS: ReadonlySet<string> = new Set([
  'createdAt',
  'camera',
  'environment',
  'origin',
  'notes'
])

/**
 * What a scene package builds, as a key: {@link contentHash} of everything in it but {@link SCENE_UNBUILT_FIELDS},
 * and the year its view is set in (which ages its fires). Two packages with the same key build the same scene.
 */
export function scenePackageKey(pkg: unknown, viewYear: number | null): string {
  return `${contentHash(pkg, SCENE_UNBUILT_FIELDS)}/${viewYear ?? 'no year'}`
}

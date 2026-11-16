import { test } from 'node:test'
import assert from 'node:assert/strict'
import { PerspectiveCamera, Vector3 } from 'three'
import {
  captureFar,
  clippedShare,
  colourCounts,
  cropColumns,
  frameCamera,
  guardedAspect,
  km,
  percent,
  pixelShare,
  projectPinhole,
  rasterizeClassLayers,
  readbackRows,
  shareOutside,
  splitClassDepth,
  stemClassCode,
  stemClassOf,
  stemClassPalette,
  encodeStemClass,
  contentHash,
  putYaws,
  scenePackageKey,
  SCENE_UNBUILT_FIELDS,
  takeYaws,
  unprojectPixel,
  type CaptureFrame,
  type ClassLayer
} from './pgmaps-capture-math.ts'

/** A seeded generator for the fixtures (mulberry32). */
function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const DEG = Math.PI / 180
/** A frame of `width × height` pixels with horizontal field `hfov`, the vertical field following from the aspect. */
const frameOf = (bearing: number, pitch: number, hfov: number, width: number, height: number): CaptureFrame => ({
  bearingDegrees: bearing,
  pitchDegrees: pitch,
  horizontalFovDegrees: hfov,
  verticalFovDegrees: (2 * Math.atan((Math.tan((hfov * DEG) / 2) * height) / width)) / DEG,
  width,
  height
})

/** The pixel three.js draws a package point at, with the camera `frameCamera` sets up. */
function threePixel(frame: CaptureFrame, eye: number[], point: number[]): [number, number] | null {
  const pose = frameCamera(eye, frame)
  const camera = new PerspectiveCamera(pose.fov, pose.aspect, 0.3, 30000)
  camera.position.set(...pose.position)
  camera.quaternion.set(...pose.quaternion)
  camera.updateMatrixWorld()
  camera.updateProjectionMatrix()
  const p = new Vector3(point[0], point[2], -point[1]).project(camera)
  if (p.z < -1 || p.z > 1) return null
  return [((p.x + 1) / 2) * frame.width, ((1 - p.y) / 2) * frame.height]
}

test("three's camera puts every point on the pixel PGMaps' pinhole does, to half a pixel", () => {
  const rand = rng(7)
  let checked = 0
  for (let k = 0; k < 200; k++) {
    const frame = frameOf(
      rand() * 360,
      (rand() - 0.5) * 60,
      30 + rand() * 70,
      [960, 1200, 1600, 800][k % 4],
      [540, 900, 1200, 800][k % 4]
    )
    const eye = [(rand() - 0.5) * 4000, (rand() - 0.5) * 4000, 600 + rand() * 300]
    for (let i = 0; i < 20; i++) {
      // Points in front of the eye, spread across the frame and out to 6 km.
      const d = 5 + rand() ** 2 * 6000
      const b = (frame.bearingDegrees + (rand() - 0.5) * frame.horizontalFovDegrees) * DEG
      const point = [eye[0] + Math.sin(b) * d, eye[1] + Math.cos(b) * d, eye[2] + (rand() - 0.6) * d * 0.4]
      const pinhole = projectPinhole(frame, eye, point)
      if (!pinhole) continue
      const drawn = threePixel(frame, eye, point)
      assert.ok(drawn, `three drew nothing for a point the pinhole put at ${pinhole}`)
      assert.ok(
        Math.hypot(drawn[0] - pinhole[0], drawn[1] - pinhole[1]) < 0.5,
        `frame ${JSON.stringify(frame)}: pinhole ${pinhole}, three ${drawn}`
      )
      checked++
    }
  }
  assert.ok(checked > 2000, `only ${checked} points checked`)
})

test('the camera looks along the bearing with no roll, and its aspect comes from the fields', () => {
  const frame = frameOf(90, 0, 60, 1200, 900)
  const pose = frameCamera([10, 20, 700], frame)
  const camera = new PerspectiveCamera(pose.fov, pose.aspect)
  camera.quaternion.set(...pose.quaternion)
  const forward = new Vector3(0, 0, -1).applyQuaternion(camera.quaternion)
  const up = new Vector3(0, 1, 0).applyQuaternion(camera.quaternion)
  // Bearing 90 is east: three's +x.
  assert.ok(forward.distanceTo(new Vector3(1, 0, 0)) < 1e-12)
  assert.ok(up.distanceTo(new Vector3(0, 1, 0)) < 1e-12)
  assert.deepEqual(pose.position, [10, 700, -20])
  assert.ok(Math.abs(pose.aspect - 1200 / 900) < 1e-12)
  // A field pair that does not match the pixels keeps the fields (the pinhole's), not the pixels.
  const skew = { ...frame, verticalFovDegrees: 40 }
  assert.ok(Math.abs(frameCamera([0, 0, 0], skew).aspect - Math.tan(30 * DEG) / Math.tan(20 * DEG)) < 1e-12)
})

test('a pixel unprojected at its depth projects back onto itself', () => {
  const frame = frameOf(200, -5, 60, 960, 540)
  const eye = [100, -50, 680]
  for (const [c, r, d] of [
    [0, 0, 300],
    [480, 270, 1200],
    [959, 539, 40]
  ]) {
    const p = unprojectPixel(frame, eye, c, r, d)
    const back = projectPinhole(frame, eye, p)
    assert.ok(back && Math.abs(back[0] - (c + 0.5)) < 1e-6 && Math.abs(back[1] - (r + 0.5)) < 1e-6)
  }
  // Every pixel at 10 km is outside a 6 km grid; at 10 m it is inside.
  const extent = { west: -3000, south: -3000, east: 3000, north: 3000 }
  assert.equal(shareOutside(new Float32Array(960 * 540).fill(10_000), frame, eye, extent), 1)
  assert.equal(shareOutside(new Float32Array(960 * 540).fill(10), frame, eye, extent), 0)
  assert.equal(shareOutside(new Float32Array(960 * 540), frame, eye, extent), 0)
})

const drawn = { west: -3000, south: -3000, east: 3000, north: 3000 }

test('class layers rasterise bottom first, holes stay out, cells are taken by their centres', () => {
  const square = (x0: number, y0: number, s: number): Array<[number, number]> => [
    [x0, y0],
    [x0 + s, y0],
    [x0 + s, y0 + s],
    [x0, y0 + s],
    [x0, y0]
  ]
  const raster = rasterizeClassLayers(
    [
      { id: 'landform', rgb: [10, 20, 30], polygons: [[square(0, 0, 100), square(40, 40, 20)]] },
      { id: 'block', rgb: [200, 0, 0], polygons: [[square(80, 80, 20)]] }
    ],
    drawn,
    { minCell: 1 }
  )
  assert.equal(raster.cell, 1)
  assert.equal(raster.width, 100)
  assert.equal(raster.height, 100)
  assert.equal(raster.west, 0)
  const at = (x: number, y: number) => {
    const o =
      (Math.floor((y - raster.south) / raster.cell) * raster.width + Math.floor((x - raster.west) / raster.cell)) * 4
    return [...raster.data.subarray(o, o + 4)]
  }
  assert.deepEqual(at(10.5, 10.5), [10, 20, 30, 255])
  assert.deepEqual(at(50.5, 50.5), [0, 0, 0, 0]) // the hole
  assert.deepEqual(at(90.5, 90.5), [200, 0, 0, 255]) // the later layer on top
  assert.deepEqual(at(10.5, 90.5), [10, 20, 30, 255]) // row 0 is south: the block is north-east only
  assert.deepEqual(at(90.5, 10.5), [10, 20, 30, 255])
  let covered = 0
  for (let i = 3; i < raster.data.length; i += 4) if (raster.data[i]) covered++
  assert.equal(covered, 100 * 100 - 20 * 20)
})

test('the raster never exceeds its size and stops at the drawn terrain', () => {
  const big = rasterizeClassLayers(
    [
      {
        id: 'far',
        rgb: [1, 2, 3],
        polygons: [
          [
            [
              [-5000, -1000],
              [5000, -1000],
              [5000, 1000],
              [-5000, 1000]
            ]
          ]
        ]
      }
    ],
    drawn,
    { maxSize: 1024 }
  )
  assert.ok(big.width <= 1024 && big.height <= 1024)
  assert.equal(big.west, -3000)
  assert.ok(Math.abs(big.west + big.width * big.cell - 3000) < big.cell)
  const none = rasterizeClassLayers([], drawn)
  assert.equal(none.data[3], 0)
})

test('the share of a layer past the drawn terrain is its clipped area', () => {
  const half = [
    [
      [2000, 0],
      [4000, 0],
      [4000, 1000],
      [2000, 1000]
    ]
  ] as Array<Array<[number, number]>>
  assert.ok(Math.abs(clippedShare([half], drawn) - 0.5) < 1e-12)
  assert.equal(clippedShare([[half[0].map(([x, y]) => [x - 3000, y] as [number, number])]], drawn), 0)
  // A hole outside the terrain does not count as cut ground.
  const holed = [
    [
      [2000, 0],
      [4000, 0],
      [4000, 1000],
      [2000, 1000]
    ],
    [
      [3200, 200],
      [3800, 200],
      [3800, 800],
      [3200, 800]
    ]
  ] as Array<Array<[number, number]>>
  assert.ok(Math.abs(clippedShare([holed], drawn) - (1_000_000 - 360_000) / (2_000_000 - 360_000)) < 1e-12)
})

test('readback rows lose their padding and come out top first', () => {
  // 3 × 2 RGBA bytes, rows padded to 256 bytes, the last row unpadded (as WebGPU's copy is sized).
  const padded = new Uint8Array(256 + 12)
  padded.set([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 0)
  padded.set([4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6], 256)
  assert.deepEqual(
    [...readbackRows(padded, 3, 2, 4, false)],
    [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6]
  )
  const bottomUp = new Float32Array([4, 5, 6, 1, 2, 3])
  assert.deepEqual([...readbackRows(bottomUp, 3, 2, 1, true)], [1, 2, 3, 4, 5, 6])
  assert.throws(() => readbackRows(new Uint8Array(10), 3, 2, 4, false))
})

test('the class pass and depth split from one float readback; sky is transparent and 0', () => {
  const pixels = new Float32Array([1, 0, 0, 512.5, 0, 0, 0, 0, 1, 1, 1, 12.25])
  const { classes, depth } = splitClassDepth(pixels, 3, 1)
  assert.deepEqual([...classes], [255, 0, 0, 255, 0, 0, 0, 0, 255, 255, 255, 255])
  assert.deepEqual([...depth], [512.5, 0, 12.25])
})

test('the far plane reaches the farthest drawn ground', () => {
  assert.equal(captureFar([0, 0, 700], drawn, 7600, 8000), 8000)
  assert.equal(captureFar([0, 0, 700], drawn, 7600, 1000), 7700)
  assert.equal(captureFar([2000, 2000, 700], { west: -6000, south: -6000, east: 6000, north: 6000 }, 7600, 8000), 11500)
})

test('pixel shares unproject through depth and count only the pixels asked about', () => {
  const frame: CaptureFrame = {
    bearingDegrees: 0,
    pitchDegrees: 0,
    horizontalFovDegrees: 60,
    verticalFovDegrees: 40,
    width: 8,
    height: 4
  }
  // Top two rows sky, the third 100 m off, the last 2000 m off.
  const depth = new Float32Array(32)
  depth.fill(100, 16, 24)
  depth.fill(2000, 24, 32)
  const far = pixelShare(
    depth,
    frame,
    [0, 0, 0],
    (_x, y) => y > 1000,
    () => true,
    1
  )
  assert.equal(far, 0.5)
  const none = pixelShare(
    depth,
    frame,
    [0, 0, 0],
    () => true,
    () => false,
    1
  )
  assert.equal(none, 0)
  assert.equal(
    pixelShare(new Float32Array(32), frame, [0, 0, 0], () => true),
    0
  )
})

test('colour counts key pixels by colour, and the rest as other or sky', () => {
  const classes = new Uint8ClampedArray([255, 255, 255, 255, 255, 128, 0, 255, 1, 2, 3, 255, 0, 0, 0, 0])
  assert.deepEqual(colourCounts(classes, { vegetation: [255, 255, 255], water: [255, 128, 0] }), {
    vegetation: 1,
    water: 1,
    other: 1,
    sky: 1
  })
})

test('short notes round kilometres and percentages for the sheet', () => {
  assert.equal(km(600), '0.6 km')
  assert.equal(km(5640), '5.6 km')
  assert.equal(km(15_000), '15 km')
  assert.equal(percent(0.4812), '48%')
  assert.equal(percent(0.025), '2.5%')
})

test('a stem takes the class of the ground under its own stem; only unlisted ground screens', () => {
  const landform: ClassLayer = {
    id: 'landform',
    rgb: [255, 0, 255],
    polygons: [
      [
        [
          [0, 0],
          [100, 0],
          [100, 100],
          [0, 100],
          [0, 0]
        ]
      ]
    ]
  }
  const block: ClassLayer = {
    id: 'proposedIn',
    rgb: [0, 255, 0],
    polygons: [
      [
        [
          [40, 40],
          [60, 40],
          [60, 60],
          [40, 60],
          [40, 40]
        ]
      ]
    ]
  }
  // A second layer sharing the landform's colour adds no palette entry.
  const twin: ClassLayer = { ...landform, id: 'landform-2' }
  const layers = [landform, twin, block]
  const palette = stemClassPalette(layers)
  assert.deepEqual(palette, [landform.rgb, block.rgb])
  const drawn = { west: -500, south: -500, east: 500, north: 500 }
  const near = rasterizeClassLayers(layers, drawn, { minCell: 1 })
  const far = rasterizeClassLayers(
    [
      {
        id: 'far',
        rgb: [0, 255, 0],
        polygons: [
          [
            [
              [600, 0],
              [700, 0],
              [700, 100],
              [600, 100],
              [600, 0]
            ]
          ]
        ]
      }
    ],
    { west: -2000, south: -2000, east: 2000, north: 2000 },
    { minCell: 4 }
  )
  const code = (x: number, y: number) => stemClassCode({ near, far }, drawn, palette, x, y)
  assert.equal(code(10.5, 10.5), 1) // on the landform
  assert.equal(code(50.5, 50.5), 2) // in the block, painted over the landform
  assert.equal(code(-20, 10), 0) // unlisted ground: screening
  assert.equal(code(650, 50), 2) // past the terrain grid, from the far drape
  assert.equal(code(650, 500), 0)
  assert.equal(stemClassCode({ near, far: null }, drawn, palette, 650, 50), 0)
})

test('a yaw carries its class code as whole turns without changing the rotation', () => {
  for (const yaw of [0, 1, -2.5, 6.283, -6.2831, 12.9])
    for (const code of [0, 1, 7, 31]) {
      const encoded = Math.fround(encodeStemClass(yaw, code)) // as the GPU attribute holds it
      assert.equal(stemClassOf(encoded), code)
      assert.ok(Math.abs(Math.cos(encoded) - Math.cos(yaw)) < 2e-3)
      assert.ok(Math.abs(Math.sin(encoded) - Math.sin(yaw)) < 2e-3)
    }
})

test('a picture drawn wider by a guard band crops back to exactly the frame', () => {
  const frame: CaptureFrame = {
    bearingDegrees: 30,
    pitchDegrees: 2,
    horizontalFovDegrees: 40,
    verticalFovDegrees: 12,
    width: 400,
    height: 120
  }
  const guard = 40
  const wide = frame.width + 2 * guard
  const pose = frameCamera([0, 0, 0], frame)
  const aspect = guardedAspect(pose.aspect, frame.width, guard)
  // The wide pinhole: same vertical field, horizontal half-angle tangent tanV · aspect.
  const tanV = Math.tan((frame.verticalFovDegrees * Math.PI) / 360)
  const wideFrame: CaptureFrame = {
    ...frame,
    width: wide,
    horizontalFovDegrees: (2 * Math.atan(tanV * aspect) * 180) / Math.PI
  }
  const r = rng(7)
  for (let n = 0; n < 50; n++) {
    const point = [r() * 2000 - 1000, r() * 2000 - 1000, r() * 100 - 50]
    const a = projectPinhole(frame, [0, 0, 0], point)
    const b = projectPinhole(wideFrame, [0, 0, 0], point)
    if (!a || !b) continue
    assert.ok(Math.abs(b[0] - guard - a[0]) < 1e-6)
    assert.ok(Math.abs(b[1] - a[1]) < 1e-6)
  }
  const rows = new Uint8Array(wide * 2 * 4).map((_, i) => i % 251)
  const cropped = cropColumns(rows, wide, 2, 4, guard, frame.width)
  assert.equal(cropped.length, frame.width * 2 * 4)
  assert.equal(cropped[0], rows[guard * 4])
  assert.equal(cropped[frame.width * 4], rows[(wide + guard) * 4])
})

/** A small scene package with every kind of value a real one carries (numbers, nulls, nested rings, strings, flags). */
const samplePackage = () => {
  const r = rng(11)
  const grid = (n: number) => ({
    west: -30,
    south: -30,
    spacing: 30,
    columns: n,
    rows: n,
    heights: Array.from({ length: n * n }, () => Math.round(r() * 20000) / 10),
    canopy: Array.from({ length: n * n }, () => Math.round(r() * 100) / 100),
    cover: Array.from({ length: n * n }, (_, i) => (i % 5 === 0 ? null : Math.floor(r() * 8))),
    burntYear: Array.from({ length: n * n }, (_, i) => (i % 7 === 0 ? 2024 : null))
  })
  return {
    format: 'pgmaps-forest-scene',
    version: 1,
    createdAt: '2026-11-15T09:22:40.059Z',
    origin: { lng: -116.96, lat: 50.25, elevationMeters: 558 },
    terrain: grid(9),
    farTerrain: grid(5),
    skylineTerrain: { ...grid(4), earthCurvature: { earthRadiusMeters: 6371000, refraction: 0.13 } },
    trees: {
      fields: ['x', 'y', 'z', 'species', 'height', 'crownWidth', 'tone'],
      species: ['pine', 'spruce'],
      count: 40,
      data: Array.from({ length: 280 }, () => Math.round(r() * 1000) / 10)
    },
    species: {
      pine: { crownRatio: 0.14, crownBase: 0.54, broadleaf: false },
      spruce: { crownBase: 0.3, broadleaf: false }
    },
    openings: [{ kind: 'harvested', rings: [[[-6, 2], [4, 2], [4, 9], [-6, 2]]], ageYears: 3, retainedShare: null }],
    debris: { fields: ['x', 'y', 'z', 'kind'], count: 0, data: [] },
    roadScars: [{ kind: 'fill', ring: [[-3, 1], [0, 1], [0, 2], [-3, 1]] }],
    water: [],
    fires: [{ fireNumber: 'N71650', year: 2024, yearsSinceFire: 2 }],
    camera: { position: [0, 0, 559.6], bearingDegrees: 90, pitchDegrees: -5.5, verticalFovDegrees: 36.9 },
    environment: { type: 'pgmaps-environment', dateTime: '2026-11-15T14:00:00-07:00', weather: 'clear' },
    treeRadiusMeters: 12000,
    notes: ['one', 'two']
  }
}

/** Every place of a value a number sits at (arrays and objects, any depth), as a path of keys. */
function numberPaths(value: unknown, path: Array<string | number> = [], out: Array<Array<string | number>> = []) {
  if (typeof value === 'number') out.push(path)
  else if (Array.isArray(value)) for (let i = 0; i < value.length; i++) numberPaths(value[i], [...path, i], out)
  else if (value && typeof value === 'object')
    for (const [k, v] of Object.entries(value)) numberPaths(v, [...path, k], out)
  return out
}
const setAt = (value: unknown, path: Array<string | number>, to: (was: number) => unknown) => {
  let at = value as Record<string | number, unknown>
  for (const key of path.slice(0, -1)) at = at[key] as Record<string | number, unknown>
  const last = path[path.length - 1]
  at[last] = to(at[last] as number)
}

test('a content hash is the same for a copy and different for any one changed value', () => {
  const pkg = samplePackage()
  const hash = contentHash(pkg)
  assert.match(hash, /^[0-9a-f]{16}$/)
  // A structured clone (what postMessage hands over) and a JSON round trip are other objects with the same content.
  assert.equal(contentHash(structuredClone(pkg)), hash)
  assert.equal(contentHash(JSON.parse(JSON.stringify(pkg))), hash)
  // Key order is not content.
  assert.equal(contentHash(Object.fromEntries(Object.entries(pkg).reverse())), hash)
  // Every number of the package, one at a time: nudged by the least it can be, zeroed, and made null.
  const paths = numberPaths(pkg)
  assert.ok(paths.length > 500)
  const seen = new Set([hash])
  for (const path of paths) {
    for (const change of [
      (v: number) => (v === 0 ? 5e-324 : v * (1 + 2 ** -52)),
      (v: number) => (v === 0 ? 1 : 0),
      () => null
    ]) {
      const other = structuredClone(pkg)
      setAt(other, path, change)
      const changed = contentHash(other)
      assert.notEqual(changed, hash, `a change at ${path.join('.')} went unseen`)
      seen.add(changed)
    }
  }
  // And no two of those changes hash alike.
  assert.equal(seen.size, 1 + paths.length * 3)
  // Strings, flags and a missing or an extra key.
  for (const change of [
    (p: ReturnType<typeof samplePackage>) => (p.trees.species[1] = 'fir'),
    (p: ReturnType<typeof samplePackage>) => (p.species.pine.broadleaf = true),
    (p: ReturnType<typeof samplePackage>) => (p.openings[0].kind = 'harvesteD'),
    (p: ReturnType<typeof samplePackage>) => delete (p as { water?: unknown }).water,
    (p: ReturnType<typeof samplePackage>) => ((p as { more?: unknown }).more = undefined),
    (p: ReturnType<typeof samplePackage>) => p.terrain.heights.push(0),
    (p: ReturnType<typeof samplePackage>) => p.terrain.cover.pop()
  ]) {
    const other = structuredClone(pkg)
    change(other)
    assert.notEqual(contentHash(other), hash)
  }
})

test('a content hash tells kinds, shapes and typed arrays apart, and reads every byte', () => {
  const distinct = [
    [1, 2, 3],
    [1, 2],
    [[1, 2], 3],
    [1, [2, 3]],
    [[1], [2, 3]],
    [[1, 2], [3]],
    [1, 2, null],
    [1, 2, 0],
    [1, 2, undefined],
    [0],
    [-0],
    [null],
    [false],
    [''],
    ['0'],
    [NaN],
    [],
    {},
    { a: [] },
    { a: {} },
    null,
    undefined,
    0,
    '0',
    false,
    new Float32Array([1, 2, 3]),
    new Float64Array([1, 2, 3]),
    new Int32Array([1, 2, 3]),
    new Uint8Array([1, 2, 3]),
    new Uint8Array([1, 2, 3, 0]),
    new Uint8Array([1, 2, 3]).buffer,
    new Date(5),
    { a: 1, b: 2 },
    { a: 2, b: 1 },
    { ab: 1 },
    { a: { b: 1 } }
  ]
  const hashes = distinct.map((v) => contentHash(v))
  assert.equal(new Set(hashes).size, distinct.length)
  for (let i = 0; i < distinct.length; i++) assert.equal(contentHash(structuredClone(distinct[i])), hashes[i])
  // A typed array: any one element, the last odd bytes, and where it starts in its buffer.
  const r = rng(3)
  const stand = Float32Array.from({ length: 7 * 300 }, () => r() * 1000)
  const hash = contentHash(stand)
  assert.equal(contentHash(stand.slice()), hash)
  for (let i = 0; i < stand.length; i++) {
    const other = stand.slice()
    other[i] = Math.fround(other[i] * (1 + 2 ** -23)) === other[i] ? other[i] + 1 : other[i] * (1 + 2 ** -23)
    assert.notEqual(contentHash(other), hash, `element ${i}`)
  }
  const bytes = Uint8Array.from({ length: 23 }, (_, i) => i * 7)
  for (let i = 0; i < bytes.length; i++) {
    const other = bytes.slice()
    other[i] ^= 1
    assert.notEqual(contentHash(other), contentHash(bytes), `byte ${i}`)
  }
  // The same values at an odd offset of a larger buffer hash as the same view on its own.
  const padded = new Uint8Array(3 + bytes.length)
  padded.set(bytes, 3)
  assert.equal(contentHash(new Uint8Array(padded.buffer, 3, bytes.length)), contentHash(bytes))
  // What it cannot read never equals anything, itself included.
  const map = new Map([[1, 2]])
  assert.notEqual(contentHash({ a: map }), contentHash({ a: map }))
  // Top-level keys can be left out, and only there.
  assert.equal(contentHash({ a: 1, b: 2 }, new Set(['b'])), contentHash({ a: 1, b: 3 }, new Set(['b'])))
  assert.notEqual(contentHash({ a: { b: 2 } }, new Set(['b'])), contentHash({ a: { b: 3 } }, new Set(['b'])))
})

test('a scene package builds the same scene when only what a build never reads has changed', () => {
  const pkg = samplePackage()
  const key = scenePackageKey(pkg, 2026)
  assert.equal(scenePackageKey(structuredClone(pkg), 2026), key)
  // PGMaps writes a new `createdAt` into every package, and its camera, environment, origin and notes are applied to
  // the built scene as they come.
  const later = structuredClone(pkg)
  later.createdAt = '2026-11-15T09:40:00.000Z'
  later.camera.bearingDegrees = 12
  later.environment.weather = 'overcast'
  later.origin.lat = 51
  later.notes.push('three')
  assert.equal(scenePackageKey(later, 2026), key)
  assert.deepEqual([...SCENE_UNBUILT_FIELDS].sort(), ['camera', 'createdAt', 'environment', 'notes', 'origin'])
  // The year the view is set in ages the fires: another year is another scene.
  assert.notEqual(scenePackageKey(pkg, 2027), key)
  assert.notEqual(scenePackageKey(pkg, null), key)
  // Anything a build reads, anywhere: every number outside those fields.
  for (const path of numberPaths(pkg)) {
    if (SCENE_UNBUILT_FIELDS.has(String(path[0]))) continue
    const other = structuredClone(pkg)
    setAt(other, path, (v) => v + 0.01)
    assert.notEqual(scenePackageKey(other, 2026), key, `a change at ${path.join('.')} kept the scene`)
  }
})

test('a capture codes stems from their own yaws and puts them back exactly', () => {
  const r = rng(5)
  // Scatter instances: x, z, yaw, width, height; yaws of any sign and size, as float32.
  const list = Float32Array.from({ length: 5 * 200 }, (_, i) => (i % 5 === 2 ? (r() - 0.5) * 40 : r() * 100))
  const before = list.slice()
  const yaws = takeYaws(list, 5, 2, 200)
  assert.equal(yaws.length, 200)
  const code = (n: number, round: number) => (n * 7 + round * 3) % 31
  const coded = (round: number) => {
    for (let n = 0; n < 200; n++) list[n * 5 + 2] = encodeStemClass(yaws[n], code(n, round))
    return list.slice()
  }
  const first = coded(0)
  for (let n = 0; n < 200; n++) {
    assert.equal(stemClassOf(first[n * 5 + 2]), code(n, 0))
    // Nothing but the yaw is touched.
    for (const k of [0, 1, 3, 4]) assert.equal(first[n * 5 + k], before[n * 5 + k])
  }
  // Another capture's classes in between leave no trace: the same classes code to the same floats again.
  coded(1)
  assert.deepEqual(coded(0), first)
  // Coding a yaw that was already coded would not: under whole turns a float32 loses its last bits.
  let drifted = 0
  for (let n = 0; n < 200; n++) if (encodeStemClass(first[n * 5 + 2], 0) !== encodeStemClass(yaws[n], 0)) drifted++
  assert.ok(drifted > 0)
  putYaws(list, 5, 2, yaws)
  assert.deepEqual(list, before)
})

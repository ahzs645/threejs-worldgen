import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  alphaCoverage,
  bakePatch,
  canopyCutoffMeters,
  CARD_ASPECT,
  cappedWidthScale,
  coverageMips,
  coverageScale,
  coxMunk,
  elevationFrame,
  finalStandKey,
  inBearings,
  inStandSector,
  jitterOffsets,
  linearToSrgbByte,
  linearToSrgbBytes,
  meanSrgbBytes,
  MAX_STRETCH,
  median,
  patchSpan,
  patchStems,
  PATCH_RADIUS,
  rasterTree,
  reflectionBlurPx,
  SILHOUETTE_CROWN_RATIO,
  SNAG_SLOT,
  SPECIES_ORDER,
  SRGB_TO_LINEAR,
  syntheticStand
} from './pgmaps-far-forest-math.ts'

test('the dissolve distance is where a tree is under the pixel count tall', () => {
  // 1200 × 582 frame, 19.8° vertical field: f = 291 / tan(9.9°) ≈ 1668 px; a 22 m tree is 4 px at 9.2 km.
  const d = canopyCutoffMeters(22, 582, 19.8, 4)
  assert.ok(Math.abs(d - (22 * (291 / Math.tan((9.9 * Math.PI) / 180))) / 4) < 1e-6)
  assert.ok(d > 9000 && d < 9300)
  // Twice the pixels: twice as far; a smaller threshold: farther.
  assert.ok(Math.abs(canopyCutoffMeters(22, 1164, 19.8, 4) - 2 * d) < 1e-6)
  assert.ok(canopyCutoffMeters(22, 582, 19.8, 2) > d)
  assert.equal(canopyCutoffMeters(0, 582, 19.8), Infinity)
  assert.equal(median([3, 1, 2]), 2)
  assert.equal(median([4, 1, 2, 3]), 2.5)
  assert.ok(Number.isNaN(median([])))
})

test('a stem is never stretched sideways past 1.5 times its shape', () => {
  assert.equal(cappedWidthScale(10, 1), MAX_STRETCH)
  assert.equal(cappedWidthScale(0.1, 1), 1 / MAX_STRETCH)
  assert.equal(cappedWidthScale(1.2, 1), 1.2)
  // The far stems' squat blobs: 55 m crowns on a grown 6 m width, 22 m tall on a grown 22 m height.
  assert.ok(cappedWidthScale(55 / 6, 1) / 1 <= 1.5)
})

test('coverage-preserving mips keep an alpha-tested tree card’s coverage at every level', () => {
  // A row of spruce cards, the needles' ragged edges a texel or two wide: box filtering thins their alpha, so a plain
  // mip loses coverage under a 0.5 alpha test (the far forest's speckle and thinning); the scaled one keeps it.
  const tree = rasterTree('spruce', 16, 64, 0.4, 5, 2)
  const w = 64
  const h = 64
  const data = new Uint8Array(w * h * 4)
  for (let k = 0; k < 4; k++) for (let y = 0; y < h; y++) data.set(tree.data.subarray(y * 16 * 4, (y + 1) * 16 * 4), (y * w + k * 16) * 4)
  const alphaOf = (level: { data: Uint8Array; width: number; height: number }) =>
    Uint8Array.from({ length: level.width * level.height }, (_, i) => level.data[i * 4 + 3])
  const base = alphaCoverage(alphaOf({ data, width: w, height: h }), 0.5)
  const levels = coverageMips({ data, width: w, height: h }, w, h, 0.5, 4)
  assert.equal(levels.length, 5) // 64, 32, 16, 8, 4
  // A plain box filter of the same atlas, for comparison.
  let plain = { data, width: w, height: h }
  for (const level of levels.slice(1)) {
    const half = new Uint8Array(level.width * level.height * 4)
    for (let y = 0; y < level.height; y++)
      for (let x = 0; x < level.width; x++) {
        let a = 0
        for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) a += plain.data[((2 * y + dy) * plain.width + 2 * x + dx) * 4 + 3]
        half[(y * level.width + x) * 4 + 3] = Math.round(a / 4)
      }
    plain = { data: half, width: level.width, height: level.height }
    const coverage = alphaCoverage(alphaOf(level), 0.5)
    const plainCoverage = alphaCoverage(alphaOf(plain), 0.5)
    assert.ok(Math.abs(coverage - base) <= Math.abs(plainCoverage - base) + 1e-9, `${level.width}: ${coverage} vs plain ${plainCoverage}, base ${base}`)
    assert.ok(Math.abs(coverage - base) < 0.06, `${level.width}: ${coverage} vs ${base}`)
  }
  assert.ok(coverageScale([100, 100, 0, 0], 0.5, 0.5) > 1.27)
  assert.equal(coverageScale([255, 255, 255, 255], 0.5, 1), 1, 'opaque stays opaque')
})

test('cells of an atlas are reduced on their own', () => {
  // Two 4 × 4 cells: the left opaque, the right empty. No level lets the left bleed into the right.
  const data = new Uint8Array(8 * 4 * 4)
  for (let y = 0; y < 4; y++)
    for (let x = 0; x < 4; x++) {
      const o = (y * 8 + x) * 4
      data.set([255, 0, 0, 255], o)
    }
  const levels = coverageMips({ data, width: 8, height: 4 }, 4, 4, 0.5, 1)
  const last = levels[levels.length - 1]
  assert.equal(last.width, 2)
  assert.equal(last.data[3], 255)
  assert.equal(last.data[7], 0)
})

test('tree silhouettes stand taller than wide and are dark under their whorls', () => {
  for (const species of ['spruce', 'fir', 'pine', 'aspen'] as const) {
    const tree = rasterTree(species, 32, 128, 0.4, 1)
    let cover = 0
    let minX = 99
    let maxX = -1
    let minY = 999
    let maxY = -1
    for (let y = 0; y < 128; y++)
      for (let x = 0; x < 32; x++)
        if (tree.data[(y * 32 + x) * 4 + 3] > 127) {
          cover++
          minX = Math.min(minX, x)
          maxX = Math.max(maxX, x)
          minY = Math.min(minY, y)
          maxY = Math.max(maxY, y)
        }
    assert.ok(cover > 40, species)
    const widthM = ((maxX - minX + 1) / 32) * 0.4
    const heightM = (maxY - minY + 1) / 128
    assert.ok(heightM / widthM > 2, `${species}: ${heightM / widthM}`)
    assert.ok(widthM <= SILHOUETTE_CROWN_RATIO[species] * 1.35, `${species} width ${widthM}`)
  }
})

test('a patch is baked at its closure and spans its foreshortened depth and its trees', () => {
  const stems = patchStems('spruce', 0.72, 3)
  const crownArea = stems.reduce((s, t) => s + Math.PI * (t.crown / 2) ** 2, 0)
  const closure = crownArea / (Math.PI * PATCH_RADIUS ** 2)
  assert.ok(closure > 0.55 && closure < 1.0, `closure ${closure}`)
  assert.ok(stems.every((t) => Math.hypot(t.x, t.y) <= PATCH_RADIUS))
  assert.deepEqual(patchStems('spruce', 0.72, 3), stems, 'deterministic')
  assert.equal(patchSpan(0), 22)
  // Ground 2R back on a 30° rise stands 2R · tan 30° up; the trees on it at full height.
  assert.ok(Math.abs(patchSpan(30) - (2 * PATCH_RADIUS * Math.tan(Math.PI / 6) + 22)) < 1e-9)
  assert.equal(patchSpan(80), patchSpan(60), 'capped at 60°')
  // From level, the crowns close into a band; from above, the gaps between them show.
  const cover = (elevation: number) => {
    const sprites = [rasterTree('spruce', 16, 64, CARD_ASPECT, 1, 2), rasterTree('spruce', 16, 64, CARD_ASPECT, 2, 2)]
    const r = bakePatch(stems, 'spruce', sprites, elevation, 64, 64)
    let n = 0
    for (let i = 0; i < 64 * 64; i++) if (r.data[i * 4 + 3] > 127) n++
    return n / (64 * 64)
  }
  const level = cover(0)
  const above = cover(65)
  assert.ok(level > 0.3, `level ${level}`)
  assert.ok(above < level + 0.2)
  assert.deepEqual(elevationFrame(0), { index: 0, next: 1, t: 0 })
  assert.equal(elevationFrame(17).index, 3)
  assert.equal(elevationFrame(90).index, 6)
  assert.equal(elevationFrame(36.5).index, 5)
})

test('wind roughens a lake by Cox–Munk, scaled for its fetch; calm water is a mirror', () => {
  assert.equal(coxMunk(0).roughness, 0)
  const five = coxMunk(5)
  assert.ok(Math.abs(five.slopeVariance - 0.25 * (0.003 + 0.00512 * 5)) < 1e-12)
  assert.ok(coxMunk(10).roughness > five.roughness)
  assert.ok(Math.abs(coxMunk(5, 1).slopeVariance - (0.003 + 0.0256)) < 1e-12, 'open sea')
  const blur = reflectionBlurPx(five.slopeVariance, 1668)
  assert.ok(Math.abs(blur.vertical - 2 * five.roughness * 1668) < 1e-9)
  assert.ok(blur.horizontal < blur.vertical)
  assert.equal(reflectionBlurPx(0, 1668).vertical, 0)
})

test('the final stand’s sector wraps past north', () => {
  assert.ok(inBearings(355, [350, 380]))
  assert.ok(inBearings(10, [350, 380]))
  assert.ok(!inBearings(30, [350, 380]))
  assert.ok(inBearings(200, [0, 360]))
  const eye = [100, 100]
  const stand = { radiusMeters: 1000, bearings: [80, 100] as [number, number] }
  assert.ok(inStandSector(eye, stand, 600, 100)) // due east
  assert.ok(!inStandSector(eye, stand, 100, 600)) // due north
  assert.ok(!inStandSector(eye, stand, 1200, 100)) // too far
})

test('a synthetic stand is real density where the ground is forest and nothing elsewhere', () => {
  const stand = syntheticStand(
    [0, 0, 0],
    [0, 90],
    1000,
    (x) => (x > 500 ? 0 : 0.5),
    () => 100,
    [
      { species: 'spruce', share: 3, height: 24 },
      { species: 'pine', share: 1, height: 20 }
    ],
    { density: 800 }
  )
  assert.equal(stand.data.length, stand.count * 7)
  // About 400 stems/ha over the forested part of the quarter disc (x ≤ 500).
  let east = 0
  let pine = 0
  for (let i = 0; i < stand.count; i++) {
    const x = stand.data[i * 7]
    const y = stand.data[i * 7 + 1]
    assert.ok(x >= 0 && y >= 0 && Math.hypot(x, y) <= 1000)
    if (x > 550) east++
    if (stand.data[i * 7 + 3] === 1) pine++
    assert.equal(stand.data[i * 7 + 2], 100)
  }
  assert.ok(east < stand.count * 0.02, 'none on open ground')
  assert.ok(stand.count > 1000)
  assert.ok(pine / stand.count > 0.18 && pine / stand.count < 0.32)
})

test('jitter samples spread over the pixel and average to its centre', () => {
  const j = jitterOffsets(32)
  assert.equal(j.length, 32)
  assert.deepEqual(j[0], [0, 0])
  const mean = j.reduce((s, [x, y]) => [s[0] + x, s[1] + y], [0, 0]).map((v) => v / j.length)
  assert.ok(Math.abs(mean[0]) < 0.05 && Math.abs(mean[1]) < 0.05)
  assert.ok(j.every(([x, y]) => Math.abs(x) <= 0.5 && Math.abs(y) <= 0.5))
  for (const b of [0, 1, 54, 128, 255]) assert.equal(linearToSrgbByte(SRGB_TO_LINEAR[b]), b)
})

test('a snag card is a thin trunk with stubs: it covers a fraction of a living tree, so it screens less', () => {
  assert.equal(SNAG_SLOT, SPECIES_ORDER.length)
  const cover = (r: { data: Uint8Array }) => {
    let sum = 0
    for (let i = 3; i < r.data.length; i += 4) sum += r.data[i] / 255
    return sum
  }
  // The atlas cells: 64 × 128 at the card aspect.
  for (const seed of [1, 8, 15]) {
    const snag = cover(rasterTree('snag', 64, 128, CARD_ASPECT, seed, 3))
    const spruce = cover(rasterTree('spruce', 64, 128, CARD_ASPECT, seed, 3))
    const pine = cover(rasterTree('pine', 64, 128, CARD_ASPECT, seed, 3))
    assert.ok(snag > 0, 'a snag is drawn')
    // About a quarter of a spruce's and a third of a lodgepole's (the narrowest crown), bold as the trunk is drawn.
    const said = `snag ${snag.toFixed(0)} vs spruce ${spruce.toFixed(0)}, pine ${pine.toFixed(0)}`
    assert.ok(snag < 0.3 * spruce && snag < 0.4 * pine, said)
  }
  // Its trunk reaches the ground and its top: some pixel is drawn in the bottom and the top tenth of the card.
  const snag = rasterTree('snag', 64, 128, CARD_ASPECT, 3, 3)
  const rowCover = (y: number) => {
    let sum = 0
    for (let x = 0; x < 64; x++) sum += snag.data[(y * 64 + x) * 4 + 3]
    return sum
  }
  assert.ok(rowCover(127) > 0)
  let topDrawn = false
  for (let y = 0; y < 18; y++) topDrawn ||= rowCover(y) > 0
  assert.ok(topDrawn)
})

test('a burnt stand patch stands as many snags as the living stand stood trees', () => {
  const snags = patchStems('snag', 0.4, 7)
  const spruce = patchStems('spruce', 0.4, 7)
  assert.equal(snags.length, spruce.length)
  // Each snag's card is its stubs' reach, narrower than the spruce's crown.
  assert.ok(snags[0].crown < spruce[0].crown)
  assert.ok(SILHOUETTE_CROWN_RATIO.snag < SILHOUETTE_CROWN_RATIO.spruce)
})

test('a stand over its stem cap is thinned evenly, never cut off at the rows read last', () => {
  const ground = () => 0
  const mix = [{ species: 'spruce', share: 1, height: 20 }]
  const full = syntheticStand([0, 0, 0], [0, 360], 1000, () => 0.5, ground, mix, { density: 400, seed: 3 })
  const capped = syntheticStand([0, 0, 0], [0, 360], 1000, () => 0.5, ground, mix, {
    density: 400,
    seed: 3,
    maxStems: Math.round(full.count / 4)
  })
  assert.ok(capped.count <= Math.round(full.count / 4) && capped.count > full.count / 5)
  let north = 0
  let south = 0
  for (let i = 0; i < capped.count; i++) (capped.data[i * 7 + 1] > 0 ? north++ : south++)
  assert.ok(north > 0.4 * capped.count && south > 0.4 * capped.count, `north ${north}, south ${south}`)
})

test('a placed final stand is kept for the same stems from the same eye, and for nothing else', () => {
  const ground = () => 600
  const mix = [
    { species: 'spruce', share: 3, height: 24 },
    { species: 'pine', share: 1, height: 20 }
  ]
  const stand = syntheticStand([0, 0, 0], [340, 380], 700, () => 0.5, ground, mix, { density: 40, seed: 9 })
  assert.ok(stand.count > 100)
  const eye = [0, 0, 559.6]
  const key = finalStandKey(stand, eye, 600)
  // PGMaps sends the stand with every capture, a fresh buffer each time; its radius and bearings place nothing.
  const again = { ...stand, data: stand.data.slice(), radiusMeters: 900, bearings: [0, 360] as [number, number] }
  assert.equal(finalStandKey(again, [0, 0, 559.6], 600), key)
  // The eye's height is not read (stems are placed in plan); its place and where the stand starts are.
  assert.equal(finalStandKey(stand, [0, 0, 700], 600), key)
  assert.notEqual(finalStandKey(stand, [0.5, 0, 559.6], 600), key)
  assert.notEqual(finalStandKey(stand, [0, -0.5, 559.6], 600), key)
  assert.notEqual(finalStandKey(stand, eye, 601), key)
  // Any one value of any one stem.
  for (let i = 0; i < stand.data.length; i++) {
    const data = stand.data.slice()
    data[i] += data[i] === 0 ? 1 : Math.abs(data[i]) * 2 ** -20
    assert.notEqual(data[i], stand.data[i])
    assert.notEqual(finalStandKey({ ...stand, data }, eye, 600), key, `value ${i}`)
  }
  // Fewer stems read, another species list, another field order.
  assert.notEqual(finalStandKey({ ...stand, count: stand.count - 1 }, eye, 600), key)
  assert.notEqual(finalStandKey({ ...stand, species: ['spruce', 'fir'] }, eye, 600), key)
  assert.notEqual(finalStandKey({ ...stand, fields: [...stand.fields].reverse() }, eye, 600), key)
  // The same numbers as a plain array are not the same message, and are placed again.
  assert.notEqual(finalStandKey({ ...stand, data: Array.from(stand.data) }, eye, 600), key)
})

test('the table of sRGB steps gives the very bytes of the formula, at every step and between', () => {
  const f64 = new Float64Array(1)
  const bits = new BigUint64Array(f64.buffer)
  /** The double `k` places after `v`. */
  const after = (v: number, k: number) => {
    f64[0] = v
    bits[0] += BigInt(k)
    return f64[0]
  }
  const values: number[] = [NaN, -1, -0, 0, 5e-324, 1e-9, 0.0031308, 0.00313080001, 0.5, 1 - 2 ** -53, 1, 1.5]
  values.push(Infinity, -Infinity)
  // Either side of every step of the formula (found here by bisection, not from the table).
  for (let b = 1; b <= 255; b++) {
    let below = 0
    let at = 1
    for (let k = 0; k < 80; k++) {
      const mid = (below + at) / 2
      if (linearToSrgbByte(mid) >= b) at = mid
      else below = mid
    }
    for (let k = -40; k <= 40; k++) values.push(after(at, k))
  }
  let seed = 12345
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
    return seed / 4294967296
  }
  for (let i = 0; i < 200000; i++) values.push(random() * 1.1 - 0.05, random() ** 5, Math.fround(random() * 24) / 24)
  const bytes = linearToSrgbBytes(values)
  for (let i = 0; i < values.length; i++)
    assert.equal(bytes[i], linearToSrgbByte(values[i]) || 0, `at ${values[i]}`)
})

test('samples added up in linear light average to the mean picture, opaque', () => {
  // Three samples of two pixels; the sum's alpha (3, one a sample) is not read.
  const samples = [
    [10, 128, 255, 40, 41, 42],
    [12, 128, 0, 40, 43, 42],
    [11, 130, 255, 41, 42, 44]
  ]
  const sum = new Float32Array(8)
  for (const s of samples)
    for (let p = 0; p < 2; p++) {
      for (let c = 0; c < 3; c++) sum[p * 4 + c] += SRGB_TO_LINEAR[s[p * 3 + c]]
      sum[p * 4 + 3] += 1
    }
  const picture = meanSrgbBytes(sum, 3)
  assert.equal(picture.length, 8)
  for (let p = 0; p < 2; p++) {
    for (let c = 0; c < 3; c++) assert.equal(picture[p * 4 + c], linearToSrgbByte(sum[p * 4 + c] / 3))
    assert.equal(picture[p * 4 + 3], 255)
  }
  // The mean is taken in linear light: black and white average to 188, not 128.
  assert.deepEqual([...meanSrgbBytes(Float32Array.of(1, 0.5, 0, 9), 2)], [188, 137, 0, 255])
  // One sample is that sample, to the byte, for every byte.
  const one = new Float32Array(256 * 4)
  for (let v = 0; v < 256; v++) one.fill(SRGB_TO_LINEAR[v], v * 4, v * 4 + 3)
  const back = meanSrgbBytes(one, 1)
  for (let v = 0; v < 256; v++) assert.deepEqual([...back.subarray(v * 4, v * 4 + 4)], [v, v, v, 255])
})

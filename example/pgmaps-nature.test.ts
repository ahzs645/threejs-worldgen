import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  burntShare,
  canopyClosure,
  COVER_CLASSES,
  CLASS_CLOSURE_CAP,
  closureCap,
  coverClassSampler,
  coverCounts,
  keepStem,
  natureWeights,
  onBareGround,
  onBurntGround,
  rockStands,
  openShare,
  SCENE_COVER,
  stemHash,
  thinShare
} from './pgmaps-nature.ts'

const { forest, sparse, shrub, herb, rock, water, cut } = SCENE_COVER
const grid = (cover: Array<number | null> | undefined, west = -10, spacing = 10) => ({
  west,
  south: west,
  spacing,
  columns: 3,
  rows: 3,
  cover
})
const close = (a: number, b: number) => Math.abs(a - b) < 1e-6

test('the codes are PGMaps SCENE_COVER', () => {
  assert.deepEqual(SCENE_COVER, { forest: 0, sparse: 1, shrub: 2, herb: 3, rock: 4, water: 5, cut: 6, burnt: 7 })
  assert.equal(CLASS_CLOSURE_CAP.length, 8)
})

test('cover weights blend bilinearly over a cell and are exact at a post', () => {
  // South row forest, middle row rock, north row herb.
  const at = coverClassSampler([grid([forest, forest, forest, rock, rock, rock, herb, herb, herb])])
  assert.ok(at)
  const w = at(0, 0)!
  assert.equal(w[rock], 1)
  const half = Array.from(at(0, -5)!)
  assert.ok(close(half[rock], 0.5) && close(half[forest], 0.5))
  const quarter = Array.from(at(3, 7.5)!)
  assert.ok(close(quarter[herb], 0.75) && close(quarter[rock], 0.25))
  assert.equal(at(40, 0), null)
})

test('unknown posts drop out; a cell of unknowns is unknown', () => {
  const at = coverClassSampler([grid([null, null, null, null, rock, null, null, null, null])])!
  assert.equal(at(2, 2)![rock], 1)
  const none = coverClassSampler([grid(new Array(9).fill(null))])!
  assert.equal(none(0, 0), null)
  // Codes outside the table are unknown too.
  const odd = coverClassSampler([grid(new Array(9).fill(9))])!
  assert.equal(odd(0, 0), null)
  assert.equal(coverClassSampler([grid(undefined), null, undefined]), null)
  assert.equal(coverClassSampler([grid([1, 2])]), null)
})

test('the near grid answers first, the coarser ones past it', () => {
  const near = grid(new Array(9).fill(shrub))
  const far = grid(new Array(9).fill(rock), -1000, 1000)
  const at = coverClassSampler([near, far])!
  assert.equal(at(0, 0)![shrub], 1)
  assert.equal(at(500, 500)![rock], 1)
  // A near grid that knows nothing there hands the point on.
  const blank = coverClassSampler([grid(new Array(9).fill(null)), far])!
  assert.equal(blank(0, 0)![rock], 1)
})

test('closure caps: none on rock and water, scattered trees on herb and shrub, a full stand on forest and cut', () => {
  const one = (code: number) => {
    const w = new Float32Array(COVER_CLASSES)
    w[code] = 1
    return w
  }
  assert.equal(closureCap(one(rock)), 0)
  assert.equal(closureCap(one(water)), 0)
  assert.equal(closureCap(one(forest)), 1)
  assert.equal(closureCap(one(cut)), 1)
  assert.equal(closureCap(null), 1)
  assert.ok(closureCap(one(herb)) < closureCap(one(shrub)))
  assert.ok(closureCap(one(shrub)) < closureCap(one(sparse)))
  assert.ok(closureCap(one(sparse)) < 1)
  const edge = new Float32Array(COVER_CLASSES)
  edge[rock] = 0.5
  edge[forest] = 0.5
  assert.equal(closureCap(edge), 0.5)
})

test('thinning keeps a stand already under its cap and drops every stem on rock', () => {
  assert.equal(thinShare(0.2, 0.3), 1)
  assert.equal(thinShare(0.6, 0.3), 0.5)
  assert.equal(thinShare(0.6, 0), 0)
  assert.equal(thinShare(Number.NaN, 0.3), 1)
  let kept = 0
  for (let i = 0; i < 4000; i++) if (keepStem(i * 3.7, i * -1.3, 0.25)) kept++
  assert.ok(Math.abs(kept / 4000 - 0.25) < 0.03, String(kept))
  for (let i = 0; i < 100; i++) assert.equal(keepStem(i, i, 0), false)
  for (let i = 0; i < 100; i++) assert.equal(keepStem(i, i, 1), true)
  assert.equal(stemHash(12.5, -3), stemHash(12.5, -3))
  for (let i = 0; i < 100; i++) {
    const h = stemHash(i * 7.1, i * 2.3)
    assert.ok(h >= 0 && h < 1)
  }
})

test('the canopy surface never stands on rock, herb or shrub ground', () => {
  const w = new Float32Array(COVER_CLASSES)
  for (const code of [rock, herb, shrub, water]) {
    w.fill(0)
    w[code] = 1
    assert.equal(openShare(w), 1)
    assert.equal(canopyClosure(0.8, w), 0)
  }
  w.fill(0)
  w[sparse] = 1
  assert.equal(canopyClosure(0.8, w), CLASS_CLOSURE_CAP[sparse])
  w.fill(0)
  w[forest] = 1
  assert.equal(canopyClosure(0.8, w), 0.8)
  assert.equal(canopyClosure(0.8, null), 0.8)
  // A blend: mostly forest stands, mostly rock does not.
  w.fill(0)
  w[forest] = 0.7
  w[rock] = 0.3
  assert.ok(canopyClosure(0.8, w) > 0.5)
  w[forest] = 0.3
  w[rock] = 0.7
  assert.equal(canopyClosure(0.8, w), 0)
})

test('the ground material weights carry rock, herb, shrub and sparse', () => {
  const w = new Float32Array(COVER_CLASSES)
  w[rock] = 0.25
  w[herb] = 0.25
  w[shrub] = 0.125
  w[sparse] = 0.125
  w[forest] = 0.25
  const out = new Float32Array(8)
  natureWeights(w, out, 4)
  assert.deepEqual(Array.from(out), [0, 0, 0, 0, 0.25, 0.25, 0.125, 0.125])
  natureWeights(null, out, 0)
  assert.deepEqual(Array.from(out.subarray(0, 4)), [0, 0, 0, 0])
  assert.deepEqual(coverCounts(grid([0, 4, 4, null, 5, 3, 2, 1, 9])), { codes: [1, 1, 1, 1, 2, 1, 0, 0], unknown: 2 })
  assert.equal(coverCounts(grid(undefined)), null)
})

test('no package stem on rock or water: half or more of the blended cover', async () => {
  const { bareShare, onBareGround } = await import('./pgmaps-nature.ts')
  const w = new Float32Array(COVER_CLASSES)
  w[rock] = 0.3
  w[water] = 0.3
  w[forest] = 0.4
  assert.ok(close(bareShare(w), 0.6))
  assert.equal(onBareGround(w), true)
  w[water] = 0
  w[forest] = 0.7
  assert.equal(onBareGround(w), false)
  assert.equal(onBareGround(null), false)
})

test('skyline sea posts lie at the dropped sea level; lakes inland keep their height', async () => {
  const { droppedSeaLevel, skylineWaterHeight, SEA_SNAP_M } = await import('./pgmaps-nature.ts')
  const curvature = { earthRadiusMeters: 6_371_000, refraction: 0.13 }
  const level = droppedSeaLevel(30_000, curvature)
  // About 61 m at 30 km.
  assert.ok(Math.abs(level + (30_000 ** 2 / (2 * 6_371_000)) * 0.87) < 1e-9)
  assert.ok(level < -60 && level > -62)
  assert.equal(droppedSeaLevel(0, curvature), -0)
  assert.equal(skylineWaterHeight(null, level), level)
  assert.equal(skylineWaterHeight(level + 3, level), level)
  assert.equal(skylineWaterHeight(level - 4, level), level)
  assert.equal(skylineWaterHeight(level + SEA_SNAP_M + 20, level), level + SEA_SNAP_M + 20)
})

test('burnt ground grows no green stand and no canopy surface, and its stems are snags', () => {
  const { burnt } = SCENE_COVER
  const at = coverClassSampler([grid([burnt, burnt, burnt, burnt, forest, forest, forest, forest, forest])])!
  const w = at(-10, -10)!
  assert.equal(burntShare(w), 1)
  assert.ok(onBurntGround(w))
  assert.equal(closureCap(w), 0)
  assert.equal(openShare(w), 1)
  assert.equal(canopyClosure(0.6, w), 0)
  // Natural ground weights leave burnt to its own attribute.
  const out = new Float32Array(4)
  natureWeights(w, out, 0)
  assert.deepEqual(Array.from(out), [0, 0, 0, 0])
  // Half a cell into the forest, a stem is no longer on burnt ground.
  const edge = at(0, 5)!
  assert.ok(!onBurntGround(edge))
  assert.equal(burntShare(null), 0)
})

test('rock carrying a stand (the package closure over it) keeps its stems and its canopy', () => {
  const w = new Float32Array(COVER_CLASSES)
  w[rock] = 1
  assert.ok(!rockStands(0) && !rockStands(Number.NaN) && rockStands(0.4))
  // Bare rock: no stems, no canopy.
  assert.ok(onBareGround(w))
  assert.equal(closureCap(w), 0)
  assert.equal(canopyClosure(0.5, w), 0)
  // A stand on rock: its stems stand, its own closure holds, the canopy surface stands on it.
  assert.ok(!onBareGround(w, true))
  assert.equal(closureCap(w, true), 1)
  assert.equal(canopyClosure(0.5, w, true), 0.5)
  // Water stays bare whatever the closure says.
  const wet = new Float32Array(COVER_CLASSES)
  wet[water] = 1
  assert.ok(onBareGround(wet, true))
})

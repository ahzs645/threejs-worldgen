import { test } from 'node:test'
import assert from 'node:assert/strict'
import { binCrownWidth, binSeed, heightBin, openingSeed, pileSeed, TREE_HEIGHT_LADDER } from './pgmaps-tree-bins.ts'

/** A seeded generator for the fixtures (mulberry32). */
function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const SPECIES = ['pine', 'spruce', 'fir', 'douglasFir', 'aspen', 'birch']
const RATIOS: Record<string, number> = { pine: 0.14, spruce: 0.17, fir: 0.12, douglasFir: 0.2, aspen: 0.22 }

type Stem = { id: number; species: string; height: number; width: number }
function stand(n: number, seed: number): Stem[] {
  const rand = rng(seed)
  return Array.from({ length: n }, (_, id) => {
    const height = 3 + rand() ** 0.7 * 40
    return { id, species: SPECIES[Math.floor(rand() * SPECIES.length)], height, width: height * (0.15 + rand() * 0.3) }
  })
}
const classOf = (s: Stem) => {
  const bin = heightBin(s.height)
  return { bin, width: binCrownWidth(s.species, bin, RATIOS[s.species]), seed: binSeed(s.species, bin) }
}

test('the ladder rises about ×1.4 a step', () => {
  for (let i = 1; i < TREE_HEIGHT_LADDER.length; i++) {
    const r = TREE_HEIGHT_LADDER[i] / TREE_HEIGHT_LADDER[i - 1]
    assert.ok(r > 1.3 && r < 1.55, `step ${i}: ×${r.toFixed(2)}`)
  }
})

test('a stem goes to the class nearest its height in log terms', () => {
  assert.equal(TREE_HEIGHT_LADDER[heightBin(23.3)], 22)
  assert.equal(TREE_HEIGHT_LADDER[heightBin(13.8)], 16) // 13.8/11 = 1.25 > 16/13.8 = 1.16
  assert.equal(TREE_HEIGHT_LADDER[heightBin(1)], 4)
  assert.equal(TREE_HEIGHT_LADDER[heightBin(500)], 62)
  assert.equal(heightBin(Number.NaN), 0)
  for (const s of stand(2000, 3)) {
    const scale = s.height / TREE_HEIGHT_LADDER[heightBin(s.height)]
    if (s.height >= 4 && s.height <= 62) assert.ok(scale > 0.8 && scale < 1.25, `${s.height} scaled ×${scale}`)
  }
})

test('a class width reads the species ratio, else the default table, never other stems', () => {
  const bin = TREE_HEIGHT_LADDER.indexOf(22)
  assert.equal(binCrownWidth('pine', bin, 0.14), 4.5) // 0.14 × 1.5 × 22 = 4.62
  assert.equal(binCrownWidth('pine', bin, null), 4.5)
  assert.equal(binCrownWidth('pine', bin, 0.2), binCrownWidth('douglasFir', bin))
  assert.equal(binCrownWidth('unknown', 0), 1)
})

test('a harvest leaves every retained stem in the same class, width and seed', () => {
  const before = stand(5000, 11)
  const classes = new Map(before.map((s) => [s.id, classOf(s)]))
  const rand = rng(42)
  // Remove a random 40% and shuffle what is left, as a later package might list it.
  const after = before.filter(() => rand() >= 0.4)
  for (let i = after.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[after[i], after[j]] = [after[j], after[i]]
  }
  assert.ok(after.length > 2500 && after.length < 3500)
  for (const s of after) assert.deepEqual(classOf(s), classes.get(s.id), `stem ${s.id}`)
})

test('class seeds depend on species and class only, and differ between them', () => {
  const seeds = new Set<number>()
  for (const sp of SPECIES)
    for (let b = 0; b < TREE_HEIGHT_LADDER.length; b++) {
      assert.equal(binSeed(sp, b), binSeed(sp, b))
      seeds.add(binSeed(sp, b))
    }
  assert.equal(seeds.size, SPECIES.length * TREE_HEIGHT_LADDER.length)
})

test('an opening seeds from its own geometry: reordering the list or the ring changes nothing', () => {
  const square = (x: number, y: number, s: number): [number, number][] => [
    [x, y],
    [x + s, y],
    [x + s, y + s],
    [x, y + s]
  ]
  const openings = [[square(0, 0, 100)], [square(250, -40, 60)], [square(-300, 500, 220)]]
  const seeds = openings.map(openingSeed)
  assert.equal(new Set(seeds).size, 3)
  const reordered = [openings[2], openings[0], openings[1]]
  assert.deepEqual(reordered.map(openingSeed), [seeds[2], seeds[0], seeds[1]])
  // The same ring starting at another vertex.
  const ring = openings[1][0]
  assert.equal(openingSeed([[...ring.slice(2), ...ring.slice(0, 2)]]), seeds[1])
})

test('a debris pile seeds from its own position, whatever its index', () => {
  assert.equal(pileSeed(12.345, -6.7, 4), pileSeed(12.345, -6.7, 4))
  assert.notEqual(pileSeed(12.345, -6.7, 4), pileSeed(12.355, -6.7, 4))
})

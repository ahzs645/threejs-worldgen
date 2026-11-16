import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  canopySampler,
  coverSampler,
  cutAge,
  DEFAULT_CUT_AGE_YEARS,
  greenShare,
  retainedShare,
  srgbToLinear,
  stemCoverSampler,
  VIA_CUT_LINEAR,
  VIA_CUT_SRGB,
  viaFreshCut
} from './pgmaps-cover.ts'

const grid = (canopy: Array<number | null> | undefined, west = -10, spacing = 10) => ({
  west,
  south: west,
  spacing,
  columns: 3,
  rows: 3,
  canopy
})

test('an opening without a date is drawn at the default age, never negative', () => {
  assert.equal(cutAge(undefined), DEFAULT_CUT_AGE_YEARS)
  assert.equal(cutAge(null), DEFAULT_CUT_AGE_YEARS)
  assert.equal(cutAge(Number.NaN), DEFAULT_CUT_AGE_YEARS)
  assert.equal(cutAge(0), 0)
  assert.equal(cutAge(-2), 0)
  assert.equal(cutAge(7.5), 7.5)
  assert.equal(retainedShare(undefined), 0)
  assert.equal(retainedShare(1.4), 1)
  assert.equal(retainedShare(0.3), 0.3)
})

test('green-up rises with age through its stages', () => {
  assert.ok(greenShare(0) < 0.15)
  assert.ok(greenShare(1) < 0.15)
  assert.ok(greenShare(3) > 0.3 && greenShare(3) < 0.6)
  assert.ok(greenShare(5) > 0.7)
  assert.equal(greenShare(15), 1)
  for (let a = 0; a < 30; a++) assert.ok(greenShare(a + 1) >= greenShare(a))
})

test('canopy reads bilinear over known posts, NaN where unknown or outside', () => {
  const at = canopySampler(grid([0, 0, 0, 1, 1, 1, null, null, null]))
  assert.ok(at)
  assert.equal(at(0, 0), 1)
  assert.equal(at(-5, -5), 0.5)
  // Above the middle row only unknown posts lie beyond it: the known ones carry the value.
  assert.equal(at(0, 5), 1)
  assert.ok(Number.isNaN(at(50, 0)))
  assert.equal(canopySampler(grid(undefined)), null)
  assert.equal(canopySampler(grid([1, 2])), null)
  const unknown = canopySampler(grid(new Array(9).fill(null)))
  assert.ok(unknown && Number.isNaN(unknown(0, 0)))
})

test('closure from stems: crown area over cell area, open where no stem stands, unknown past their reach', () => {
  // A 300 × 300 m patch of stems 5 m apart with 4 m crowns: 12.57 m² of crown per 25 m², about half closed.
  const data: number[] = []
  for (let x = -150; x < 150; x += 5) for (let y = -150; y < 150; y += 5) data.push(x, y, 0, 0, 20, 4, 0.5)
  const stems = { fields: ['x', 'y', 'z', 'species', 'height', 'crownWidth', 'tone'], count: data.length / 7, data }
  const at = stemCoverSampler(stems, 400)
  assert.ok(at)
  assert.ok(Math.abs(at(0, 0) - (Math.PI * 4) / 25) < 0.02, String(at(0, 0)))
  assert.equal(at(260, -260), 0)
  assert.ok(Number.isNaN(at(500, 0)))
})

test('the package canopy comes first, then its stems', () => {
  const data = [0, 0, 0, 0, 20, 30, 0.5]
  const stems = { fields: ['x', 'y', 'z', 'species', 'height', 'crownWidth', 'tone'], count: 1, data }
  const withCanopy = coverSampler({ terrain: grid([0, 0, 0, 0, 0.2, 0, 0, 0, 0]), trees: stems, treeRadiusMeters: 400 })
  assert.equal(withCanopy.source, 'package')
  assert.equal(withCanopy.at(0, 0), 0.2)
  // Past the package grid the stems answer.
  assert.equal(withCanopy.at(150, 150), 0)
  const fromStems = coverSampler({ terrain: grid(undefined), trees: stems, treeRadiusMeters: 200 })
  assert.equal(fromStems.source, 'stems')
  assert.ok(fromStems.at(0, 0) > 0)
  const none = coverSampler({ terrain: grid(undefined) })
  assert.equal(none.source, 'none')
  assert.ok(Number.isNaN(none.at(0, 0)))
})

test('the skyline grid answers past the far grid', () => {
  const near = grid([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
  const skyline = grid([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], -1000, 1000)
  const cover = coverSampler({ terrain: near, skylineTerrain: skyline })
  assert.equal(cover.at(0, 0), 0.5)
  assert.equal(cover.at(800, 0), 0.9)
})

test('the VIA look paints fresh cut ground pinkish tan for a year, then lets it green up', () => {
  assert.equal(VIA_CUT_SRGB, '#b89a86')
  const [r, g, b] = srgbToLinear(VIA_CUT_SRGB)
  assert.ok(Math.abs(r - 0.479) < 0.002 && Math.abs(g - 0.323) < 0.002 && Math.abs(b - 0.238) < 0.002)
  // Warm: red over green over blue, and the albedo under the renders' lit colour.
  assert.ok(VIA_CUT_LINEAR[0] > VIA_CUT_LINEAR[1] && VIA_CUT_LINEAR[1] > VIA_CUT_LINEAR[2])
  assert.ok(VIA_CUT_LINEAR[0] < r)
  assert.equal(viaFreshCut(0), 1)
  assert.equal(viaFreshCut(1), 1)
  assert.ok(viaFreshCut(1.5) > 0 && viaFreshCut(1.5) < 1)
  assert.equal(viaFreshCut(2), 0)
  assert.equal(viaFreshCut(10), 0)
})

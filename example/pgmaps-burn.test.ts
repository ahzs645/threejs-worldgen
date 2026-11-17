import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  burnAge,
  burnAgeSampler,
  DEFAULT_BURN_AGE_YEARS,
  regrowthShare,
  shrubShare,
  SNAG_CROWN_SHARE,
  snagForm,
  standingSnagShare,
  viewYear
} from './pgmaps-burn.ts'

test('a burnt post without a fire year is a recent fire', () => {
  assert.equal(burnAge(null), DEFAULT_BURN_AGE_YEARS)
  assert.equal(burnAge(undefined), DEFAULT_BURN_AGE_YEARS)
  assert.equal(burnAge(Number.NaN), DEFAULT_BURN_AGE_YEARS)
  assert.equal(burnAge(-2), 0)
  assert.equal(burnAge(7), 7)
})

test('regrowth closes over the char with the years, shrubs coming in later', () => {
  assert.ok(regrowthShare(0) <= 0.12)
  assert.ok(regrowthShare(3) > 0.75 && regrowthShare(3) < 0.9)
  assert.equal(regrowthShare(10), 1)
  for (let a = 0; a < 20; a += 0.5) {
    assert.ok(regrowthShare(a + 0.5) >= regrowthShare(a))
    assert.ok(shrubShare(a + 0.5) >= shrubShare(a))
  }
  assert.ok(shrubShare(1) < 0.35 && shrubShare(12) > 0.6)
})

test('snags stand for the first years, half by fifteen, never fewer than a fifth', () => {
  assert.equal(standingSnagShare(0), 1)
  assert.equal(standingSnagShare(3), 1)
  assert.ok(Math.abs(standingSnagShare(15) - 0.5) < 1e-9)
  assert.equal(standingSnagShare(60), 0.2)
})

test("a snag keeps its stem's height unless broken, and only its stubs of the crown", () => {
  const whole = snagForm(24, 5, 0.8)
  assert.equal(whole.height, 24)
  assert.equal(whole.crownWidth, 5 * SNAG_CROWN_SHARE)
  const broken = snagForm(24, 5, 0.1)
  assert.ok(broken.height >= 24 * 0.6 && broken.height < 24 * 0.9)
  assert.ok(snagForm(10, 0, 0.5).crownWidth > 0)
})

test("the view's year: the environment's date, else when the package was written", () => {
  assert.equal(viewYear({ environment: { dateTime: '2025-08-15T13:00:00-07:00' }, createdAt: '2026-01-01' }), 2025)
  assert.equal(viewYear({ createdAt: '2026-11-14T00:00:00Z' }), 2026)
  assert.equal(viewYear({}), null)
})

test("years since the fire: the post's season (PGMaps burntYear), aged by the package's fires", () => {
  const grid = (burntYear: Array<number | null>) => ({ west: 0, south: 0, spacing: 10, columns: 2, rows: 2, burntYear })
  const fires = [
    { fireNumber: 'K20001', year: 2021, yearsSinceFire: 4, sizeHectares: 120 },
    { fireNumber: null, year: 2017, yearsSinceFire: 8, sizeHectares: null }
  ]
  const age = burnAgeSampler({ terrain: grid([2021, null, null, 2017]), fires, createdAt: '2026-11-14' })!
  assert.equal(age(1, 1), 4)
  assert.equal(age(9, 9), 8)
  // The nearest post that carries a season answers in the cell; outside the grid nothing does.
  assert.equal(age(4, 4), 4)
  assert.ok(Number.isNaN(age(50, 50)))
  // A season the fires table does not list is aged by the view's year; a later season reads 0.
  const bare = burnAgeSampler({
    terrain: grid([2019, 2030, null, null]),
    environment: { dateTime: '2025-07-01T12:00' }
  })!
  assert.equal(bare(0, 0), 6)
  assert.equal(bare(10, 0), 0)
  assert.ok(Number.isNaN(burnAgeSampler({ terrain: grid([2019, null, null, null]) })!(0, 0)))
  // The far grid answers past the near one.
  const far = { west: -100, south: -100, spacing: 50, columns: 5, rows: 5, burntYear: new Array(25).fill(2021) }
  assert.equal(burnAgeSampler({ terrain: grid([null, null, null, null]), farTerrain: far, fires })!(80, 80), 4)
  assert.equal(burnAgeSampler({}), null)
})

// Burnt ground in a PGMaps package (cover 7, `SCENE_COVER.burnt`): a recent wildfire. Its stand is drawn as the VIA
// renders draw one (the Gun Lake CP007 West Gun Boat Launch render): bare grey-black snags, standing at the stand's own
// stems, over yellow-green herb and grey-green shrub regrowth that closes with the years since the fire; far off, a
// dark grey-green mottle streaked by the snags. Here: the fire's age at a point, how far the regrowth has come, and
// each snag's form. Display only: nothing here moves a figure. No DOM, no three.js: the tests run these in Node.

/** Years since the fire a burnt post is drawn at when the package gives no fire year (a recent wildfire). */
export const DEFAULT_BURN_AGE_YEARS = 3

/** A burnt post's drawn age, years: the given one, else {@link DEFAULT_BURN_AGE_YEARS}; never negative. */
export function burnAge(years: number | null | undefined): number {
  return typeof years === 'number' && Number.isFinite(years) ? Math.max(0, years) : DEFAULT_BURN_AGE_YEARS
}

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

/**
 * The share of burnt ground the regrowth covers at `age` years since the fire: charred soil and ash with a little
 * fireweed the first summer, herbs and grass over most of it by three years, shrubs closing it by ten. The rest is
 * char, ash and down wood. Matches the ground shader's stages (`burntGround` in pgmaps-ground.ts).
 */
export function regrowthShare(age: number): number {
  return Math.min(1, 0.1 + 0.72 * smooth(0, 3, age) + 0.18 * smooth(3, 10, age))
}

/** The share of the regrowth that is shrub (grey-green) rather than herb (yellow-green), rising with the years. */
export function shrubShare(age: number): number {
  return 0.3 + 0.35 * smooth(2, 12, age)
}

/**
 * The share of a burnt stand's stems still standing as snags `age` years after the fire: all for the first years,
 * about half by fifteen (pine snags fall over a decade or two), a fifth by thirty.
 */
export function standingSnagShare(age: number): number {
  return Math.max(0.2, 1 - 0.5 * smooth(3, 15, age) - 0.3 * smooth(15, 30, age))
}

/** A snag's crown (its dead branch stubs) against the living crown's width. */
export const SNAG_CROWN_SHARE = 0.4

/**
 * A snag's drawn form from its stem's height and crown width and its own number `u` in [0, 1) (`stemHash`): a third of
 * them broken off at 60–90% of the height, every crown cut to its bare branch stubs ({@link SNAG_CROWN_SHARE}).
 */
export function snagForm(height: number, crownWidth: number, u: number): { height: number; crownWidth: number } {
  const broken = u < 0.33
  const k = broken ? 0.6 + 0.3 * (u / 0.33) : 1
  return { height: height * k, crownWidth: Math.max(0.3, crownWidth * SNAG_CROWN_SHARE) }
}

// ---------------------------------------------------------------- the fire's year

/** A package grid's layout and its fire season per burnt post (PGMaps `burntYear`; null where not burnt). */
export interface FireGrid {
  west: number
  south: number
  spacing: number
  columns: number
  rows: number
  burntYear?: Array<number | null> | null
}

/** One wildfire behind cover 7 (PGMaps `fires`): its season and the years from it to the assessment year. */
export interface SceneFire {
  fireNumber?: string | null
  year: number
  yearsSinceFire?: number | null
  /** The same, under the shorter name. */
  yearsSince?: number | null
  sizeHectares?: number | null
}

/** What the fire's age is read from: the grids' `burntYear`, the package's `fires` and the year the view is set in. */
export interface FireInput {
  terrain?: FireGrid | null
  farTerrain?: FireGrid | null
  skylineTerrain?: FireGrid | null
  fires?: SceneFire[] | null
  createdAt?: string | null
  environment?: { dateTime?: string | null; utc?: string | null } | null
}

/** The year the package's view is set in, for a season `fires` does not list: its environment's date, else `createdAt`. */
export function viewYear(input: FireInput): number | null {
  for (const text of [input.environment?.dateTime, input.environment?.utc, input.createdAt]) {
    const year = typeof text === 'string' ? Number.parseInt(text.slice(0, 4), 10) : NaN
    if (Number.isFinite(year) && year > 1800) return year
  }
  return null
}

/**
 * Years since the fire at a package point, NaN where nothing says: the season at the nearest of the cell's four posts
 * that carries one in `burntYear` (grids in order near, far, skyline), aged by the package's `fires` (its
 * `yearsSinceFire` for that season, at the scene's assessment year), else by the view's year ({@link viewYear}); never
 * negative. null when no grid carries `burntYear`.
 */
export function burnAgeSampler(input: FireInput): ((x: number, y: number) => number) | null {
  const year = viewYear(input)
  const since = new Map<number, number>()
  for (const fire of input.fires ?? []) {
    const years = fire?.yearsSinceFire ?? fire?.yearsSince
    if (typeof fire?.year === 'number' && typeof years === 'number' && Number.isFinite(years))
      since.set(fire.year, years)
  }
  const ageOf = (season: number) => {
    const known = since.get(season)
    if (known !== undefined) return Math.max(0, known)
    return year !== null ? Math.max(0, year - season) : NaN
  }
  const grids: Array<{ grid: FireGrid; values: Array<number | null> }> = []
  for (const grid of [input.terrain, input.farTerrain, input.skylineTerrain])
    if (grid && grid.columns > 1 && grid.rows > 1 && grid.burntYear?.length === grid.columns * grid.rows)
      grids.push({ grid, values: grid.burntYear })
  if (!grids.length) return null
  return (x, y) => {
    for (const { grid, values } of grids) {
      const gx = (x - grid.west) / grid.spacing
      const gy = (y - grid.south) / grid.spacing
      if (!(gx >= 0 && gy >= 0 && gx <= grid.columns - 1 && gy <= grid.rows - 1)) continue
      const x0 = Math.min(grid.columns - 2, Math.floor(gx))
      const y0 = Math.min(grid.rows - 2, Math.floor(gy))
      let best = Infinity
      let season = NaN
      for (const [c, r] of [
        [x0, y0],
        [x0 + 1, y0],
        [x0, y0 + 1],
        [x0 + 1, y0 + 1]
      ]) {
        const v = values[r * grid.columns + c]
        if (typeof v !== 'number' || !Number.isFinite(v)) continue
        const d = (c - gx) ** 2 + (r - gy) ** 2
        if (d < best) {
          best = d
          season = v
        }
      }
      if (Number.isFinite(season)) return ageOf(season)
    }
    return NaN
  }
}

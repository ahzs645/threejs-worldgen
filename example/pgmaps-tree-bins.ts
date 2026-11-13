// How the PGMaps stand's stems are sorted into grown variants, and how the ground's stumps and slash are seeded: every
// rule here reads only the stem (or the opening, or the pile) itself, never the rest of the package. A harvest that
// removes stems, or a package that lists its openings in another order, leaves everything that is still there drawn
// exactly as before. No DOM, no three.js: the tests run these in Node.

/**
 * Height classes grown for every species, metres. A fixed ladder of about ×1.4 a step (4·√2ᵏ, rounded), so a stem is
 * never scaled by more than about ±20% from the class it is drawn with, and a class never moves when other stems come
 * or go. Only classes that have stems are grown: a stand of 14–32 m stems takes four (11, 16, 22, 31).
 */
export const TREE_HEIGHT_LADDER: readonly number[] = [4, 6, 8, 11, 16, 22, 31, 44, 62]

/** The ladder index nearest a stem's height in log terms (ties go to the lower class). */
export function heightBin(height: number): number {
  const h = Number.isFinite(height) && height > 0 ? height : TREE_HEIGHT_LADDER[0]
  let best = 0
  let bestDistance = Math.abs(Math.log(TREE_HEIGHT_LADDER[0] / h))
  for (let i = 1; i < TREE_HEIGHT_LADDER.length; i++) {
    const d = Math.abs(Math.log(TREE_HEIGHT_LADDER[i] / h))
    if (d < bestDistance) {
      best = i
      bestDistance = d
    }
  }
  return best
}

/**
 * Species crown width as a share of height, used when the package's species table has no `crownRatio`. The values are
 * PGMaps' own species defaults (`SPECIES_CROWN_RATIO`, src/pages/dev-forestry/forest.ts).
 */
export const DEFAULT_CROWN_RATIO: Readonly<Record<string, number>> = {
  pine: 0.14,
  spruce: 0.17,
  fir: 0.12,
  douglasFir: 0.2,
  aspen: 0.22,
  birch: 0.2
}
const FALLBACK_CROWN_RATIO = 0.17

/**
 * How much wider the grown class is than the species ratio says. PGMaps widens each stem's crown until the stand shows
 * its inventory crown closure; in the real test package the stems' median width is about 1.5× their species ratio
 * (pine 0.22 against 0.14, spruce 0.26 against 0.17). Growing the class near that keeps the typical stem's horizontal
 * scale near 1, so its branches and leaf cards are not stretched. A constant, so the class never depends on the stems.
 */
export const CROWN_WIDTH_FACTOR = 1.5

/** A class's grown crown width, metres: the species ratio × the class height × {@link CROWN_WIDTH_FACTOR}, to 0.5 m. */
export function binCrownWidth(species: string, bin: number, crownRatio?: number | null): number {
  const ratio =
    typeof crownRatio === 'number' && Number.isFinite(crownRatio) && crownRatio > 0
      ? crownRatio
      : (DEFAULT_CROWN_RATIO[species] ?? FALLBACK_CROWN_RATIO)
  const height = TREE_HEIGHT_LADDER[bin]
  return Math.max(1, Math.round(ratio * CROWN_WIDTH_FACTOR * height * 2) / 2)
}

/** FNV-1a of a string, unsigned 32-bit. */
export function hashText(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193)
  return h >>> 0
}

/** A 32-bit finaliser (murmur3's), so nearby inputs give unrelated seeds. */
function mix(h: number): number {
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b)
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35)
  return (h ^ (h >>> 16)) >>> 0
}

/** The generator seed of a species' height class: from the species and the class only, never the build order. */
export function binSeed(species: string, bin: number): number {
  return mix(hashText(`${species}@${TREE_HEIGHT_LADDER[bin]}`))
}

type Ring = ReadonlyArray<readonly [number, number]>

/**
 * A seed for an opening's stumps and strewn branches, from its own outer ring: the bounding box to the decimetre and
 * the vertex count. Where the opening sits in the package's list, and which vertex its ring starts at, do not matter.
 */
export function openingSeed(rings: ReadonlyArray<Ring>): number {
  const outer = rings[0] ?? []
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const [x, y] of outer) {
    x0 = Math.min(x0, x)
    y0 = Math.min(y0, y)
    x1 = Math.max(x1, x)
    y1 = Math.max(y1, y)
  }
  const dm = (v: number) => (Number.isFinite(v) ? Math.round(v * 10) : 0)
  return mix(hashText(`opening ${dm(x0)} ${dm(y0)} ${dm(x1)} ${dm(y1)} ${outer.length}`))
}

/** A seed for one debris pile, from its own position (to the centimetre) and footprint. */
export function pileSeed(x: number, y: number, footprint: number): number {
  const cm = (v: number) => (Number.isFinite(v) ? Math.round(v * 100) : 0)
  return mix(hashText(`pile ${cm(x)} ${cm(y)} ${cm(footprint)}`))
}

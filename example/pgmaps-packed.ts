// A package's long numeric arrays as PGMaps may send them (its scenePosting.ts): `Float64Array`s, NaN for null, their
// buffers handed over by the post and not copied number by number (about a second of the browser's unpacking for a
// capture's package). `unpackArrays` turns them back, in place, into the plain arrays a build reads: the same doubles
// and nulls, so the package is value for value the one PGMaps packed and nothing downstream changes. A page says it
// takes them in its hello (`ACCEPTS`); a plain package passes through untouched. No DOM, no GPU.

/** What this engine's hello tells PGMaps it takes beyond the plain protocol. */
export const ACCEPTS = ['packed-arrays', 'scene-keys'] as const

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value) && !ArrayBuffer.isView(value)

/** Every `Float64Array` in the package back to a plain array (NaN: null), in place. Returns how many it turned. */
export function unpackArrays(value: unknown): number {
  if (Array.isArray(value)) {
    let turned = 0
    if (value.length && isRecord(value[0])) for (const entry of value) turned += unpackArrays(entry)
    return turned
  }
  if (!isRecord(value)) return 0
  let turned = 0
  for (const name of Object.keys(value)) {
    const entry = value[name]
    if (entry instanceof Float64Array) {
      const plain = new Array<number | null>(entry.length)
      for (let i = 0; i < entry.length; i++) {
        const v = entry[i]
        plain[i] = Number.isNaN(v) ? null : v
      }
      value[name] = plain
      turned++
    } else turned += unpackArrays(entry)
  }
  return turned
}

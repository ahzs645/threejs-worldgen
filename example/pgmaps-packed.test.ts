import { test } from 'node:test'
import assert from 'node:assert/strict'
import { ACCEPTS, unpackArrays } from './pgmaps-packed.ts'

/** What PGMaps' packScene does to a long array: the same doubles, NaN for null. */
const packed = (values: Array<number | null>): Float64Array =>
  Float64Array.from(values, (v) => (v === null ? Number.NaN : v))

test('packed arrays come back as the plain arrays they were, nulls and all, wherever they sit', () => {
  const heights = Array.from({ length: 2000 }, (_, i) => (i % 97 === 0 ? null : 700 + Math.sin(i) * 80.123456789))
  const cover = Array.from({ length: 2000 }, (_, i) => i % 8)
  const data = Array.from({ length: 2100 }, (_, i) => -0.5 + i * 0.731)
  const plain = {
    format: 'pgmaps-forest-scene',
    terrain: { columns: 40, heights, cover, burntYear: heights.map(() => null) },
    trees: { fields: ['x', 'y'], count: 300, data },
    models: { fir: { positions: data.slice(0, 1500) } },
    water: [
      {
        rings: [
          [
            [0, 1],
            [2, 3]
          ]
        ],
        elevationMeters: null
      }
    ],
    notes: ['a note']
  }
  // biome-ignore lint/suspicious/noExplicitAny: a package as it arrives, before it is checked
  const sent = structuredClone(plain) as Record<string, any>
  sent.terrain.heights = packed(heights)
  sent.terrain.cover = packed(cover)
  sent.terrain.burntYear = packed(plain.terrain.burntYear)
  sent.trees.data = packed(data)
  sent.models.fir.positions = packed(plain.models.fir.positions)
  assert.equal(unpackArrays(sent), 5)
  assert.deepEqual(sent, plain)
  assert.ok(Array.isArray(sent.terrain.heights) && sent.terrain.heights[0] === null)
})

test('a plain package is left exactly as it is', () => {
  const plain = { terrain: { heights: [1, null, 3] }, openings: [{ rings: [[[0, 0]]] }], notes: [] }
  const copy = structuredClone(plain)
  assert.equal(unpackArrays(copy), 0)
  assert.deepEqual(copy, plain)
  assert.equal(unpackArrays(null), 0)
})

test('the hello names both things PGMaps may then send', () => {
  assert.deepEqual([...ACCEPTS], ['packed-arrays', 'scene-keys'])
})

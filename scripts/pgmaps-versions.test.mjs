import { test } from 'node:test'
import assert from 'node:assert/strict'
import { compareVersionsDescending, versionsIndex } from './pgmaps-versions.mjs'

test('orders versions newest first, a release before its prereleases', () => {
  const sorted = ['1.0.0', '1.10.0', '1.2.0', '1.2.0-rc.1', '1.2.0-rc.10', '0.9.9'].sort(compareVersionsDescending)
  assert.deepEqual(sorted, ['1.10.0', '1.2.0', '1.2.0-rc.10', '1.2.0-rc.1', '1.0.0', '0.9.9'])
})

test('indexes releases and picks the newest non-prerelease as latest', () => {
  const index = versionsIndex([
    { version: '1.0.0', tag: 'pgmaps-scene-v1.0.0', commit: 'abc1234', date: '2026-11-15T00:00:00Z' },
    { version: '1.1.0-rc.1', commit: 'def5678' },
    { version: 'not-a-version' }
  ])
  assert.equal(index.latest, '1.0.0')
  assert.deepEqual(
    index.versions.map((v) => [v.version, v.tag, v.page]),
    [
      ['1.1.0-rc.1', 'pgmaps-scene-v1.1.0-rc.1', '1.1.0-rc.1/pgmaps-scene.html'],
      ['1.0.0', 'pgmaps-scene-v1.0.0', '1.0.0/pgmaps-scene.html']
    ]
  )
  assert.equal(versionsIndex([]).latest, null)
})

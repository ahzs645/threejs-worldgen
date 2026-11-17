import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  boundedDissolve,
  DEFAULT_RENDER_OPTIONS,
  FramePacer,
  mergeRenderOptions,
  renderOptionsFromParams
} from './pgmaps-render-options.ts'

test('adapter URLs and messages normalize the same independent controls', () => {
  const url = renderOptionsFromParams(
    new URLSearchParams(
      'renderScale=.7&maxfps=44.6&gpuqueue=2.8&dissolve=relative&cutoffpx=3&referenceheight=1079.7&culling=off'
    )
  )
  assert.deepEqual(url, {
    resolutionScale: 0.7,
    maxFps: 45,
    maxInFlight: 3,
    dissolveMode: 'relative',
    dissolvePixels: 3,
    dissolveReferenceHeight: 1080,
    frustumCulling: false
  })
  assert.deepEqual(
    mergeRenderOptions(DEFAULT_RENDER_OPTIONS, {
      resolutionScale: NaN,
      maxFps: Infinity,
      dissolveMode: 'bad',
      frustumCulling: 'false'
    }),
    DEFAULT_RENDER_OPTIONS
  )
  const clamped = mergeRenderOptions(DEFAULT_RENDER_OPTIONS, {
    resolutionScale: 2,
    maxFps: 8,
    maxInFlight: 5,
    dissolvePixels: 30,
    dissolveReferenceHeight: 40
  })
  assert.equal(clamped.resolutionScale, 1)
  assert.equal(clamped.maxFps, 15)
  assert.equal(clamped.maxInFlight, 3)
  assert.equal(clamped.dissolvePixels, 16)
  assert.equal(clamped.dissolveReferenceHeight, 240)
})

test('relative dissolve is size-invariant; pixel dissolve follows frame size; both respect actual coverage and eye offset', () => {
  const relative = mergeRenderOptions(DEFAULT_RENDER_OPTIONS, { dissolveMode: 'relative' })
  const cutoff = (h: number, options = relative, offset = 0) => boundedDissolve(20, h, 40, options, 12000, offset, 64)
  assert.ok(Math.abs(cutoff(540) - cutoff(2160)) < 1e-8)
  assert.ok(cutoff(2160, DEFAULT_RENDER_OPTIONS) > cutoff(540, DEFAULT_RENDER_OPTIONS))
  assert.equal(cutoff(2160, DEFAULT_RENDER_OPTIONS, 11000), 936)
  assert.equal(cutoff(2160, DEFAULT_RENDER_OPTIONS, 12500), 0)
  assert.ok(boundedDissolve(30, 8192, 20, relative, 2400, 200, 64, 2) <= 2136)
})

test('queue and FPS pacing release after completion or rejection without blocking input', async () => {
  const p = new FramePacer()
  const failures: string[] = []
  p.onFailure = (message) => {
    failures.push(message)
  }
  const options = mergeRenderOptions(DEFAULT_RENDER_OPTIONS, { maxFps: 30, maxInFlight: 1 })
  let finish!: () => void
  assert.equal(p.canSubmit(0, options), true)
  p.submitted(
    0,
    new Promise<void>((resolve) => {
      finish = resolve
    })
  )
  assert.equal(p.canSubmit(100, options), false)
  finish()
  await p.drain()
  assert.equal(p.completed, 1)
  assert.equal(p.canSubmit(10, options), false)
  assert.equal(p.canSubmit(34, options), true)
  p.submitted(34, Promise.reject(new Error('device lost')))
  await p.drain()
  assert.equal(p.inFlight, 0)
  assert.equal(p.failed, 1)
  assert.equal(p.lastError, 'device lost')
  assert.deepEqual(failures, ['device lost'])
  assert.equal(p.completed, 1)
  assert.equal(p.canSubmit(68, options), true)
})

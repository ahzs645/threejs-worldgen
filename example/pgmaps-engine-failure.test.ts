import { test } from 'node:test'
import assert from 'node:assert/strict'
import { EngineFailure } from './pgmaps-engine-failure.ts'

test('failure during an awaited build prevents subsequent baking and preserves one error across loader catch/state replacement', async () => {
  const failure = new EngineFailure()
  const errors: string[] = []
  const notify = (message: string) => {
    errors.push(message)
  }
  let release!: () => void
  let bakes = 0
  const build = (async () => {
    await new Promise<void>((resolve) => {
      release = resolve
    })
    failure.check()
    bakes++
  })()
  assert.equal(failure.progress('building the ground'), 'building the ground')
  assert.equal(failure.report('GPU completion failed', notify), true)
  release()
  await assert.rejects(build, /GPU completion failed/)
  assert.equal(bakes, 0)
  assert.equal(failure.report('loader also caught failure', notify), false)
  assert.deepEqual(errors, ['GPU completion failed'])
  assert.equal(failure.progress('baking the far forest'), 'GPU completion failed')
  assert.deepEqual(failure.decorate({ ready: false, package: 'new scene' }), {
    ready: true,
    package: 'new scene',
    error: 'GPU completion failed',
    lastError: 'GPU completion failed'
  })
})

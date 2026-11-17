import { test } from 'node:test'
import assert from 'node:assert/strict'
import { gpuCompletion } from './pgmaps-gpu-pacing.ts'

test('WebGPU uses the submission queue completion and propagates failure to the pacer', async () => {
  let completed = false
  const gpu = gpuCompletion({
    device: {
      queue: {
        async onSubmittedWorkDone() {
          completed = true
        }
      }
    }
  })
  assert.equal(gpu.backend, 'webgpu')
  await gpu.done()
  assert.equal(completed, true)
  await assert.rejects(
    gpuCompletion({
      device: {
        queue: {
          onSubmittedWorkDone() {
            return Promise.reject(new Error('lost'))
          }
        }
      }
    }).done(),
    /lost/
  )
})

test('WebGL polls fences asynchronously and deletes them after completion or failure', async () => {
  let polls = 0
  let deleted = 0
  const gl = {
    SYNC_GPU_COMMANDS_COMPLETE: 1,
    TIMEOUT_EXPIRED: 2,
    WAIT_FAILED: 3,
    fenceSync: () => ({}),
    flush() {},
    clientWaitSync: (): number => (++polls === 1 ? 2 : 4),
    deleteSync() {
      deleted++
    }
  }
  const gpu = gpuCompletion({ gl })
  assert.equal(gpu.backend, 'webgl2')
  const pending = gpu.done()
  assert.equal(polls, 0)
  await pending
  assert.equal(polls, 2)
  assert.equal(deleted, 1)
  gl.clientWaitSync = () => 3
  await assert.rejects(gpu.done(), /failed/)
  assert.equal(deleted, 2)
  assert.equal(gpuCompletion({}).backend, 'unavailable')
})

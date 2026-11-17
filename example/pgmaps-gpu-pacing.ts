/** three r186 removed Renderer.waitForGPU; complete submitted work using the active backend's own primitive. */
export function gpuCompletion(backend: unknown): { backend: string; done(): Promise<void> } {
  const active = backend as {
    device?: { queue?: { onSubmittedWorkDone(): Promise<void> } }
    gl?: WebGL2RenderingContext
  }
  const queue = active.device?.queue
  if (queue?.onSubmittedWorkDone)
    return {
      backend: 'webgpu',
      done: async () => {
        await queue.onSubmittedWorkDone()
      }
    }
  const gl = active.gl
  if (gl?.fenceSync)
    return {
      backend: 'webgl2',
      async done() {
        const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
        if (!fence) return Promise.reject(new Error('Unable to fence GPU work'))
        gl.flush()
        return new Promise<void>((resolve, reject) => {
          const poll = () => {
            try {
              const status = gl.clientWaitSync(fence, 0, 0)
              if (status === gl.TIMEOUT_EXPIRED) {
                setTimeout(poll, 4)
                return
              }
              gl.deleteSync(fence)
              if (status === gl.WAIT_FAILED) reject(new Error('GPU fence failed'))
              else resolve()
            } catch (error) {
              reject(error)
            }
          }
          setTimeout(poll, 0)
        })
      }
    }
  // Explicitly label unsupported completion; frame caps still work, but this is no GPU completion measurement.
  return { backend: 'unavailable', done: () => Promise.resolve() }
}

/** PGMaps adapter controls. These never change the shared worldgen defaults. */
export interface RenderOptions {
  resolutionScale: number
  maxFps: number
  maxInFlight: number
  dissolveMode: 'pixels' | 'relative'
  dissolvePixels: number
  dissolveReferenceHeight: number
  frustumCulling: boolean
}

export const DEFAULT_RENDER_OPTIONS: Readonly<RenderOptions> = Object.freeze({
  resolutionScale: 1,
  maxFps: 0,
  maxInFlight: 2,
  dissolveMode: 'pixels',
  dissolvePixels: 4,
  dissolveReferenceHeight: 1080,
  frustumCulling: true
})

const bounded = (v: unknown, lo: number, hi: number, fallback: number) =>
  typeof v === 'number' && Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : fallback

export function mergeRenderOptions(base: Readonly<RenderOptions>, input: unknown): RenderOptions {
  const p = input && typeof input === 'object' ? (input as Partial<RenderOptions>) : {}
  return {
    resolutionScale: bounded(p.resolutionScale, 0.5, 1, base.resolutionScale),
    maxFps: p.maxFps === 0 ? 0 : Math.round(bounded(p.maxFps, 15, 120, base.maxFps)),
    maxInFlight: Math.round(bounded(p.maxInFlight, 1, 3, base.maxInFlight)),
    dissolveMode: p.dissolveMode === 'relative' || p.dissolveMode === 'pixels' ? p.dissolveMode : base.dissolveMode,
    dissolvePixels: bounded(p.dissolvePixels, 0.5, 16, base.dissolvePixels),
    dissolveReferenceHeight: Math.round(bounded(p.dissolveReferenceHeight, 240, 4320, base.dissolveReferenceHeight)),
    frustumCulling: typeof p.frustumCulling === 'boolean' ? p.frustumCulling : base.frustumCulling
  }
}

export function renderOptionsFromParams(params: URLSearchParams): RenderOptions {
  const n = (key: string) => (params.has(key) ? Number(params.get(key)) : undefined)
  return mergeRenderOptions(DEFAULT_RENDER_OPTIONS, {
    resolutionScale: n('renderScale'),
    maxFps: n('maxfps'),
    maxInFlight: n('gpuqueue'),
    dissolveMode: params.get('dissolve'),
    dissolvePixels: n('cutoffpx'),
    dissolveReferenceHeight: n('referenceheight'),
    frustumCulling: params.has('culling') ? !['0', 'false', 'off'].includes(params.get('culling')!) : undefined
  })
}

/** Actual sampled patch reach, reduced by the eye's offset and the full edge of the last patch. */
export function boundedDissolve(
  height: number,
  frameHeight: number,
  vfov: number,
  options: Readonly<RenderOptions>,
  coverageRadius: number,
  eyeOffset: number,
  guard: number,
  requestedPx = options.dissolvePixels
): number {
  const px =
    options.dissolveMode === 'relative' ? (requestedPx * frameHeight) / options.dissolveReferenceHeight : requestedPx
  const focal = frameHeight / 2 / Math.tan((vfov * Math.PI) / 360)
  const desired = (Math.max(0, height) * focal) / Math.max(0.001, px)
  return Math.min(Math.max(0, coverageRadius - Math.max(0, eyeOffset) - Math.max(0, guard)), Math.max(1500, desired))
}

/** Submission pacing with asynchronous completion, so input and the camera stay fresh while the GPU catches up. */
export class FramePacer {
  inFlight = 0
  completed = 0
  failed = 0
  lastError: string | null = null
  onFailure: ((message: string) => void) | null = null
  completionMs = 0
  lastSubmitted = -Infinity
  private pending = new Set<Promise<void>>()
  canSubmit(now: number, options: Readonly<RenderOptions>): boolean {
    return (
      this.inFlight < options.maxInFlight &&
      (!options.maxFps || now - this.lastSubmitted >= 1000 / options.maxFps - 0.25)
    )
  }
  submitted(now: number, completion: Promise<unknown>): void {
    this.lastSubmitted = now
    this.inFlight++
    const task = completion
      .then(() => {
        this.completed++
        const elapsed = performance.now() - now
        this.completionMs = this.completionMs ? 0.9 * this.completionMs + 0.1 * elapsed : elapsed
      })
      .catch((error) => {
        this.failed++
        this.lastError = error instanceof Error ? error.message : String(error)
        try {
          this.onFailure?.(this.lastError)
        } catch {}
      })
      .finally(() => {
        this.inFlight--
        this.pending.delete(task)
      })
    this.pending.add(task)
  }
  async drain(): Promise<void> {
    await Promise.all([...this.pending])
  }
}

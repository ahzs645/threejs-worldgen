// Where the PGMaps scene runs. The same scene code (pgmaps-scene.ts) runs in two hosts:
//   - its own page (pgmaps-scene.html): opened alone, or in a frame PGMaps lays over its road view, the messages
//     through window.parent;
//   - a worker PGMaps starts itself (pgmaps-worker.ts, built by `npm run build:pgmaps-worker`), drawing into the
//     OffscreenCanvas PGMaps hands it, the messages through the worker's own port. A worker has no window, no document
//     and no DOM events: PGMaps sends its size ({ type: 'pgmaps-worker-viewport' }) and places the canvas where the
//     scene asks ({ type: 'pgmaps-engine-viewport' }), and the scene's page-only parts (the HUD, the orbit controls,
//     the file picker) stand down.
// Every message the scene sends goes through `post`, so the protocol is the same in both hosts. PGMaps' side of it
// is src/pages/dev-forestry/engineChannel.ts there.

/** The size to draw at, in CSS pixels, and the device's pixels per CSS pixel. */
export type Viewport = { width: number; height: number; pixelRatio: number }
/** Where the drawing sits in its host's box (the lens kept: fitCameraViewport letterboxes it), CSS pixels. */
export type Placement = { left: number; top: number; width: number; height: number }

export interface SceneHost {
  /** Running in a worker PGMaps started (no DOM). */
  readonly worker: boolean
  /** Driven by a parent (PGMaps): a frame, or a worker, which always is. */
  readonly embedded: boolean
  /** Settles once the scene can make its renderer: at once on a page, when PGMaps hands over the canvas in a worker. */
  readonly ready: Promise<void>
  /** A message to PGMaps, its buffers handed over. Nothing when the page is opened alone. */
  post(message: Record<string, unknown>, transfer?: Transferable[]): void
  /** A line for whoever looks at the page (its HUD); a worker has none. */
  status(text: string): void
  /** The canvas the renderer draws into: the worker's OffscreenCanvas, or undefined for one the renderer makes. */
  canvas(): OffscreenCanvas | undefined
  viewport(): Viewport
  /** Called when the viewport changes. */
  onResize(listener: () => void): void
  /** Shows the renderer's canvas at `view`. */
  place(canvas: HTMLCanvasElement | OffscreenCanvas, view: Placement): void
  /** Puts the renderer's canvas on the page. */
  mount(canvas: HTMLCanvasElement | OffscreenCanvas): void
  /** Names on the global object, for checking from the page (`pgmaps`). */
  expose(values: Record<string, unknown>): void
  /** The global `pgmaps` state, when there is one. */
  state<T = Record<string, unknown>>(): T | undefined
}

const params = new URLSearchParams(location.search)
const expose = (values: Record<string, unknown>): void => void Object.assign(globalThis, values)
const state = <T>() => (globalThis as unknown as { pgmaps?: T }).pgmaps

function pageHost(): SceneHost {
  const hud = document.getElementById('hud')
  const framed = window.parent !== window
  if (params.has('embed') && hud) hud.hidden = true
  return {
    worker: false,
    embedded: params.has('embed'),
    ready: Promise.resolve(),
    post: (message, transfer = []) => {
      if (framed) window.parent.postMessage(message, { targetOrigin: '*', transfer })
    },
    status: (text) => {
      if (hud) hud.textContent = text
    },
    canvas: () => undefined,
    viewport: () => ({ width: innerWidth, height: innerHeight, pixelRatio: devicePixelRatio }),
    onResize: (listener) => addEventListener('resize', listener),
    place: (canvas, view) => {
      Object.assign((canvas as HTMLCanvasElement).style, {
        position: 'absolute',
        left: `${view.left}px`,
        top: `${view.top}px`
      })
    },
    mount: (canvas) => void document.body.appendChild(canvas as HTMLCanvasElement),
    expose,
    state
  }
}

function workerHost(): SceneHost {
  let canvas: OffscreenCanvas | undefined
  let size: Viewport = { width: 1, height: 1, pixelRatio: 1 }
  let started = () => {}
  const ready = new Promise<void>((resolve) => (started = resolve))
  const listeners = new Set<() => void>()
  const send = (message: Record<string, unknown>, transfer: Transferable[] = []): void =>
    (
      globalThis as unknown as { postMessage: (data: unknown, options: StructuredSerializeOptions) => void }
    ).postMessage(message, { transfer })
  const viewportOf = (raw: unknown): Viewport | null => {
    const v = raw as Partial<Viewport> | null
    if (!v || !(Number(v.width) > 0) || !(Number(v.height) > 0)) return null
    return {
      width: Math.round(Number(v.width)),
      height: Math.round(Number(v.height)),
      pixelRatio: Number(v.pixelRatio) > 0 ? Number(v.pixelRatio) : 1
    }
  }
  addEventListener('message', (event: MessageEvent) => {
    const data = event.data as { type?: string; canvas?: unknown; viewport?: unknown } | null
    if (data?.type === 'pgmaps-worker-init' && data.canvas instanceof OffscreenCanvas) {
      canvas = data.canvas
      size = viewportOf(data.viewport) ?? size
      started()
    } else if (data?.type === 'pgmaps-worker-viewport') {
      const next = viewportOf(data.viewport)
      if (!next || (next.width === size.width && next.height === size.height && next.pixelRatio === size.pixelRatio))
        return
      size = next
      for (const listener of listeners) listener()
    }
  })
  return {
    worker: true,
    embedded: true,
    ready,
    post: send,
    status: () => {},
    canvas: () => {
      if (!canvas) throw new Error('PGMaps has not handed this worker its canvas (pgmaps-worker-init).')
      return canvas
    },
    viewport: () => size,
    onResize: (listener) => void listeners.add(listener),
    place: (_canvas, view) => send({ type: 'pgmaps-engine-viewport', ...view }),
    mount: () => {},
    expose,
    state
  }
}

/** This scene's host: a worker's when there is no document, else the page's. */
export const host: SceneHost = typeof document === 'undefined' ? workerHost() : pageHost()

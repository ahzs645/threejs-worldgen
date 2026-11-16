// The PGMaps scene as a worker PGMaps starts itself (its road view's renderer, src/pages/dev-forestry/engineChannel.ts
// in PGMaps), not a page in a frame. PGMaps hands it an OffscreenCanvas laid over its map
// ({ type: 'pgmaps-worker-init', canvas, viewport }, pgmaps-host.ts) and then speaks the same protocol as to the page
// (pgmaps-scene.ts), over the worker's port. Built alone by `npm run build:pgmaps-worker` into
// dist-pgmaps-worker/pgmaps-engine.js.
// The host first, so its listener takes the canvas; the scene, which makes its renderer as it loads, only once the
// canvas is here (a module still evaluating may not be handed messages, so it must not wait for one at its top).
import { host } from './pgmaps-host'

void host.ready.then(() => import('./pgmaps-scene'))

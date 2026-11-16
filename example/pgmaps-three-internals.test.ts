// A tripwire, not a behaviour test. pgmaps-capture.ts adds a final frame's samples up on the GPU, and for that it
// starts a new three.js node frame per sample through `renderer._nodes.nodeFrame.update()`, which is not public API.
// These read the installed three's source for the facts that code leans on. One failing after a three upgrade means:
// read the sample loop in pgmaps-capture.ts (`newFrame`, `sumOnGpu`) against the new three before trusting a capture.
// The page itself also checks at run time and falls back to reading each sample back when the frame id does not move.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const three = (path: string): string => readFileSync(new URL(`../node_modules/three/${path}`, import.meta.url), 'utf8')

test('three still keeps its node manager on renderer._nodes, with the frame on nodeFrame', () => {
  assert.match(three('src/renderers/common/Renderer.js'), /this\._nodes = new NodeManager\(/)
  assert.match(three('src/renderers/common/nodes/NodeManager.js'), /this\.nodeFrame = new NodeFrame\(\)/)
})

test('NodeFrame.update() still starts a frame by moving frameId on', () => {
  assert.match(three('src/nodes/core/NodeFrame.js'), /\n\tupdate\(\) \{\s+this\.frameId \+\+;/)
})

test('the passes still redraw once per frame id, which is why each sample needs a frame of its own', () => {
  assert.match(three('src/nodes/display/PassNode.js'), /this\.updateBeforeType = NodeUpdateType\.FRAME/)
  assert.match(three('examples/jsm/tsl/display/BloomNode.js'), /this\.updateBeforeType = NodeUpdateType\.FRAME/)
})

test('only the animation loop starts a frame: there is still no public call for it', () => {
  assert.match(three('src/renderers/common/Animation.js'), /this\.nodes\.nodeFrame\.update\(\)/)
})

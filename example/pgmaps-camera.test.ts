import assert from 'node:assert/strict'
import { test } from 'node:test'
import { PerspectiveCamera, Vector3 } from 'three'
import { fitCameraViewport } from './pgmaps-camera.ts'

const projection = { kind: 'perspective' as const, aspectRatio: 16 / 9 }

test('a saved perspective frame fits a wide or tall window without changing the lens', () => {
  const wide = fitCameraViewport(projection, 1600, 500)
  assert.ok(Math.abs(wide.width - (500 * 16) / 9) < 1e-9)
  assert.equal(wide.height, 500)
  assert.equal(wide.left, (1600 - wide.width) / 2)
  assert.equal(wide.top, 0)
  const tall = fitCameraViewport(projection, 600, 1000)
  assert.equal(tall.width, 600)
  assert.equal(tall.height, 337.5)
  assert.equal(tall.top, (1000 - tall.height) / 2)
  assert.equal(tall.aspect, projection.aspectRatio)
})

test('known landscape points keep their image coordinates after resizing or reopening', () => {
  const point = new Vector3(100, 750, -1000)
  const projected = (width: number, height: number) => {
    const view = fitCameraViewport(projection, width, height)
    const camera = new PerspectiveCamera(36.86989764584402, view.aspect, 0.3, 30000)
    camera.position.set(0, 701.513728, 0)
    camera.lookAt(0, 701.513728, -1000)
    camera.updateMatrixWorld()
    const p = point.clone().project(camera)
    return [p.x, p.y]
  }
  assert.deepEqual(projected(1600, 900), projected(600, 1000))
  assert.deepEqual(projected(1600, 900), projected(3200, 1800))
})

test('old packages use the window and malformed saved projections fail explicitly', () => {
  assert.deepEqual(fitCameraViewport(undefined, 600, 1000), { width: 600, height: 1000, left: 0, top: 0, aspect: 0.6 })
  for (const aspectRatio of [NaN, Infinity, 0, -1])
    assert.throws(() => fitCameraViewport({ ...projection, aspectRatio }, 800, 600), /projection/)
  assert.throws(() => fitCameraViewport(projection, 0, 600), /viewport/)
})

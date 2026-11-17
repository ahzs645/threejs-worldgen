import * as THREE from 'three/webgpu'
import { FIXED_INSTANCE_STRIDE, hash, type ScatterLayer } from 'threejs-scatter'

/** Fixed per-type cards are spatially tiled, independently of the streamed near stand. */
export function farStandTiles(
  impostor: NonNullable<ScatterLayer['impostor']>,
  list: number[],
  groundAt: (x: number, y: number) => number,
  name: string
): THREE.Group {
  const tiles = new Map<string, number[]>()
  for (let o = 0; o < list.length; o += FIXED_INSTANCE_STRIDE) {
    const key = `${Math.floor(list[o] / 500)},${Math.floor(list[o + 1] / 500)}`
    let members = tiles.get(key)
    if (!members) tiles.set(key, (members = []))
    members.push(o)
  }
  const source = impostor.geometry
  source.computeBoundingBox()
  const box = source.boundingBox!
  const reach = Math.max(Math.abs(box.min.x), Math.abs(box.max.x), Math.abs(box.min.z), Math.abs(box.max.z))
  const group = new THREE.Group()
  group.name = `${name}:far`
  for (const [key, members] of tiles) {
    const position = new Float32Array(members.length * 4)
    const params = new Float32Array(members.length * 4)
    const bounds = new THREE.Box3()
    for (let n = 0; n < members.length; n++) {
      const o = members[n]
      const [x, z, yaw, width, tall] = [list[o], list[o + 1], list[o + 2], list[o + 3], list[o + 4]]
      const y = groundAt(x, -z) - 0.05
      const u = hash(Math.floor(x * 1000), Math.floor(z * 1000), 0x5ca7) / 4294967296
      position.set([x, y, z, tall], n * 4)
      params.set([yaw, 1 + (u * 2 - 1) * 0.12, u, tall > 0 ? width / tall : 1], n * 4)
      // A billboard can rotate around its base. Include the entire quad's radial reach and vertical extent.
      const r = reach * Math.max(Math.abs(width), Math.abs(tall)) + 1
      bounds.expandByPoint(new THREE.Vector3(x - r, y + box.min.y * tall - r, z - r))
      bounds.expandByPoint(new THREE.Vector3(x + r, y + box.max.y * tall + r, z + r))
    }
    const geometry = new THREE.InstancedBufferGeometry()
    // Clone the small quad buffers: disposing a tile must never release the cached scatter's buffers.
    geometry.setIndex(source.index?.clone() ?? null)
    for (const [key, attribute] of Object.entries(source.attributes)) geometry.setAttribute(key, attribute.clone())
    geometry.setAttribute('scatterPosition', new THREE.InstancedBufferAttribute(position, 4))
    geometry.setAttribute('scatterParams', new THREE.InstancedBufferAttribute(params, 4))
    geometry.instanceCount = members.length
    geometry.boundingBox = bounds
    geometry.boundingSphere = bounds.getBoundingSphere(new THREE.Sphere())
    const mesh = new THREE.Mesh(geometry, impostor.material)
    mesh.name = `${name}:far ${key}`
    mesh.frustumCulled = true
    mesh.matrixAutoUpdate = false
    mesh.castShadow = false
    mesh.receiveShadow = false
    group.add(mesh)
  }
  return group
}

export function setFarCulling(object: THREE.Object3D, enabled: boolean): void {
  object.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) child.frustumCulled = enabled
  })
}

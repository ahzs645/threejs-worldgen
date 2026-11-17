import * as THREE from 'three/webgpu'

export interface TreatedTable {
  fields: readonly string[]
  kinds: readonly string[]
  count: number
  data: ArrayLike<number>
}
export interface TreatedStem {
  x: number
  y: number
  kind: string
  length: number
  diameter: number
  heading: number
}

/** Accept only finite, positive geometry. Positions and headings remain the package's own. */
export function treatedStems(table?: TreatedTable | null): TreatedStem[] {
  if (!table) return []
  const fields = ['x', 'y', 'kind', 'length', 'diameter', 'heading'].map((k) => table.fields.indexOf(k))
  if (fields.some((v) => v < 0)) return []
  const stems: TreatedStem[] = []
  for (let i = 0; i < table.count; i++) {
    const values = fields.map((k) => table.data[i * table.fields.length + k])
    if (!values.every(Number.isFinite) || values[3] <= 0 || values[4] <= 0) continue
    const [x, y, kind, length, diameter, heading] = values
    const name = table.kinds[kind]
    if (!['stump', 'felled', 'log', 'snag'].includes(name)) continue
    stems.push({ x, y, kind: name, length, diameter, heading })
  }
  return stems
}

/** Four kinds share trunk buffers/materials; felled crowns use one additional instanced draw. */
export function buildTreated(
  table: TreatedTable | undefined,
  groundAt: (x: number, y: number) => number,
  inWater: (x: number, y: number) => boolean
) {
  const object = new THREE.Group()
  object.name = 'prescribed treated stems'
  const stems = treatedStems(table).filter((stem) => !inWater(stem.x, stem.y))
  const wood = new THREE.MeshStandardNodeMaterial({ color: '#75634e', roughness: 0.95 })
  const dead = new THREE.MeshStandardNodeMaterial({ color: '#55504a', roughness: 0.95 })
  const foliage = new THREE.MeshStandardNodeMaterial({ color: '#615b3d', roughness: 1 })
  const trunk = new THREE.CylinderGeometry(0.36, 0.5, 1, 7, 1).translate(0, 0.5, 0)
  const crown = new THREE.ConeGeometry(0.5, 1, 7).translate(0, 0.5, 0)
  const matrix = new THREE.Matrix4()
  const q = new THREE.Quaternion()
  const axis = new THREE.Vector3(0, 1, 0)
  const point = new THREE.Vector3()
  const scale = new THREE.Vector3()
  const direction = new THREE.Vector3()
  for (const kind of ['stump', 'felled', 'log', 'snag']) {
    const rows = stems.filter((stem) => stem.kind === kind)
    if (!rows.length) continue
    const mesh = new THREE.InstancedMesh(trunk, kind === 'snag' ? dead : wood, rows.length)
    const crowns = kind === 'felled' ? new THREE.InstancedMesh(crown, foliage, rows.length) : null
    rows.forEach((stem, i) => {
      const lying = kind === 'felled' || kind === 'log'
      const angle = (stem.heading * Math.PI) / 180
      direction.set(lying ? Math.sin(angle) : 0, lying ? 0 : 1, lying ? -Math.cos(angle) : 0)
      q.setFromUnitVectors(axis, direction)
      point.set(stem.x, groundAt(stem.x, stem.y) + (lying ? stem.diameter * 0.5 : -0.02), -stem.y)
      scale.set(stem.diameter, stem.length, stem.diameter)
      matrix.compose(point, q, scale)
      mesh.setMatrixAt(i, matrix)
      if (crowns) {
        const length = stem.length * 0.55
        const width = Math.max(stem.diameter * 2, stem.length * 0.18)
        point.addScaledVector(direction, stem.length * 0.4)
        matrix.compose(point, q, scale.set(width, length, width))
        crowns.setMatrixAt(i, matrix)
      }
    })
    for (const drawn of crowns ? [mesh, crowns] : [mesh]) {
      drawn.name = `prescribed ${kind}${drawn === crowns ? ' crowns' : ''}`
      drawn.castShadow = drawn.receiveShadow = true
      drawn.computeBoundingBox()
      drawn.computeBoundingSphere()
      object.add(drawn)
    }
  }
  object.userData.stems = stems.length
  object.userData.provenance = 'package treated fields; felled crowns are an approximate retained crown'
  return {
    object,
    stems: stems.length,
    dispose() {
      object.removeFromParent()
      trunk.dispose()
      crown.dispose()
      wood.dispose()
      dead.dispose()
      foliage.dispose()
    }
  }
}

/** Prescribed partial cuts can coexist with ordinary clearcuts and their seeded ground components. */
export function mountTreated(
  parent: THREE.Group,
  table: TreatedTable | undefined,
  groundAt: (x: number, y: number) => number,
  inWater: (x: number, y: number) => boolean
) {
  const built = buildTreated(table, groundAt, inWater)
  parent.add(built.object)
  return built
}

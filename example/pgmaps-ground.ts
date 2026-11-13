// A PGMaps package's ground: the terrain with forest floor, harvested ground (soil, duff, dry grass) inside openings
// and exposed mineral soil on road scars, weathered (wet, snow) through threejs-weathering; stumps and slash on the cut
// ground; and the package's water bodies as threejs-water lakes, with their beds carved into the terrain.
//
// Package coordinates are local metres, x east, y north, z up; three.js is y up with north at −z, so a package point
// (x, y, z) is drawn at (x, z, −y).
import * as THREE from 'three/webgpu'
import {
  attribute,
  cameraPosition,
  distance,
  float,
  mix,
  mx_noise_float,
  positionWorld,
  smoothstep,
  vec3
} from 'three/tsl'
import { hash, mulberry32 } from 'threejs-scatter'
import { Lakes, WaterPrepass, type HydrologyResult } from 'threejs-water'
import { applyMaterialStatesTo, type MaterialStateUniforms } from 'threejs-weathering'
import { openingSeed, pileSeed } from './pgmaps-tree-bins'

type Ring = Array<[number, number]>

/** A water body: polygon rings in package metres (outer first, then holes) and the surface height. */
export interface WaterBody {
  rings: Ring[]
  /** Water surface, metres; `null` = the lowest ground along the outer ring. */
  elevationMeters: number | null
}

export interface GroundInput {
  terrain: {
    west: number
    south: number
    spacing: number
    columns: number
    rows: number
    heights: Array<number | null>
  }
  openings: Array<{ rings: Ring[] }>
  debris: { count: number; data: number[] }
  roadScars: Array<{ kind: 'cut' | 'fill'; ring: Ring }>
  water?: WaterBody[] | null
}

export function inRing(ring: Ring, x: number, y: number) {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i]
    const [xj, yj] = ring[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/** Distance from a point to a ring's edges. */
function edgeDistance(ring: Ring, x: number, y: number) {
  let best = Infinity
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [ax, ay] = ring[j]
    const [bx, by] = ring[i]
    const dx = bx - ax
    const dy = by - ay
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1)))
    best = Math.min(best, Math.hypot(ax + dx * t - x, ay + dy * t - y))
  }
  return best
}

/** A polygon with holes and its bounding box, for quick point tests. */
interface Area {
  rings: Ring[]
  box: [number, number, number, number]
  /** From the polygon's own outer ring (`openingSeed`), never its place in the package's list. */
  seed: number
}
const areaOf = (rings: Ring[]): Area => {
  const box: Area['box'] = [Infinity, Infinity, -Infinity, -Infinity]
  for (const [x, y] of rings[0] ?? []) {
    box[0] = Math.min(box[0], x)
    box[1] = Math.min(box[1], y)
    box[2] = Math.max(box[2], x)
    box[3] = Math.max(box[3], y)
  }
  return { rings, box, seed: openingSeed(rings) }
}
const inArea = (a: Area, x: number, y: number) =>
  x >= a.box[0] &&
  x <= a.box[2] &&
  y >= a.box[1] &&
  y <= a.box[3] &&
  inRing(a.rings[0], x, y) &&
  !a.rings.slice(1).some((r) => inRing(r, x, y))

/** Water bodies with their surface level resolved. */
interface Pond extends Area {
  level: number
}

export interface Ground {
  /** Ground height at a package point (x east, y north), with lake beds carved. */
  groundAt(x: number, y: number): number
  /** The terrain meshes, stumps and slash. */
  group: THREE.Group
  /** The water, or null. Add with the group; call `update` per frame. */
  water: { lakes: Lakes; prepass: WaterPrepass; box: THREE.Box3 } | null
  /** Whether a package point is inside a harvest opening. */
  isCut(x: number, y: number): boolean
  /** Whether a package point is under water (no stems, stumps or slash there). */
  inWater(x: number, y: number): boolean
  dispose(): void
}

/** Builds the ground of a package. `states` are the atmosphere's wetness and snow (shared, live). */
export async function buildGround(
  pkg: GroundInput,
  states: MaterialStateUniforms,
  options: { renderer: THREE.WebGPURenderer; camera: THREE.Camera; sun: THREE.DirectionalLight; treeRadius: number }
): Promise<Ground> {
  const { west, south, spacing, columns, rows } = pkg.terrain
  // Gaps (no source DEM there) take the nearest known height along their row, then their column, so the mesh stays
  // whole; the trees and the eye are all inside the known area.
  const heights = pkg.terrain.heights.slice()
  const fill = (index: (i: number, j: number) => number, outer: number, inner: number) => {
    for (let i = 0; i < outer; i++) {
      let last: number | null = null
      for (let j = 0; j < inner; j++) last = heights[index(i, j)] ?? last ?? null
      for (let j = inner - 1; j >= 0; j--) {
        const k = index(i, j)
        if (heights[k] === null) heights[k] = last
        else last = heights[k]
      }
    }
  }
  fill((r, c) => r * columns + c, rows, columns)
  fill((c, r) => r * columns + c, columns, rows)
  // A loop, not Math.min(...): spreading a 160,000-cell grid overflows the call stack.
  let base = Infinity
  for (const h of heights) if (h !== null && h < base) base = h
  const at = (column: number, row: number) => heights[row * columns + column] ?? base
  /** The source terrain at a package point, bilinear. */
  const terrainAt = (x: number, y: number): number => {
    const gx = Math.min(columns - 1.0001, Math.max(0, (x - west) / spacing))
    const gy = Math.min(rows - 1.0001, Math.max(0, (y - south) / spacing))
    const x0 = Math.floor(gx)
    const y0 = Math.floor(gy)
    const fx = gx - x0
    const fy = gy - y0
    return (
      at(x0, y0) * (1 - fx) * (1 - fy) +
      at(x0 + 1, y0) * fx * (1 - fy) +
      at(x0, y0 + 1) * (1 - fx) * fy +
      at(x0 + 1, y0 + 1) * fx * fy
    )
  }

  // Water: each body's level, and a bed carved under it that deepens away from the shore (so the surface never
  // z-fights a DEM that was flattened to the lake, and the water has depth to absorb light through).
  const ponds: Pond[] = []
  for (const body of pkg.water ?? []) {
    if (!body?.rings?.[0] || body.rings[0].length < 3) continue
    const area = areaOf(body.rings)
    let level = body.elevationMeters ?? Infinity
    if (!Number.isFinite(level)) {
      const outer = body.rings[0]
      for (let i = 0; i < outer.length; i++) {
        const [ax, ay] = outer[i]
        const [bx, by] = outer[(i + 1) % outer.length]
        const n = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 2))
        for (let k = 0; k < n; k++)
          level = Math.min(level, terrainAt(ax + ((bx - ax) * k) / n, ay + ((by - ay) * k) / n))
      }
    }
    if (Number.isFinite(level)) ponds.push({ ...area, level })
  }
  const carve = (x: number, y: number, h: number) => {
    for (const p of ponds) {
      if (!inArea(p, x, y)) continue
      const shore = Math.min(...p.rings.map((r) => edgeDistance(r, x, y)))
      h = Math.min(h, p.level - Math.min(4.5, 0.35 + shore * 0.09))
    }
    return h
  }
  const groundAt = ponds.length ? (x: number, y: number) => carve(x, y, terrainAt(x, y)) : terrainAt

  const openings = pkg.openings.filter((o) => o.rings?.[0]?.length >= 3).map((o) => areaOf(o.rings))
  const scars = pkg.roadScars.filter((s) => s.ring?.length >= 3).map((s) => ({ ...areaOf([s.ring]), kind: s.kind }))
  const isCut = (x: number, y: number) => openings.some((a) => inArea(a, x, y))
  const inWater = (x: number, y: number) => ponds.some((p) => inArea(p, x, y))

  const material = groundMaterial(states, options.treeRadius)
  const group = new THREE.Group()
  /** A terrain mesh `size` metres across centred on the package origin, `step` metres between vertices. */
  const mesh = (size: number, step: number) => {
    const n = Math.round(size / step)
    const geometry = new THREE.PlaneGeometry(size, size, n, n).rotateX(-Math.PI / 2)
    const position = geometry.attributes.position
    const kind = new Float32Array(position.count * 4)
    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i)
      const y = -position.getZ(i)
      position.setY(i, groundAt(x, y))
      const cut = isCut(x, y) ? 1 : 0
      const scar = scars.find((s) => inArea(s, x, y))
      // x: harvested, y: road cut bank, z: road fill, w: open to the sky (snow lies deeper there than under canopy)
      kind[i * 4] = cut
      kind[i * 4 + 1] = scar?.kind === 'cut' ? 1 : 0
      kind[i * 4 + 2] = scar?.kind === 'fill' ? 1 : 0
      kind[i * 4 + 3] = cut || scar ? 1 : 0.55
    }
    geometry.setAttribute('aGround', new THREE.BufferAttribute(kind, 4))
    geometry.computeVertexNormals()
    const m = new THREE.Mesh(geometry, material)
    m.receiveShadow = true
    return m
  }
  const near = mesh(1400, 5)
  const far = mesh(Math.min(columns, rows) * spacing - spacing, spacing * 2)
  far.position.y = -0.3 // under the near mesh where they overlap
  const apron = horizonApron(Math.min(columns, rows) * spacing - spacing, groundAt, material)
  group.add(near, far, apron)

  const stumps = stumpsOf(openings, groundAt, ponds)
  const slash = slashOf(pkg.debris, openings, groundAt, inWater)
  // Wood takes the same weather as the ground: dark and glossy when wet, snow on whatever faces up.
  for (const m of [stumps, ...slash])
    applyMaterialStatesTo(
      m.material as THREE.MeshStandardNodeMaterial,
      { wetness: states.wetness, snow: states.snow.mul(0.6) },
      { streaks: false, puddles: false }
    )
  group.add(stumps, ...slash)

  let water: Ground['water'] = null
  if (ponds.length) {
    water = await lakesOf(ponds, options)
    group.add(water.lakes.object)
  }

  return {
    groundAt,
    group,
    water,
    isCut,
    inWater,
    dispose() {
      group.removeFromParent()
      group.traverse((o) => {
        const m = o as THREE.Mesh
        if (m.isMesh) m.geometry.dispose()
      })
      material.dispose()
      for (const m of [stumps, ...slash]) (m.material as THREE.Material).dispose()
      if (water) {
        water.lakes.dispose()
        water.prepass.dispose()
      }
    }
  }
}

/**
 * The ground's look: forest floor (needle litter and moss) under the stand, fading to the colour of distant forest
 * canopy beyond the drawn trees; soil, duff, dry grass and some green regrowth inside openings; pale mineral soil on
 * road cut banks and darker fill. Fine noise fades with distance so it never shimmers. Wetness (puddles) and snow
 * (deeper in the open) come from the shared state uniforms.
 */
function groundMaterial(states: MaterialStateUniforms, treeRadius: number) {
  const kind = attribute('aGround', 'vec4')
  const p = positionWorld.xz
  const d = distance(positionWorld, cameraPosition)
  const fine = float(1).sub(smoothstep(60, 220, d))
  const n1 = mx_noise_float(p.mul(0.045))
  const n2 = mx_noise_float(p.mul(0.33))
  const n3 = mx_noise_float(p.mul(2.3)).mul(fine)
  const n4 = mx_noise_float(p.mul(0.012))

  // Under the stand: brown needle litter with mossy patches.
  const litter = vec3(0.095, 0.07, 0.044)
  const moss = vec3(0.06, 0.085, 0.032)
  const floor = mix(litter, moss, smoothstep(-0.25, 0.45, n1.add(n2.mul(0.4)))).mul(n3.mul(0.18).add(1))
  // Beyond the drawn trees the ground stands for the forest itself, seen from afar: dark canopy in a stand mosaic.
  const canopy = mix(vec3(0.028, 0.045, 0.024), vec3(0.05, 0.065, 0.032), smoothstep(-0.4, 0.5, n4))
  const forest = mix(floor, canopy, smoothstep(treeRadius * 0.55, treeRadius * 0.95, d))

  // Harvested: grey-brown duff and churned soil, straw-coloured dry grass, a little green regrowth, darker ruts.
  const duff = vec3(0.14, 0.11, 0.08)
  const soil = vec3(0.2, 0.14, 0.085)
  const grass = vec3(0.26, 0.22, 0.12)
  const regrowth = vec3(0.09, 0.115, 0.04)
  let cut = mix(duff, soil, smoothstep(-0.1, 0.5, n2.add(n3.mul(0.3))))
  cut = mix(cut, grass, smoothstep(0.15, 0.55, n1.mul(0.8).add(n2.mul(0.3))).mul(0.7))
  cut = mix(cut, regrowth, smoothstep(0.35, 0.7, n2.mul(-1).add(n3.mul(0.4))).mul(0.6))
  cut = cut.mul(n3.mul(0.22).add(1))

  // Road scars: pale gravelly cut banks, darker loose fill.
  const bank = vec3(0.3, 0.255, 0.19).mul(n3.mul(0.25).add(1))
  const fillSoil = vec3(0.19, 0.15, 0.1).mul(n3.mul(0.2).add(1))

  let color = mix(forest, cut, kind.x)
  color = mix(color, bank, kind.y)
  color = mix(color, fillSoil, kind.z)
  const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.96 })
  material.colorNode = color
  applyMaterialStatesTo(
    material,
    { wetness: states.wetness, snow: states.snow.mul(kind.w) },
    { streaks: false, puddles: true }
  )
  return material
}

/** A ring of ground from the terrain's edge out to the far plane, sloping to its mean height and the earth's curve. */
function horizonApron(size: number, groundAt: (x: number, y: number) => number, material: THREE.Material) {
  const half = size / 2 - 1
  const segments = 160
  const rings = [0, 0.04, 0.12, 0.25, 0.45, 0.7, 1]
  const outer = 7600
  const positions: number[] = []
  const kind: number[] = []
  const edge: number[] = []
  for (let s = 0; s < segments; s++) {
    const a = (s / segments) * Math.PI * 2
    const c = Math.cos(a)
    const n = Math.sin(a)
    const k = half / Math.max(Math.abs(c), Math.abs(n))
    edge.push(groundAt(c * k, n * k))
  }
  const mean = edge.reduce((a, b) => a + b, 0) / edge.length
  for (const t of rings) {
    for (let s = 0; s < segments; s++) {
      const a = (s / segments) * Math.PI * 2
      const c = Math.cos(a)
      const n = Math.sin(a)
      const r0 = half / Math.max(Math.abs(c), Math.abs(n))
      const r = r0 + (outer - r0) * t
      const h = edge[s] + (mean - edge[s]) * Math.min(1, t * 1.6) - (r * r - r0 * r0) / (2 * 6_371_000) - 0.6
      positions.push(c * r, h, -n * r)
      kind.push(0, 0, 0, 0.55)
    }
  }
  const index: number[] = []
  for (let j = 0; j < rings.length - 1; j++)
    for (let s = 0; s < segments; s++) {
      const a = j * segments + s
      const b = j * segments + ((s + 1) % segments)
      index.push(a, b, a + segments, b, b + segments, a + segments)
    }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setAttribute('aGround', new THREE.Float32BufferAttribute(kind, 4))
  geometry.setIndex(index)
  geometry.computeVertexNormals()
  // Wound to face up whichever way the angles ran.
  if (geometry.attributes.normal.getY(0) < 0) {
    index.reverse()
    geometry.setIndex(index)
    geometry.computeVertexNormals()
  }
  const m = new THREE.Mesh(geometry, material)
  m.receiveShadow = false
  return m
}

const random = (seed: number) => mulberry32(hash(seed, 0x51a5))

/**
 * Stumps across every opening: a jittered grid at about 500 stems a hectare (a harvested pine/spruce stand). Each
 * opening's stumps are seeded from its own geometry, so the list's order does not move them.
 */
function stumpsOf(openings: Area[], groundAt: (x: number, y: number) => number, ponds: Area[]) {
  const geometry = new THREE.CylinderGeometry(0.47, 0.55, 1, 10, 1, false)
  geometry.translate(0, 0.5, 0)
  // Cut face pale and fresh-sawn, sides bark; colour per vertex so the instance tint keeps both.
  const pos = geometry.attributes.position
  const normal = geometry.attributes.normal
  const colors = new Float32Array(pos.count * 3)
  const face = new THREE.Color('#bba07a')
  const bark = new THREE.Color('#6b5747')
  for (let i = 0; i < pos.count; i++) (normal.getY(i) > 0.5 ? face : bark).toArray(colors, i * 3)
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  const matrices: THREE.Matrix4[] = []
  const tints: THREE.Color[] = []
  const m = new THREE.Matrix4()
  const q = new THREE.Quaternion()
  const e = new THREE.Euler()
  const step = 4.4
  for (const a of openings) {
    const rand = random(a.seed)
    for (let x = a.box[0]; x <= a.box[2]; x += step)
      for (let y = a.box[1]; y <= a.box[3]; y += step) {
        const px = x + (rand() - 0.5) * step * 0.9
        const py = y + (rand() - 0.5) * step * 0.9
        const r = 0.11 + rand() * 0.14
        const h = 0.18 + rand() * 0.3
        const tilt = (rand() - 0.5) * 0.12
        const yaw = rand() * Math.PI * 2
        const tone = 0.8 + rand() * 0.35
        if (rand() < 0.12 || !inArea(a, px, py) || ponds.some((p) => inArea(p, px, py))) continue
        e.set(tilt, yaw, tilt * 0.7)
        m.compose(
          new THREE.Vector3(px, groundAt(px, py) - 0.08, -py),
          q.setFromEuler(e),
          new THREE.Vector3(r * 2, h, r * 2)
        )
        matrices.push(m.clone())
        tints.push(new THREE.Color(tone, tone * 0.98, tone * 0.95))
        if (matrices.length >= 40_000) break
      }
  }
  const material = new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.9 })
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, matrices.length))
  mesh.count = matrices.length
  matrices.forEach((matrix, i) => {
    mesh.setMatrixAt(i, matrix)
    mesh.setColorAt(i, tints[i])
  })
  mesh.castShadow = mesh.receiveShadow = true
  mesh.name = 'stumps'
  return mesh
}

/**
 * Slash on the cut ground. Each package debris pile becomes a heap of logs lying roughly together (some propped on the
 * others), tangled branches and rust-brown dead foliage; branches are also strewn thinly over the whole opening.
 */
function slashOf(
  debris: GroundInput['debris'],
  openings: Area[],
  groundAt: (x: number, y: number) => number,
  inWater: (x: number, y: number) => boolean
) {
  const logs: number[] = []
  const twigs: number[] = []
  const tufts: number[] = []
  const push = (
    list: number[],
    x: number,
    y: number,
    z: number,
    yaw: number,
    pitch: number,
    roll: number,
    sx: number,
    sy: number,
    sz: number
  ) => list.push(x, y, z, yaw, pitch, roll, sx, sy, sz)
  for (let i = 0; i < debris.count; i++) {
    const [x, y, , footprint] = debris.data.slice(i * 4, i * 4 + 4)
    if (inWater(x, y)) continue
    const rand = random(pileSeed(x, y, footprint))
    const heading = rand() * Math.PI
    const span = Math.max(1.5, footprint)
    const count = 5 + Math.floor(rand() * 6)
    for (let k = 0; k < count; k++) {
      const px = x + (rand() - 0.5) * span * 0.6
      const py = y + (rand() - 0.5) * span * 0.6
      const length = span * (0.45 + rand() * 0.55)
      const r = 0.04 + rand() * 0.09
      const lift = r + rand() * Math.min(0.6, span * 0.12)
      const yaw = heading + (rand() - 0.5) * 0.9
      push(logs, px, groundAt(px, py) + lift, -py, yaw, (rand() - 0.5) * 0.25, 0, r * 2, length, r * 2)
    }
    for (let k = 0; k < 26; k++) {
      const px = x + (rand() - 0.5) * span * 0.8
      const py = y + (rand() - 0.5) * span * 0.8
      const length = 0.8 + rand() * span * 0.45
      push(
        twigs,
        px,
        groundAt(px, py) + 0.1 + rand() * 0.55,
        -py,
        rand() * Math.PI * 2,
        (rand() - 0.5) * 0.9,
        0,
        0.05,
        length,
        0.05
      )
    }
    for (let k = 0; k < 4; k++) {
      const px = x + (rand() - 0.5) * span * 0.6
      const py = y + (rand() - 0.5) * span * 0.6
      const w = span * (0.18 + rand() * 0.2)
      push(
        tufts,
        px,
        groundAt(px, py) + 0.05,
        -py,
        rand() * Math.PI,
        0,
        0,
        w,
        0.25 + rand() * 0.35,
        w * (0.6 + rand() * 0.5)
      )
    }
  }
  // Branches strewn over the openings, a few hundred to the hectare.
  for (const a of openings) {
    const rand = random(a.seed ^ 0x5a5a)
    const areaM2 = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1])
    const n = Math.min(30_000, Math.round(areaM2 / 30))
    for (let i = 0; i < n; i++) {
      const px = a.box[0] + rand() * (a.box[2] - a.box[0])
      const py = a.box[1] + rand() * (a.box[3] - a.box[1])
      const length = 0.6 + rand() * 2.2
      const yaw = rand() * Math.PI * 2
      if (!inArea(a, px, py)) continue
      push(twigs, px, groundAt(px, py) + 0.04, -py, yaw, (rand() - 0.5) * 0.15, 0, 0.04 + rand() * 0.04, length, 0.04)
    }
  }
  const instanced = (
    geometry: THREE.BufferGeometry,
    color: string,
    list: number[],
    lay: boolean,
    variation: number,
    name: string
  ) => {
    const count = list.length / 9
    const material = new THREE.MeshStandardNodeMaterial({ color, roughness: 0.95 })
    const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, count))
    mesh.count = count
    const m = new THREE.Matrix4()
    const q = new THREE.Quaternion()
    const e = new THREE.Euler()
    const c = new THREE.Color()
    for (let i = 0; i < count; i++) {
      const o = i * 9
      // Tint from the piece's own position, not its index: a pile or opening gone from the list recolours nothing else.
      const rand = random(hash(Math.floor(list[o] * 1000), Math.floor(list[o + 2] * 1000), 0x7a11))
      // Logs and twigs lie along their length: the cylinder's y axis turned down to horizontal, then yawed.
      e.set(0, list[o + 3], lay ? Math.PI / 2 + list[o + 4] : 0, 'YXZ')
      m.compose(
        new THREE.Vector3(list[o], list[o + 1], list[o + 2]),
        q.setFromEuler(e),
        new THREE.Vector3(list[o + 6], list[o + 7], list[o + 8])
      )
      mesh.setMatrixAt(i, m)
      const t = 1 + (rand() - 0.5) * variation
      mesh.setColorAt(i, c.setRGB(t, t * (1 - (rand() - 0.5) * 0.08), t * 0.97))
    }
    mesh.castShadow = mesh.receiveShadow = true
    mesh.name = name
    return mesh
  }
  const log = new THREE.CylinderGeometry(0.5, 0.5, 1, 7, 1)
  const twig = new THREE.CylinderGeometry(0.3, 0.5, 1, 4, 1)
  const tuft = new THREE.IcosahedronGeometry(0.5, 1)
  {
    // Lumpy, flattened clumps of dead needles and fine branches.
    const p = tuft.attributes.position
    const rand = random(77)
    for (let i = 0; i < p.count; i++) {
      const s = 0.75 + rand() * 0.5
      p.setXYZ(i, p.getX(i) * s, Math.max(-0.1, p.getY(i)) * s, p.getZ(i) * s)
    }
    tuft.computeVertexNormals()
  }
  return [
    instanced(log, '#8a7c69', logs, true, 0.35, 'slash logs'),
    instanced(twig, '#6e5d4c', twigs, true, 0.4, 'slash branches'),
    instanced(tuft, '#6a4a30', tufts, false, 0.45, 'slash foliage')
  ]
}

/** The package's water bodies as threejs-water lakes, rasterised into a lake mask at a cell size that suits them. */
async function lakesOf(
  ponds: Pond[],
  o: { renderer: THREE.WebGPURenderer; camera: THREE.Camera; sun: THREE.DirectionalLight }
): Promise<NonNullable<Ground['water']>> {
  const box = [Infinity, Infinity, -Infinity, -Infinity]
  for (const p of ponds) {
    box[0] = Math.min(box[0], p.box[0])
    box[1] = Math.min(box[1], p.box[1])
    box[2] = Math.max(box[2], p.box[2])
    box[3] = Math.max(box[3], p.box[3])
  }
  const biggest = Math.max(...ponds.map((p) => Math.max(p.box[2] - p.box[0], p.box[3] - p.box[1])))
  const cellSize = Math.max(1.5, biggest / 400)
  const margin = cellSize * 3
  // Grid in three.js x/z: x east, z south. Row r is z = z0 + r · cellSize.
  const x0 = box[0] - margin
  const z0 = -box[3] - margin
  const cols = Math.ceil((box[2] - box[0] + 2 * margin) / cellSize) + 1
  const rows = Math.ceil((box[3] - box[1] + 2 * margin) / cellSize) + 1
  const lakes: HydrologyResult['lakes'] = []
  const lakeIndex = new Int32Array(cols * rows).fill(-1)
  ponds.forEach((p, li) => {
    const c0 = Math.max(0, Math.floor((p.box[0] - x0) / cellSize) - 1)
    const c1 = Math.min(cols - 1, Math.ceil((p.box[2] - x0) / cellSize) + 1)
    const r0 = Math.max(0, Math.floor((-p.box[3] - z0) / cellSize) - 1)
    const r1 = Math.min(rows - 1, Math.ceil((-p.box[1] - z0) / cellSize) + 1)
    const bw = c1 - c0 + 1
    const bh = r1 - r0 + 1
    const mask = new Uint8Array(bw * bh)
    let area = 0
    for (let r = 0; r < bh; r++)
      for (let c = 0; c < bw; c++) {
        const x = x0 + (c0 + c) * cellSize
        const y = -(z0 + (r0 + r) * cellSize)
        if (!inArea(p, x, y)) continue
        mask[r * bw + c] = 1
        lakeIndex[(r0 + r) * cols + c0 + c] = li
        area += cellSize * cellSize
      }
    lakes.push({ level: p.level, area, maxDepth: 4.5, bounds: [c0, r0, bw, bh], mask })
  })
  const hydrology = {
    grid: { cols, rows, cellSize, x0, z0, heights: new Float64Array(0) },
    lakes,
    lakeIndex,
    rivers: []
  } as unknown as HydrologyResult
  const prepass = new WaterPrepass({ camera: o.camera, resolutionScale: 0.5 })
  const lakes3 = await Lakes.create({
    hydrology,
    camera: o.camera,
    renderer: o.renderer,
    prepass,
    sun: o.sun,
    rivers: false,
    // Each lake is extended a few cells past its shore, under the banks; the waterline itself is where the carved bed
    // rises through the surface.
    expand: 3,
    // Northern lake water: tea-brown tannins and green, darker than the tropical default.
    absorption: [0.75, 0.42, 0.32],
    scatterColor: '#0f2a26',
    waves: { amplitude: 0.02, wavelength: [0.25, 2.5] },
    roughness: 0.05,
    // No foam: a forest lake does not break at its shore, and the contact foam (drawn wherever geometry stands in
    // front of the surface) outlines every stem seen against the water.
    foam: 0
  })
  const water = new THREE.Box3()
  for (const p of ponds) {
    water.expandByPoint(new THREE.Vector3(p.box[0], p.level - 5, -p.box[3]))
    water.expandByPoint(new THREE.Vector3(p.box[2], p.level + 1, -p.box[1]))
  }
  return { lakes: lakes3, prepass, box: water }
}

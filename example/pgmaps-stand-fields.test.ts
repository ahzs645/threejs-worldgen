import { test } from 'node:test'
import assert from 'node:assert/strict'
import * as THREE from 'three/webgpu'
import { morphology, standSampler, stemMetadata, standMetadataSummary, type StandGrid } from './pgmaps-stand-fields.ts'
import { treatedStems, mountTreated } from './pgmaps-treated.ts'
import { farStandTiles, setFarCulling } from './pgmaps-far-cards.ts'
import { recipe } from './pgmaps-tree-recipes.ts'
import { rasterTree, SPECIES_ORDER, speciesSlot } from './pgmaps-far-forest-math.ts'

test('legacy fields, zero live crown base, stable IDs and expansion metadata', () => {
  const legacy = {
    fields: ['x', 'y', 'z', 'species', 'height', 'crownWidth', 'tone'],
    species: ['pine'],
    count: 1,
    data: [3, 4, 8, 0, 20, 8, 0.5]
  }
  assert.deepEqual(stemMetadata(legacy, 0), {
    id: 'geo:3.000:4.000',
    crownBase: -1,
    dbhCm: 0,
    expansion: 1,
    form: 0,
    source: 0
  })
  const table = {
    ...legacy,
    fields: [...legacy.fields, 'crownBase', 'dbhCm', 'expansion', 'form', 'source'],
    data: [...legacy.data, 0, 40, 12, 2, 2],
    ids: ['inventory-17']
  }
  const meta = stemMetadata(table, 0)
  assert.equal(meta.id, 'inventory-17')
  assert.equal(meta.crownBase, 0)
  assert.equal(meta.expansion, 12)
  assert.deepEqual(standMetadataSummary(table), {
    records: 1,
    representedStems: 12,
    measuredDbh: 1,
    knownCrownBases: 1,
    providedIds: 1,
    forms: [0, 0, 1],
    sources: [0, 0, 1]
  })
  const grown = morphology(meta, 20, 0.5)
  assert.equal(grown.crownBase, 0)
  assert.equal(grown.form, 2)
  assert.ok(morphology({ ...meta, crownBase: 0.95 }, 20, 0.5).crownBase < 1)
  assert.ok(grown.trunkRatio! > 0)
  const unknown = morphology(stemMetadata(legacy, 0), 20, 0.35)
  assert.equal(unknown.crownBase, 0.35)
  assert.equal(unknown.trunkRatio, undefined)
})

test('stand palettes choose the finest known grid and fall through unknown cells', () => {
  const grid = (spacing: number, species: string): StandGrid => ({
    west: 0,
    south: 0,
    spacing,
    columns: 2,
    rows: 2,
    stand: {
      cells: [0, 0, 0, 0],
      records: [{ heightMeters: 14, species: [{ species, share: 1 }], source: 'inventory' }]
    }
  })
  const fine = grid(10, 'hemlock')
  fine.stand!.cells[0] = -1
  const at = standSampler([grid(100, 'cottonwood'), fine])
  assert.equal(at(0, 0)?.species[0].species, 'cottonwood')
  assert.equal(at(10, 10)?.species[0].species, 'hemlock')
  assert.equal(at(50, 50)?.source, 'inventory')
  assert.equal(at(-1, 0), null)
  assert.equal(at(101, 0), null)
})

test('nine species preserve original slots and all create visible cards and recipes', () => {
  assert.deepEqual(SPECIES_ORDER.slice(0, 6), ['pine', 'spruce', 'fir', 'douglasFir', 'aspen', 'birch'])
  assert.equal(speciesSlot('hemlock'), 6)
  assert.equal(speciesSlot('blackSpruce'), 7)
  assert.equal(speciesSlot('cottonwood'), 8)
  for (const species of SPECIES_ORDER) {
    assert.ok(rasterTree(species, 32, 64, 0.6, 7, 2).data.some((value, i) => i % 4 === 3 && value > 0))
    assert.equal(recipe(species, 20, 5, 0.2).height, 20)
  }
  assert.ok(recipe('cottonwood', 20, 5, 0.2).crownShape === 'ellipsoid')
  assert.equal(recipe('pine', 4, 2, 0.2).baseRadius, 0.12)
  assert.equal(recipe('pine', 20, 5, 0, 0.02, 2).baseRadius, 0.4)
})

test('treated stems draw once per record, retain heading and have explicit conservative bounds', () => {
  const table = {
    fields: ['x', 'y', 'z', 'kind', 'length', 'diameter', 'heading'],
    kinds: ['stump', 'felled', 'log', 'snag'],
    count: 4,
    data: [0, 0, 0, 0, 0.4, 0.3, 0, 10, 0, 0, 1, 20, 0.4, 90, 0, 10, 0, 2, 10, 0.5, 0, 5, 5, 0, 3, 12, 0.4, 0]
  }
  assert.equal(treatedStems(table).length, 4)
  assert.equal(treatedStems(table)[1].heading, 90)
  const ground = new THREE.Group()
  const ordinaryClearcutStumps = new THREE.Group()
  ordinaryClearcutStumps.name = 'stumps'
  ground.add(ordinaryClearcutStumps)
  const built = mountTreated(
    ground,
    table,
    () => 2,
    () => false
  )
  assert.equal(
    ordinaryClearcutStumps.visible,
    true,
    'prescribed partial-cut stems keep another ordinary clearcut intact'
  )
  assert.equal(ordinaryClearcutStumps.parent, ground)
  assert.equal(built.stems, 4)
  assert.equal(built.object.children.length, 5)
  for (const child of built.object.children) assert.ok((child as THREE.InstancedMesh).boundingSphere!.radius > 0)
  const log = built.object.children.find((child) => child.name === 'prescribed log') as THREE.InstancedMesh
  const matrix = new THREE.Matrix4()
  log.getMatrixAt(0, matrix)
  const tip = new THREE.Vector3(0, 1, 0).applyMatrix4(matrix)
  assert.ok(Math.abs(tip.z + 20) < 1e-5, 'north heading places log tip 10 m north from y=10')
  built.dispose()
  assert.equal(ground.children.length, 1)
  assert.equal(ordinaryClearcutStumps.visible, true)
})

test('fixed far cards tile and own quad buffers; culling includes full rotated crown and height', () => {
  const geometry = new THREE.PlaneGeometry(2, 20).translate(0, 10, 0)
  const material = new THREE.MeshBasicNodeMaterial()
  const group = farStandTiles(
    { geometry, material, distance: 500 },
    [0, 0, 0, 4, 2, 1000, 1000, 0, 1, 1],
    () => 0,
    'test'
  )
  assert.equal(group.children.length, 2)
  const first = group.children[0] as THREE.Mesh
  assert.notEqual(first.geometry.getAttribute('position'), geometry.getAttribute('position'))
  assert.ok(first.geometry.boundingSphere!.containsPoint(new THREE.Vector3(4, 40, 0)))
  setFarCulling(group, false)
  assert.ok(group.children.every((child) => !child.frustumCulled))
  group.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) (child as THREE.Mesh).geometry.dispose()
  })
  geometry.dispose()
  material.dispose()
})

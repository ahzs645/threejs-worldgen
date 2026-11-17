/** Additive scene-v1 stand fields; reads legacy seven-field tables too. No geometry or assessment changes. */
export interface StandTable {
  fields: readonly string[]
  species: readonly string[]
  count: number
  data: ArrayLike<number>
  ids?: readonly string[]
}
export interface StemMetadata {
  id: string
  crownBase: number
  dbhCm: number
  expansion: number
  form: number
  source: number
}

export function stemMetadata(table: StandTable, i: number): StemMetadata {
  const o = i * table.fields.length
  const read = (name: string, fallback: number) => {
    const k = table.fields.indexOf(name)
    const v = k < 0 ? NaN : table.data[o + k]
    return Number.isFinite(v) ? v : fallback
  }
  return {
    id: table.ids?.[i] ?? `geo:${read('x', 0).toFixed(3)}:${read('y', 0).toFixed(3)}`,
    crownBase: Math.max(-1, Math.min(0.95, read('crownBase', -1))),
    dbhCm: Math.max(0, read('dbhCm', 0)),
    expansion: Math.max(1, read('expansion', 1)),
    form: Math.round(Math.max(0, Math.min(2, read('form', 0)))),
    source: Math.round(Math.max(0, Math.min(2, read('source', 0))))
  }
}

/** Inventory expansion describes represented stems; it never duplicates visual records or changes screening. */
export function standMetadataSummary(table: StandTable) {
  const [expansion, dbh, crownBase, form, source] = ['expansion', 'dbhCm', 'crownBase', 'form', 'source'].map((k) =>
    table.fields.indexOf(k)
  )
  const summary = {
    records: table.count,
    representedStems: 0,
    measuredDbh: 0,
    knownCrownBases: 0,
    providedIds: Math.min(table.count, table.ids?.length ?? 0),
    forms: [0, 0, 0],
    sources: [0, 0, 0]
  }
  for (let i = 0; i < table.count; i++) {
    const o = i * table.fields.length
    const value = (k: number, fallback: number) =>
      k >= 0 && Number.isFinite(table.data[o + k]) ? table.data[o + k] : fallback
    summary.representedStems += Math.max(1, value(expansion, 1))
    summary.measuredDbh += value(dbh, 0) > 0 ? 1 : 0
    summary.knownCrownBases += value(crownBase, -1) >= 0 ? 1 : 0
    summary.forms[Math.round(Math.max(0, Math.min(2, value(form, 0))))]++
    summary.sources[Math.round(Math.max(0, Math.min(2, value(source, 0))))]++
  }
  return summary
}

export interface StandRecord {
  heightMeters: number
  species: Array<{ species: string; share: number; heightMeters?: number }>
  source: 'regional' | 'inventory' | 'imported'
}
export interface StandGrid {
  west: number
  south: number
  spacing: number
  columns: number
  rows: number
  stand?: { cells: number[]; records: StandRecord[] }
}

/** Nearest stand post, finest supplied grid first. Unknown cells fall through to coarser grids. */
export function standSampler(grids: Array<StandGrid | null | undefined>) {
  const ordered = grids.filter((g): g is StandGrid => !!g?.stand && g.spacing > 0).sort((a, b) => a.spacing - b.spacing)
  return (x: number, y: number): StandRecord | null => {
    for (const g of ordered) {
      const c = Math.round((x - g.west) / g.spacing)
      const r = Math.round((y - g.south) / g.spacing)
      if (
        x < g.west ||
        y < g.south ||
        x > g.west + (g.columns - 1) * g.spacing ||
        y > g.south + (g.rows - 1) * g.spacing
      )
        continue
      const k = g.stand!.cells[r * g.columns + c]
      const record = k >= 0 ? g.stand!.records[k] : undefined
      if (
        record &&
        Number.isFinite(record.heightMeters) &&
        record.heightMeters > 0 &&
        record.species.some((s) => Number.isFinite(s.share) && s.share > 0)
      )
        return record
    }
    return null
  }
}

/** Bounded morphology classes keep imported crown bases/DBH and young/shrub forms practical to grow and cache. */
export function morphology(meta: StemMetadata, height: number, fallbackBase: number) {
  const base = meta.crownBase >= 0 ? meta.crownBase : meta.form === 2 ? 0.05 : meta.form === 1 ? 0.15 : fallbackBase
  const crownBase = Math.max(0, Math.min(0.95, meta.crownBase >= 0 ? Math.round(base * 5) / 5 : base))
  const radiusRatio = meta.dbhCm > 0 ? meta.dbhCm / 200 / Math.max(1, height) : 0.012
  const classes = [0.004, 0.008, 0.012, 0.02, 0.032, 0.05]
  const trunkRatio =
    meta.dbhCm > 0
      ? classes.reduce((a, b) => (Math.abs(b - radiusRatio) < Math.abs(a - radiusRatio) ? b : a))
      : undefined
  return { crownBase, trunkRatio, form: meta.form, key: `${crownBase}:${trunkRatio}:${meta.form}` }
}

// Writes the index of the PGMaps scene page's released versions that GitHub Pages serves.
//
//   node scripts/pgmaps-versions.mjs [dir=site/pgmaps]
//
// Each release of the page (tag `pgmaps-scene-v<semver>`, .github/workflows/pgmaps-release.yml) is unpacked by
// .github/workflows/pages.yml into <dir>/<version>/, carrying the `release.json` the release build wrote. This reads
// those and writes <dir>/versions.json: `{ latest, versions: [{ version, tag, commit, date, page }] }`, newest first.
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const TAG_PREFIX = 'pgmaps-scene-v'
const SEMVER = /^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$/

/** Semver order, newest first: higher numbers first, and a release before its own prereleases. */
export function compareVersionsDescending(a, b) {
  const [, ...pa] = a.match(SEMVER) ?? []
  const [, ...pb] = b.match(SEMVER) ?? []
  for (let i = 0; i < 3; i++) if (Number(pa[i]) !== Number(pb[i])) return Number(pb[i]) - Number(pa[i])
  if (!pa[3] !== !pb[3]) return pa[3] ? 1 : -1
  return (pb[3] ?? '').localeCompare(pa[3] ?? '', 'en', { numeric: true })
}

/** The versions index from each version's `release.json` (a directory without one, or not named as a semver, is skipped). */
export function versionsIndex(releases) {
  const versions = releases
    .filter((r) => SEMVER.test(r.version))
    .sort((a, b) => compareVersionsDescending(a.version, b.version))
    .map((r) => ({
      version: r.version,
      tag: r.tag ?? `${TAG_PREFIX}${r.version}`,
      commit: r.commit ?? null,
      date: r.date ?? null,
      page: `${r.version}/pgmaps-scene.html`
    }))
  return { latest: versions.find((v) => !v.version.includes('-'))?.version ?? versions[0]?.version ?? null, versions }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const dir = resolve(process.argv[2] ?? 'site/pgmaps')
  const releases = existsSync(dir)
    ? readdirSync(dir, { withFileTypes: true })
        .filter((e) => e.isDirectory() && existsSync(join(dir, e.name, 'release.json')))
        .map((e) => ({ ...JSON.parse(readFileSync(join(dir, e.name, 'release.json'), 'utf8')), version: e.name }))
    : []
  const index = versionsIndex(releases)
  writeFileSync(join(dir, 'versions.json'), `${JSON.stringify(index, null, 2)}\n`)
  console.log(`pgmaps-versions: ${index.versions.length} versions, latest ${index.latest ?? 'none'}`)
}

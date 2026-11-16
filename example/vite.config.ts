import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { readdirSync, readFileSync } from 'node:fs'
import { execSync } from 'node:child_process'

const src = (p: string) => fileURLToPath(new URL(`../packages/${p}`, import.meta.url))
const packages = [
  'heightfield',
  'biomes',
  'scatter',
  'trees',
  'water',
  'particles',
  'weathering',
  'fauna',
  'city',
  'props',
  'vehicles',
  'sky',
  'grass',
  'worldgen'
]
/**
 * What the PGMaps scene page reports as its capture reply's `version`: `pgmaps-scene <version>+<commit>`. A release
 * build (`.github/workflows/pgmaps-release.yml`) passes the tag's version in `PGMAPS_SCENE_VERSION`; any other build
 * reports the threejs-worldgen package version marked `-dev`. The commit is this checkout's, `-dirty` with uncommitted
 * changes to tracked files.
 */
function pgmapsSceneVersion(): string {
  const git = (args: string) =>
    execSync(`git ${args}`, {
      cwd: fileURLToPath(new URL('..', import.meta.url)),
      stdio: ['ignore', 'pipe', 'ignore']
    })
      .toString()
      .trim()
  let commit = 'unknown'
  try {
    commit = git('rev-parse --short HEAD') + (git('status --porcelain --untracked-files=no') ? '-dirty' : '')
  } catch {
    // not a git checkout
  }
  const pkg = JSON.parse(readFileSync(new URL('../packages/worldgen/package.json', import.meta.url), 'utf8'))
  return `pgmaps-scene ${process.env.PGMAPS_SCENE_VERSION || `${pkg.version}-dev`}+${commit}`
}

// `PGMAPS_SCENE_ONLY=1` builds the PGMaps scene page alone, without public/ (its textures are other demos', and local
// scene packages there must never ship): the release workflow zips that build.
const pgmapsOnly = process.env.PGMAPS_SCENE_ONLY === '1'

const name = (dir: string) => (dir === 'worldgen' ? 'threejs-worldgen' : `threejs-${dir}`)

export default defineConfig({
  plugins: [react()],
  define: { __WORLDGEN_VERSION__: JSON.stringify(pgmapsSceneVersion()) },
  ...(pgmapsOnly ? { publicDir: false as const } : {}),
  // Demos run straight from each package's src/, so edits show up without a build.
  resolve: {
    alias: [
      { find: /^threejs-biomes\/worker(\?worker)?$/, replacement: src('biomes/src/worker.ts') + '$1' },
      ...packages.flatMap((dir) => [
        { find: new RegExp(`^${name(dir)}/react$`), replacement: src(`${dir}/src/react.tsx`) },
        { find: new RegExp(`^${name(dir)}$`), replacement: src(`${dir}/src/index.ts`) }
      ])
    ]
  },
  // Reachable over the tailnet (tailscale serve) as well as localhost. 8460 is deliberately far from
  // Vite's 5173 so this server never collides with another project's dev server; strictPort fails
  // loudly rather than drifting to another port.
  server: { port: 8460, strictPort: true, allowedHosts: ['.ts.net', 'localhost', '127.0.0.1'] },
  worker: { format: 'es' },
  build: {
    target: 'esnext',
    rollupOptions: {
      input: pgmapsOnly
        ? ['pgmaps-scene.html']
        : readdirSync(fileURLToPath(new URL('.', import.meta.url))).filter((f) => f.endsWith('.html'))
    }
  }
})

// The PGMaps scene's sky, light, haze and weather: threejs-sky placed at the package's date, latitude and longitude,
// aerial perspective that follows that sky, and threejs-weathering's rain and snow with wet and snowy materials.
// Driven by the parent's `pgmaps-environment` message (or URL parameters); never rebuilds the stand.
import * as THREE from 'three/webgpu'
import {
  cameraPosition,
  dot,
  exp,
  float,
  Fn,
  fog,
  length,
  max,
  mix,
  mx_fractal_noise_float,
  normalize,
  pow,
  positionWorld,
  smoothstep,
  uniform,
  vec3
} from 'three/tsl'
import { Sky, sunElevationAt, toSimTime } from 'threejs-sky'
import { setTreeEnvironment } from 'threejs-trees'
import { createMaterialStateUniforms, Precipitation, Weather, type MaterialStateUniforms } from 'threejs-weathering'

export type GroundCover = 'bare' | 'snow'
export type WeatherKind = 'clear' | 'overcast' | 'rain' | 'snow'

/** What the parent sends (`pgmaps-environment`), all optional; missing fields keep their last value. */
export interface EnvironmentInput {
  /** ISO 8601 with an offset, e.g. `2026-07-15T14:00:00-07:00`. */
  dateTime?: string | null
  latitude?: number | null
  longitude?: number | null
  ground?: GroundCover
  weather?: WeatherKind
}

/** The environment in force: `null` date or position means "the package's default". */
export interface EnvironmentState {
  dateTime: string | null
  latitude: number | null
  longitude: number | null
  ground: GroundCover
  weather: WeatherKind
}

export const DEFAULT_ENVIRONMENT: EnvironmentState = {
  dateTime: null,
  latitude: null,
  longitude: null,
  ground: 'bare',
  weather: 'clear'
}

const GROUNDS: readonly GroundCover[] = ['bare', 'snow']
const WEATHERS: readonly WeatherKind[] = ['clear', 'overcast', 'rain', 'snow']
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/** Merges a message (or URL parameters) into the environment, ignoring fields that are missing or invalid. */
export function mergeEnvironment(state: EnvironmentState, input: EnvironmentInput): EnvironmentState {
  const next = { ...state }
  if (typeof input.dateTime === 'string' && Number.isFinite(Date.parse(input.dateTime))) next.dateTime = input.dateTime
  else if (input.dateTime === null) next.dateTime = null
  if (finite(input.latitude)) next.latitude = Math.max(-89.9, Math.min(89.9, input.latitude))
  if (finite(input.longitude)) next.longitude = ((((input.longitude + 180) % 360) + 360) % 360) - 180
  if (input.ground && GROUNDS.includes(input.ground)) next.ground = input.ground
  if (input.weather && WEATHERS.includes(input.weather)) next.weather = input.weather
  return next
}

/** `?dateTime=…&latitude=…&longitude=…&ground=snow&weather=rain`, for standalone tests. */
export function environmentFromParams(params: URLSearchParams): EnvironmentInput {
  const number = (key: string) => (params.has(key) ? Number(params.get(key)) : undefined)
  return {
    dateTime: params.get('dateTime') ?? undefined,
    latitude: number('latitude') ?? number('lat'),
    longitude: number('longitude') ?? number('lng'),
    ground: (params.get('ground') ?? undefined) as GroundCover | undefined,
    weather: (params.get('weather') ?? undefined) as WeatherKind | undefined
  }
}

/**
 * How each kind of weather looks. Visibility is the meteorological range (Koschmieder: 2% contrast left), which sets
 * the aerial perspective; `sun` scales the direct light (cloud cover takes it), `fill` is the diffuse skylight under
 * cloud, `exposure` brings a dull day back up the way an eye or camera would.
 */
const WEATHER: Record<
  WeatherKind,
  {
    clouds: Record<string, unknown>
    sun: number
    shadows: boolean
    fill: number
    environment: number
    visibility: number
    /** Grey the haze takes under cloud (0 = the clear sky's horizon colour). */
    grey: number
    /** Brightness of that grey relative to the clear-sky horizon. */
    greyLevel: number
    exposure: number
    wetness: number
    foliageSnow: number
    rain: number
    snowfall: number
    wind: number
  }
> = {
  clear: {
    clouds: {
      coverage: 0.28,
      density: 1,
      color: '#ffffff',
      stratus: { amount: 0 },
      cumulus: { amount: 1, size: [400, 1400] },
      cirrus: { amount: 0 },
      silverLining: 1
    },
    sun: 1,
    shadows: true,
    fill: 0.7,
    environment: 0.45,
    visibility: 60_000,
    grey: 0,
    greyLevel: 1,
    exposure: 1.15,
    wetness: 0,
    foliageSnow: 0,
    rain: 0,
    snowfall: 0,
    wind: 0.15
  },
  overcast: {
    clouds: {
      coverage: 1,
      density: 1.8,
      color: '#d5d8dc',
      featureSize: 90_000,
      stratus: { amount: 2, altitude: [500, 1300] },
      cumulus: { amount: 1.2 },
      cirrus: { amount: 0 },
      silverLining: 0.2
    },
    sun: 0.1,
    shadows: false,
    fill: 0.9,
    environment: 0.12,
    visibility: 25_000,
    grey: 0.85,
    greyLevel: 0.75,
    exposure: 1.9,
    wetness: 0.08,
    foliageSnow: 0,
    rain: 0,
    snowfall: 0,
    wind: 0.25
  },
  rain: {
    clouds: {
      coverage: 1,
      density: 2.6,
      color: '#9fa4aa',
      featureSize: 90_000,
      stratus: { amount: 2, altitude: [350, 1000] },
      cumulus: { amount: 1.5 },
      cirrus: { amount: 0 },
      silverLining: 0.1
    },
    sun: 0.04,
    shadows: false,
    fill: 0.7,
    environment: 0.1,
    visibility: 8000,
    grey: 0.95,
    greyLevel: 0.55,
    exposure: 2.1,
    wetness: 0.6,
    foliageSnow: 0,
    rain: 6,
    snowfall: 0,
    wind: 0.45
  },
  snow: {
    clouds: {
      coverage: 1,
      density: 2.2,
      color: '#c4c8cd',
      featureSize: 90_000,
      stratus: { amount: 2, altitude: [300, 900] },
      cumulus: { amount: 1.2 },
      cirrus: { amount: 0 },
      silverLining: 0.1
    },
    sun: 0.05,
    shadows: false,
    fill: 0.85,
    environment: 0.1,
    visibility: 5000,
    grey: 0.95,
    greyLevel: 0.8,
    exposure: 1.6,
    wetness: 0.3,
    foliageSnow: 0.3,
    rain: 0,
    snowfall: 2.5,
    wind: 0.3
  }
}

/** Lowest sun the scene is lit at: below it the clock moves to the nearest dusk (or dawn), so night is never black. */
const DUSK_ELEVATION = -3

export interface Atmosphere {
  sky: Sky
  /** Wetness and snow for the ground (and anything else not a tree); trees take theirs through `setTreeEnvironment`. */
  states: MaterialStateUniforms
  precipitation: Precipitation
  /** The environment as last applied, with the date and position actually used. */
  readonly applied: { state: EnvironmentState; latitude: number; longitude: number; date: string; sunElevation: number }
  /** Applies an environment (cheap: no rebuild). `origin` is the package's own position, for the defaults. */
  apply(state: EnvironmentState, origin: { lat: number; lng: number } | null): void
  /** Ground height in three.js space for rain splashes. */
  setGround(ground: ((x: number, z: number) => number) | null): void
  /** Once a frame, before rendering. `follow` is the point the sun's shadows centre on. */
  update(seconds: number, follow: THREE.Vector3): void
  dispose(): void
}

/** The sky, sun, skylight, haze and weather for a renderer and scene. */
export async function createAtmosphere(
  renderer: THREE.WebGPURenderer,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera
): Promise<Atmosphere> {
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFShadowMap

  /**
   * Everything the sky is given, rebuilt on each change and passed to `reset` so no earlier weather lingers. Cloud
   * altitudes in the table are above the ground; `ground` (metres above sea level) lifts them onto it.
   */
  const skyInput = (
    latitude: number,
    longitude: number,
    date: number,
    w: (typeof WEATHER)[WeatherKind],
    ground: number
  ) => {
    const clouds = structuredClone(w.clouds) as Record<string, { altitude?: number[] } | unknown>
    const lift = (genus: string, fallback: [number, number]) => {
      const g = (clouds[genus] ?? {}) as { altitude?: number[] }
      const [lo, hi] = g.altitude ?? fallback
      clouds[genus] = { ...g, altitude: [lo + ground, hi + ground] }
    }
    lift('cumulus', [1100, 2100])
    lift('stratus', [600, 1400])
    return {
      clock: { date, paused: true },
      observer: { latitude, longitude },
      // A little more aerosol than the default (the interior's summer air: a pale sky near the horizon), and more
      // march steps than the default: the sky is baked once per change (below), so it can afford them, and the clouds'
      // edges no longer step.
      atmosphere: { mie: 6e-6, steps: 24, lightSteps: 6 },
      lighting: {
        autoExposure: false,
        fog: false,
        lightDistance: 220,
        sunIntensity: 3.4 * w.sun,
        shadows: w.shadows,
        environmentIntensity: w.environment
      },
      shootingStars: { rate: 0, shower: 0 },
      clouds: { steps: 64, lightSteps: 6, ...clouds }
    }
  }
  const sky = await Sky.create({
    renderer,
    scene,
    camera,
    ...skyInput(54, -122.6, Date.UTC(new Date().getUTCFullYear(), 6, 15, 21), WEATHER.clear, 0)
  })
  // Never night here (see DUSK_ELEVATION): no stars or planets over the daytime sky.
  sky.object.remove(sky.stars.object, sky.shootingStars.object)
  const sun = sky.sun
  sun.shadow.mapSize.set(4096, 4096)
  Object.assign(sun.shadow.camera, { left: -55, right: 55, top: 55, bottom: -55, near: 1, far: 500 })
  sun.shadow.camera.updateProjectionMatrix()
  sun.shadow.bias = -0.0004
  sun.shadow.normalBias = 0.04
  // Skylight the sky's own image light leaves out: light bounced off the ground (warm, from below) on a clear day,
  // and the grey of a cloud deck, which the image light (the clear sky) does not see.
  const fill = new THREE.HemisphereLight('#dfe3e8', '#5c4a36', 0)
  scene.add(fill)

  // The sky is baked. Its clock is paused, so the dome and clouds only change when the environment does (or the eye
  // travels far): they are rendered into a cube once then and drawn as the background, instead of ray-marching the
  // atmosphere and the clouds behind every pixel of every frame.
  const skyScene = new THREE.Scene()
  const skyTarget = new THREE.CubeRenderTarget(512, { type: THREE.HalfFloatType, generateMipmaps: false })
  const skyCamera = new THREE.CubeCamera(1, 400_000, skyTarget)
  const bakedAt = new THREE.Vector3(Infinity, 0, 0)
  let bakeSky = true
  const takeSky = () => skyScene.add(sky.dome, sky.clouds.object)
  takeSky()
  scene.background = skyTarget.texture

  // Aerial perspective: Koschmieder extinction over the distance travelled, towards the horizon sky's colour, with a
  // warm forward-scattered glow around the sun (Mie). Applied to every material through `scene.fogNode`.
  const hazeColor = uniform(new THREE.Color('#c4cdd6'))
  const glowColor = uniform(new THREE.Color('#000000'))
  const sunDirection = uniform(new THREE.Vector3(0, 1, 0))
  const extinction = uniform(3.912 / 60_000)
  scene.fog = null
  scene.fogNode = Fn(() => {
    const ray = positionWorld.sub(cameraPosition)
    const d = length(ray)
    const mu = dot(ray.div(max(d, 1e-3)), sunDirection)
    // biome-ignore lint/suspicious/noExplicitAny: colour uniforms are typed narrowly in three's TSL typings
    const inscatter = (hazeColor as any).add((glowColor as any).mul(pow(max(mu, float(0)), float(6))))
    const amount = float(1).sub(exp(d.mul(extinction).negate()))
    // biome-ignore lint/suspicious/noExplicitAny: three's typings take a float for the fog colour
    return (fog as any)(inscatter, amount)
  })() as unknown as THREE.Scene['fogNode']

  // A cloud deck for dull weather. threejs-sky's clouds come from a weather field with clear gaps however high the
  // coverage, so overcast, rain and snow draw a closed grey deck instead: a little lighter towards the sun, with
  // stratus texture, fading into the haze at the horizon.
  const deckColor = uniform(new THREE.Color(0.6, 0.6, 0.62))
  const deckMaterial = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, depthWrite: false })
  deckMaterial.fog = false
  {
    const dir = normalize(positionWorld.sub(cameraPosition))
    const texture = mx_fractal_noise_float(dir.mul(vec3(2.5, 7, 2.5)), 4, 2, 0.5)
    const glow = pow(max(dot(dir, sunDirection), float(0)), float(6)).mul(0.4)
    // biome-ignore lint/suspicious/noExplicitAny: colour uniforms are typed narrowly in three's TSL typings
    const body = (deckColor as any).mul(texture.mul(0.22).add(1)).mul(glow.add(1))
    deckMaterial.colorNode = mix(hazeColor as never, body, smoothstep(-0.02, 0.3, dir.y))
  }
  const deck = new THREE.Mesh(new THREE.SphereGeometry(5000, 64, 32), deckMaterial)
  deck.renderOrder = -500
  deck.frustumCulled = false
  deck.visible = false
  skyScene.add(deck)

  const states = createMaterialStateUniforms()
  const precipitation = new Precipitation({
    weather: new Weather({}, null),
    camera,
    ground: null,
    radius: 45,
    height: 32,
    override: { precipitation: 0, snowFraction: 0, wind: [0, 0] }
  })
  scene.add(precipitation.object)

  let look = WEATHER.clear
  let state = DEFAULT_ENVIRONMENT
  const applied = { state, latitude: 54, longitude: -122.6, date: '', sunElevation: 0 }
  const _c = new THREE.Color()
  const _grey = new THREE.Color()

  function apply(next: EnvironmentState, origin: { lat: number; lng: number } | null) {
    state = next
    look = WEATHER[next.weather]
    const latitude = next.latitude ?? origin?.lat ?? 54
    const longitude = next.longitude ?? origin?.lng ?? -122.6
    // The date: as sent, or a mid-July early afternoon in local solar time (about 13:00) for the package's place.
    let ms: number
    if (next.dateTime) ms = Date.parse(next.dateTime)
    else ms = Date.UTC(new Date().getUTCFullYear(), 6, 15, 13) - (longitude / 15) * 3600_000
    const time = duskFloor(sky, toSimTime(ms), latitude, longitude)
    const ground = Math.max(0, Math.round(camera.position.y / 50) * 50)
    sky.reset(skyInput(latitude, longitude, Math.round(Date.UTC(2000, 0, 1, 12) + time * 1000), look, ground))
    sky.update(0)
    applied.state = next
    applied.latitude = latitude
    applied.longitude = longitude
    applied.date = new Date(Date.UTC(2000, 0, 1, 12) + time * 1000).toISOString()
    applied.sunElevation = sky.state.sunElevation
    takeSky() // a rebuilt dome goes back under the sky's own root
    const dull = next.weather !== 'clear'
    deck.visible = dull
    sky.clouds.object.visible = !dull
    bakeSky = true
    extinction.value = 3.912 / look.visibility

    const snowGround = next.ground === 'snow'
    states.snow.value = snowGround ? 1 : 0
    states.wetness.value = look.wetness
    setTreeEnvironment({
      wetness: look.wetness,
      snow: Math.max(look.foliageSnow, snowGround ? 0.6 : 0),
      windStrength: look.wind
    })
    precipitation.set({
      override: {
        precipitation: look.rain || look.snowfall,
        snowFraction: look.snowfall > 0 ? 1 : 0,
        wind: look.rain ? [2.2, 0.8] : [0.9, 0.3]
      }
    })
  }

  /** Haze, sun glow, cloud deck, skylight fill and exposure from the sky's current lighting. */
  function updateLight() {
    const lighting = sky.lighting
    if (!lighting) return
    const day = THREE.MathUtils.smoothstep(sky.state.sunElevation, -6, 12)
    // The haze: the sky's horizon radiance, a little less saturated than the sky itself (low-altitude aerosol
    // scatters white), turned grey under cloud.
    const [r, g, b] = lighting.fogColor
    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    _c.setRGB(r, g, b).lerp(_grey.setRGB(lum, lum, lum), 0.35)
    if (look.grey > 0) _c.lerp(_grey.setRGB(lum, lum * 1.01, lum * 1.03).multiplyScalar(look.greyLevel), look.grey)
    hazeColor.value.copy(_c)
    deckColor.value.copy(_c).multiplyScalar(1.15)
    if (sky.sun.visible) glowColor.value.copy(sky.sun.color).multiplyScalar(lum * 0.6 * look.sun)
    else glowColor.value.setRGB(0, 0, 0)
    sunDirection.value.set(...sky.state.sunDirection)
    // Bounce light and the cloud deck's light fade with the daylight.
    fill.intensity = look.fill * (0.1 + 0.9 * day)
    fill.visible = fill.intensity > 1e-3
    renderer.toneMappingExposure = THREE.MathUtils.clamp(sky.exposure, 0.3, 6) * look.exposure
  }

  return {
    sky,
    states,
    precipitation,
    applied,
    apply,
    setGround(ground) {
      precipitation.ground = ground
    },
    update(seconds, follow) {
      sky.follow(follow)
      sky.update()
      updateLight()
      if (bakeSky || camera.position.distanceToSquared(bakedAt) > 250 * 250) {
        bakeSky = false
        bakedAt.copy(camera.position)
        if (skyCamera.coordinateSystem !== renderer.coordinateSystem) {
          skyCamera.coordinateSystem = renderer.coordinateSystem
          skyCamera.updateCoordinateSystem()
        }
        skyCamera.position.copy(camera.position)
        deck.position.copy(camera.position)
        skyCamera.updateMatrixWorld()
        const target = renderer.getRenderTarget()
        skyCamera.update(renderer, skyScene)
        renderer.setRenderTarget(target)
      }
      precipitation.update(seconds)
    },
    dispose() {
      precipitation.object.removeFromParent()
      fill.removeFromParent()
      sky.object.add(sky.dome, sky.clouds.object)
      sky.dispose()
      skyTarget.dispose()
      scene.background = null
    }
  }
}

/**
 * The simulation time to light the scene at: `time` itself while the sun is at or above {@link DUSK_ELEVATION}, else
 * the nearest moment (before or after) it is, so a night view shows dusk or dawn rather than black.
 */
function duskFloor(sky: Sky, time: number, latitude: number, longitude: number): number {
  const at = (t: number) => sunElevationAt(sky.system, t, latitude, longitude)
  if (at(time) >= DUSK_ELEVATION) return time
  const step = 600
  let best: number | null = null
  for (let k = 1; k <= 13 * 6 && best === null; k++) {
    for (const t of [time - k * step, time + k * step]) {
      if (at(t) < DUSK_ELEVATION) continue
      // Bisect back to the crossing, so the sun sits just at the floor.
      let lo = t
      let hi = t < time ? t + step : t - step
      for (let i = 0; i < 20; i++) {
        const mid = (lo + hi) / 2
        if (at(mid) >= DUSK_ELEVATION) lo = mid
        else hi = mid
      }
      best = lo
      break
    }
  }
  if (best !== null) return best
  // Polar night: the day's highest sun.
  let top = time
  for (let k = -72; k <= 72; k++) if (at(time + k * 600) > at(top)) top = time + k * 600
  return top
}

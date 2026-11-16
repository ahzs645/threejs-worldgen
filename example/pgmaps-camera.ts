/** Perspective replay framing for PGMaps' additive v1 camera metadata. No DOM or three.js. */
export type CameraProjection = { kind: 'perspective'; aspectRatio: number }

/** Fit the saved frame inside a window without changing the lens. Older packages fill the window. */
export function fitCameraViewport(projection: CameraProjection | undefined, width: number, height: number) {
  if (!(Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0))
    throw new Error('The camera needs a positive viewport.')
  const aspect = projection?.aspectRatio ?? width / height
  if ((projection && projection.kind !== 'perspective') || !Number.isFinite(aspect) || aspect <= 0)
    throw new Error('Unsupported PGMaps camera projection.')
  const fittedWidth = Math.min(width, height * aspect)
  const fittedHeight = fittedWidth / aspect
  return {
    width: fittedWidth,
    height: fittedHeight,
    left: (width - fittedWidth) / 2,
    top: (height - fittedHeight) / 2,
    aspect
  }
}

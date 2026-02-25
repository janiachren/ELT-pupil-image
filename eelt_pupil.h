#ifndef EELT_PUPIL_H
#define EELT_PUPIL_H

#ifdef __cplusplus
extern "C" {
#endif

/* Main entry: generates an (npt x npt) pupil reflectivity image.
 * Returns a malloc'ed double* (row-major), or NULL on error. Caller must free().
 *
 * refl: reflectivity per segment (length = nseg). If NULL, uses 1.0 per segment.
 * nseg: number of segments (nominal = 798)
 * npt: grid size (N)
 * dspider: spider width (meters)
 * i0, j0: pupil center indices (can be fractional)
 * pixscale: pixel size (meters per pixel)
 * gap: hard gap threshold (meters)
 * rotdegree: rotation angle (degrees)
 * D: pupil diameter (meters)
 * softGap: 0 = hard gaps; 1 = soft gaps (Lorentzian model)
 */
double* generateEeltPupilReflectivity(
    const double* refl, int nseg, int npt,
    double dspider, double i0, double j0,
    double pixscale, double gap,
    double rotdegree, double D,
    int softGap
);

#ifdef __cplusplus
}
#endif

#endif /* EELT_PUPIL_H */

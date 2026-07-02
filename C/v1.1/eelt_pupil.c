#define _GNU_SOURCE
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "eelt_pupil.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== Python parity constants ===== */
static const double V3 = 1.7320508075688772;         /* sqrt(3) */
static const double PITCH = 1.244683637214;          /* diameter of inscribed circle */

/* ===== Utility: allocate zeroed image ===== */
static double* alloc_image(int N) {
    size_t n = (size_t)N * (size_t)N;
    double* a = (double*)malloc(n * sizeof(double));
    if (!a) return NULL;
    for (size_t i = 0; i < n; ++i) a[i] = 0.0;
    return a;
}

/* ===== Hex pattern creation (createHexaPattern) =====
 * Returns flattened arrays x[], y[] of centers.
 */
static void createHexaPattern(double pitch, double supportSize, double** outx, double** outy, int* outN) {
    int nx = (int)ceil((supportSize / 2.0) / pitch) + 1;
    int ny = (int)ceil((supportSize / 2.0) / pitch / V3) + 1;

    int lenx = 2 * nx + 1;
    int leny = 2 * ny + 1;

    /* Build grids akin to np.meshgrid + flatten */
    int total = lenx * leny;
    double* x = (double*)malloc((size_t)total * sizeof(double));
    double* y = (double*)malloc((size_t)total * sizeof(double));

    if (!x || !y) { free(x); free(y); *outx = NULL; *outy = NULL; *outN = 0; return; }

    int idx = 0;
    for (int iy = 0; iy < leny; ++iy) {
        double yval = (V3 * pitch) * (iy - ny);
        for (int ix = 0; ix < lenx; ++ix) {
            double xval = pitch * (ix - nx);
            x[idx] = xval;
            y[idx] = yval;
            idx++;
        }
    }

    /* Build peak-axis and flat-axis arrays, then append shifted versions */
    double* peak_axis  = (double*)malloc((size_t)total * 2 * sizeof(double));
    double* flat_axis  = (double*)malloc((size_t)total * 2 * sizeof(double));
    if (!peak_axis || !flat_axis) { free(x); free(y); free(peak_axis); free(flat_axis); *outx = NULL; *outy = NULL; *outN = 0; return; }

    for (int i = 0; i < total; ++i) {
        peak_axis[i]       = x[i];
        peak_axis[i+total] = x[i] + pitch / 2.0;
        flat_axis[i]       = y[i];
        flat_axis[i+total] = y[i] + pitch * V3 / 2.0;
    }

    free(x); free(y);

    *outx = flat_axis;     /* return (flat_axis, peak_axis) to match Python (x,y) tuple order */
    *outy = peak_axis;
    *outN = total * 2;
}

/* ===== ESO segment ordering (reorganizeSegmentsOrderESO) =====
 * Sort centers by sector-based metric, returning reordered arrays.
 */
static void reorganizeSegmentsOrderESO(const double* xin, const double* yin, int n, double** xout, double** yout) {
    const double pi_3 = M_PI / 3.0;
    const double pi_6 = M_PI / 6.0;
    const double pix2 = 2.0 * M_PI;
    const double A = 100.0;

    /* Compute angle per point */
    double* t = (double*)malloc((size_t)n * sizeof(double));
    int* idxs = (int*)malloc((size_t)n * sizeof(int));
    if (!t || !idxs) { free(t); free(idxs); *xout = NULL; *yout = NULL; return; }

    for (int i = 0; i < n; ++i) {
        t[i] = fmod(atan2(yin[i], xin[i]) + pi_6 - 1e-3 + pix2, pix2);
        idxs[i] = i;
    }

    /* Collect sorted points by sectors */
    double* X = (double*)malloc((size_t)n * sizeof(double));
    double* Y = (double*)malloc((size_t)n * sizeof(double));
    int w = 0;

    for (int k = 0; k < 6; ++k) {
        double u = k * pi_3;
        /* gather indices in sector */
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (t[i] > k * pi_3 && t[i] < (k + 1) * pi_3) count++;
        }
        if (count == 0) continue;

        /* collect distances and temporary arrays */
        double* dist = (double*)malloc((size_t)count * sizeof(double));
        int* sel = (int*)malloc((size_t)count * sizeof(int));
        int j = 0;
        for (int i = 0; i < n; ++i) {
            if (t[i] > k * pi_3 && t[i] < (k + 1) * pi_3) {
                dist[j] = (A * cos(u) - sin(u)) * xin[i] + (cos(u) + A * sin(u)) * yin[i];
                sel[j] = i;
                j++;
            }
        }

        /* sort sel by dist ascending */
        for (int a = 1; a < count; ++a) {
            double keyd = dist[a];
            int keyi = sel[a];
            int b = a - 1;
            while (b >= 0 && dist[b] > keyd) {
                dist[b + 1] = dist[b];
                sel[b + 1] = sel[b];
                b--;
            }
            dist[b + 1] = keyd;
            sel[b + 1] = keyi;
        }

        /* append ordered points */
        for (int a = 0; a < count; ++a) {
            int ii = sel[a];
            X[w] = xin[ii];
            Y[w] = yin[ii];
            w++;
        }

        free(dist); free(sel);
    }

    /* shrink to actual */
    double* Xs = (double*)malloc((size_t)w * sizeof(double));
    double* Ys = (double*)malloc((size_t)w * sizeof(double));
    if (!Xs || !Ys) { free(X); free(Y); *xout = NULL; *yout = NULL; return; }
    memcpy(Xs, X, (size_t)w * sizeof(double));
    memcpy(Ys, Y, (size_t)w * sizeof(double));
    free(X); free(Y); free(t); free(idxs);

    *xout = Xs;
    *yout = Ys;
}

/* ===== Coordinates of segment corners (generateCoordSegments) =====
 * Returns hx[6*nseg], hy[6*nseg], logically shaped as (6, nseg).
 */
static void generateCoordSegments(double D, double rot, double** hx, double** hy, int* out_nseg) {
    /* Build hex center pattern, then filter by ll radius window */
    double* lx_all = NULL; double* ly_all = NULL; int n_all = 0;
    createHexaPattern(PITCH, 35.0 * PITCH, &lx_all, &ly_all, &n_all);
    if (!lx_all || !ly_all) { *hx = NULL; *hy = NULL; *out_nseg = 0; return; }

    /* empirical inner/outer radius in "pitch" units, as Python */
    const double inner_rad = 4.1;
    const double outer_rad = 15.4;

    /* filter valid centers */
    double* lx = (double*)malloc((size_t)n_all * sizeof(double));
    double* ly = (double*)malloc((size_t)n_all * sizeof(double));
    int n_keep = 0;
    for (int i = 0; i < n_all; ++i) {
        double ll = sqrt(lx_all[i]*lx_all[i] + ly_all[i]*ly_all[i]);
        if (ll > inner_rad * PITCH && ll < outer_rad * PITCH) {
            lx[n_keep] = lx_all[i];
            ly[n_keep] = ly_all[i];
            n_keep++;
        }
    }
    free(lx_all); free(ly_all);

    /* reorder segments */
    double* lx_re = NULL; double* ly_re = NULL;
    reorganizeSegmentsOrderESO(lx, ly, n_keep, &lx_re, &ly_re);
    free(lx); free(ly);
    if (!lx_re || !ly_re) { *hx = NULL; *hy = NULL; *out_nseg = 0; return; }

    /* hexagon vertex prototype (6 points on circle with radius pitch/V3) */
    double th[6];
    for (int i = 0; i < 6; ++i) th[i] = (double)i * (2.0 * M_PI / 6.0);
    double hx0[6], hy0[6];
    for (int i = 0; i < 6; ++i) {
        hx0[i] = cos(th[i]) * PITCH / V3;
        hy0[i] = sin(th[i]) * PITCH / V3;
    }

    int nseg = n_keep;
    double* X = (double*)malloc((size_t)6 * (size_t)nseg * sizeof(double));
    double* Y = (double*)malloc((size_t)6 * (size_t)nseg * sizeof(double));
    if (!X || !Y) { free(lx_re); free(ly_re); free(X); free(Y); *hx = NULL; *hy = NULL; *out_nseg = 0; return; }

    /* combine lx,ly with hx0,hy0, apply correction rrc and rotation */
    const double R = 95.7853;
    for (int s = 0; s < nseg; ++s) {
        for (int v = 0; v < 6; ++v) {
            double x = lx_re[s] + hx0[v];
            double y = ly_re[s] + hy0[v];
            double r = sqrt(x*x + y*y);
            double rrc = R / r * atan(r / R);
            x *= rrc; y *= rrc;

            /* rescale if D != 40 */
            const double nominalD = 40.0;
            if (D != nominalD) {
                x *= D / nominalD;
                y *= D / nominalD;
            }

            /* rotation matrix [[cos, sin],[-sin, cos]] */
            double xr =  cos(rot) * x + sin(rot) * y;
            double yr = -sin(rot) * x + cos(rot) * y;

            X[s*6 + v] = xr;
            Y[s*6 + v] = yr;
        }
    }

    free(lx_re); free(ly_re);
    *hx = X; *hy = Y; *out_nseg = nseg;
}

/* ===== Polygon rasterization (fillPolygon) =====
 * Hard-gap mode: inside polygon if all signed distances to edges > gap.
 * Soft-gap mode: returns dist-to-edge used to attenuate value.
 *
 * Writes into provided image 'a' (N x N, row-major) either mask or attenuation.
 * When want_indices==1, returns count of written pixels and populates idx buffers.
 */
static void rasterize_polygon(
    const double* px, const double* py, /* vertices [6] */
    double i0, double j0, double scale, double gap,
    int N, int softGap,
    int ix0, int iy0, int segdiam,
    double* a, /* output image (N x N) */
    double value /* reflectivity or attenuation multiplier */
) {
    /* Bounding box in pixel indices for speed */
    int xmin = ix0;
    int ymin = iy0;
    int xmax = ix0 + segdiam - 1;
    int ymax = iy0 + segdiam - 1;
    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax >= N) xmax = N - 1;
    if (ymax >= N) ymax = N - 1;

    /* Precompute edges */
    double ex[6], ey[6], vx[6], vy[6], vnorm[6];
    for (int i = 0; i < 6; ++i) {
        int j = (i + 1) % 6;
        ex[i] = px[j] - px[i];
        ey[i] = py[j] - py[i];
        vnorm[i] = sqrt(ex[i]*ex[i] + ey[i]*ey[i]);
        vx[i] = ex[i] / vnorm[i];
        vy[i] = ey[i] / vnorm[i];
    }

    /* For each pixel in bbox, compute world coords and edge tests */
    for (int jj = ymin; jj <= ymax; ++jj) {
        double Y = (jj - j0) * scale;
        for (int ii = xmin; ii <= xmax; ++ii) {
            double X = (ii - i0) * scale;

            /* Center of polygon (mean of vertices) */
            double x0 = 0.0, y0 = 0.0;
            for (int k = 0; k < 6; ++k) { x0 += px[k]; y0 += py[k]; }
            x0 /= 6.0; y0 /= 6.0;

            /* Signed distance to each edge */
            int inside = 1;
            double min_edge = 1e9;
            for (int k = 0; k < 6; ++k) {
                double crossprod = vx[k] * (Y - py[k]) - vy[k] * (X - px[k]);
                if (crossprod <= gap) { inside = 0; break; }
                if (crossprod < min_edge) min_edge = crossprod;
            }

            if (inside) {
                if (softGap) {
                    /* Lorentzian attenuation depth=(gap/scale/pi); dist in meters; scale is m/pixel */
                    double atten = 1.0 - (gap / scale / M_PI) / (1.0 + (min_edge / scale) * (min_edge / scale));
                    a[(size_t)jj * (size_t)N + (size_t)ii] = value * atten;
                } else {
                    a[(size_t)jj * (size_t)N + (size_t)ii] = value;
                }
            }
        }
    }
}

/* ===== Spider mask (fillSpider) =====
 * Writes multiplicative mask (True=keep, False=block => multiply by 0).
 */
static void apply_spider(int N, int nspider, double dspider, double i0, double j0, double scale, double rot, double* img) {
    if (nspider <= 0 || dspider <= 0.0) return;

    /* Prepare mesh (X,Y) in meters centered at (i0,j0) */
    for (int j = 0; j < N; ++j) {
        double Y = (j - j0) * scale;
        for (int i = 0; i < N; ++i) {
            double X = (i - i0) * scale;
            int blocked = 0;
            double w = 2.0 * M_PI / (double)nspider;
            for (int k = 0; k < nspider; ++k) {
                double angle = k * w - rot;
                double proj = X * cos(angle) + Y * sin(angle);
                if (fabs(proj) < dspider / 2.0) { blocked = 1; break; }
            }
            if (blocked) img[(size_t)j * (size_t)N + (size_t)i] *= 0.0;
        }
    }
}

/* ===== Segment properties (generateSegmentProperties) ===== */
static double* generateSegmentProperties(
    const double* attribute, int attr_len,
    const double* hx, const double* hy, int nseg,
    double i0, double j0, double scale, double gap, int N, double D, int softGap
) {
    double* pupil = alloc_image(N);
    if (!pupil) return NULL;

    /* Centers and support window in pixels (as Python) */
    double* x0 = (double*)malloc((size_t)nseg * sizeof(double));
    double* y0 = (double*)malloc((size_t)nseg * sizeof(double));
    if (!x0 || !y0) { free(pupil); free(x0); free(y0); return NULL; }

    for (int s = 0; s < nseg; ++s) {
        double xm = 0.0, ym = 0.0;
        for (int v = 0; v < 6; ++v) {
            xm += hx[s*6 + v];
            ym += hy[s*6 + v];
        }
        xm /= 6.0; ym /= 6.0;
        x0[s] = xm / scale + i0;
        y0[s] = ym / scale + j0;
    }

    double hexrad = 0.75 * D / 40.0 / scale;
    int* ix0 = (int*)malloc((size_t)nseg * sizeof(int));
    int* iy0 = (int*)malloc((size_t)nseg * sizeof(int));
    int* segdiam = (int*)malloc((size_t)nseg * sizeof(int));
    if (!ix0 || !iy0 || !segdiam) { free(pupil); free(x0); free(y0); free(ix0); free(iy0); free(segdiam); return NULL; }

    for (int s = 0; s < nseg; ++s) {
        ix0[s] = (int)floor(x0[s] - hexrad) - 1;
        iy0[s] = (int)floor(y0[s] - hexrad) - 1;
        segdiam[s] = (int)ceil(hexrad * 2.0 + 1.0) + 1;
    }

    /* reflectivity attributes: expect length == nseg, else use scalar replicate */
    int use_scalar = 0;
    double scalar = 1.0;
    if (!attribute || attr_len <= 0) { use_scalar = 1; scalar = 1.0; }

    /* Fill polygons per segment */
    for (int s = 0; s < nseg; ++s) {
        double val = use_scalar ? scalar : attribute[s];
        rasterize_polygon(
            &hx[s*6], &hy[s*6],
            i0, j0, scale, gap,
            N, softGap,
            ix0[s], iy0[s], segdiam[s],
            pupil, val
        );
    }

    free(x0); free(y0); free(ix0); free(iy0); free(segdiam);
    return pupil;
}

/* ===== Entry point (generateEeltPupilReflectivity) ===== */
double* generateEeltPupilReflectivity(
    const double* refl, int nseg, int npt,
    double dspider, double i0, double j0,
    double pixscale, double gap,
    double rotdegree, double D,
    int softGap
) {
    /* Build segment corner coordinates */
    double* hx = NULL; double* hy = NULL; int nseg_geom = 0;
    double rot = rotdegree * M_PI / 180.0;

    generateCoordSegments(D, rot, &hx, &hy, &nseg_geom);
    if (!hx || !hy || nseg_geom <= 0) { free(hx); free(hy); return NULL; }

    /* If caller provides nseg smaller than geometry, use the min */
    int nuse = nseg < nseg_geom ? nseg : nseg_geom;

    /* Fill pupil with attributes */
    double* pupil = generateSegmentProperties(
        refl, nuse, hx, hy, nuse, i0, j0, pixscale, gap, npt, D, softGap
    );

    free(hx); free(hy);
    if (!pupil) return NULL;

    /* Apply spiders (multiplicative boolean mask) */
    if (dspider > 0.0) {
        apply_spider(npt, 3, dspider, i0, j0, pixscale, rot, pupil);
    }

    return pupil;
}

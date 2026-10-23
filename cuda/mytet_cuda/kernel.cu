// mytetration CUDA kernel — one source for the product map, trajectory recording and the one-step test.
//
// Compiled by NVRTC with --fmad=false (no FMA contraction). The original computes, per pixel,
//     z = c;  for k in range(max_iter): z = c ** z;  if np.abs(z) > escape_radius: diverged, break
// with NumPy complex128 *scalars*. What those two operations are on the reference PC
// (numpy 2.5.4, Windows x64, CPU dispatch X86_V3) was established by probing (numerical experiments against NumPy):
//   * c ** z   = npy_cpow (numpy/_core/src/npymath/npy_math_complex.c.src): special cases for b == 0,
//                a == 0, small integer real exponents (|n| < 100, repeated cmul / cdiv), otherwise the
//                platform cpow (UCRT). UCRT cpow == pow() for (real a > 0, real b), else
//                cexp(_Cmulcc(b, clog(a))) with non-fused complex multiply and naive cexp
//                exp(x)*cos(y), exp(x)*sin(y) plus a few inf/NaN special cases.
//   * np.abs(z)= the ufunc CDOUBLE_absolute AVX2/FMA3 SIMD loop: sqrt(fma(r, r, 1)) * larger.
// Mirrored here bit-for-bit where the operations are IEEE basic ops (integer powers, cmul, cdiv, abs)
// and with CUDA libm (exp, sincos, log, log1p, atan2, pow) where the CPU uses UCRT libm.

#define TET_INF __longlong_as_double(0x7ff0000000000000LL)
#define TET_NAN __longlong_as_double(0x7ff8000000000000LL)

struct tet_c {
    double re, im;
};

__device__ __forceinline__ tet_c tet_mk(double re, double im) {
    tet_c r;
    r.re = re;
    r.im = im;
    return r;
}

// numpy cmul: (ar*br - ai*bi, ar*bi + ai*br) — explicit round-to-nearest ops, never contracted.
__device__ __forceinline__ tet_c tet_cmul(tet_c a, tet_c b) {
    return tet_mk(__dsub_rn(__dmul_rn(a.re, b.re), __dmul_rn(a.im, b.im)),
                  __dadd_rn(__dmul_rn(a.re, b.im), __dmul_rn(a.im, b.re)));
}

// numpy cdiv (Smith's algorithm as written in npy_math_complex.c.src).
__device__ __forceinline__ tet_c tet_cdiv(tet_c a, tet_c b) {
    double abs_br = fabs(b.re), abs_bi = fabs(b.im);
    if (abs_br >= abs_bi) {
        if (abs_br == 0.0 && abs_bi == 0.0) {
            return tet_mk(__ddiv_rn(a.re, abs_br), __ddiv_rn(a.im, abs_bi));
        }
        double rat = __ddiv_rn(b.im, b.re);
        double scl = __ddiv_rn(1.0, __dadd_rn(b.re, __dmul_rn(b.im, rat)));
        return tet_mk(__dmul_rn(__dadd_rn(a.re, __dmul_rn(a.im, rat)), scl),
                      __dmul_rn(__dsub_rn(a.im, __dmul_rn(a.re, rat)), scl));
    }
    double rat = __ddiv_rn(b.re, b.im);
    double scl = __ddiv_rn(1.0, __dadd_rn(b.im, __dmul_rn(b.re, rat)));
    return tet_mk(__dmul_rn(__dadd_rn(__dmul_rn(a.re, rat), a.im), scl),
                  __dmul_rn(__dsub_rn(__dmul_rn(a.im, rat), a.re), scl));
}

// np.abs for complex128: numpy simd_cabsolute_f64 (AVX2+FMA3 path), bit-exact.
__device__ __forceinline__ double tet_cabs(double re, double im) {
    re = fabs(re);
    im = fabs(im);
    const bool re_inf = (re == TET_INF), im_inf = (im == TET_INF);
    im = re_inf ? TET_INF : im;
    re = im_inf ? TET_INF : re;
    const bool re_nnan = !isnan(re), im_nnan = !isnan(im);
    im = re_nnan ? im : TET_NAN;
    re = im_nnan ? re : TET_NAN;
    const double larger = (re > im) ? re : im;    // _mm256_max_pd(re, im)
    const double smaller = (im < re) ? im : re;   // _mm256_min_pd(im, re)
    const bool div_ok = !((larger == 0.0) || (smaller == TET_INF));
    const double ratio = div_ok ? __ddiv_rn(smaller, larger) : 0.0;
    const double h = __dsqrt_rn(__fma_rn(ratio, ratio, 1.0));
    return __dmul_rn(h, larger);
}

// Two-sum (exact sum as hi + lo).
__device__ __forceinline__ void tet_two_sum(double a, double b, double* s, double* e) {
    double t = __dadd_rn(a, b);
    double bb = __dsub_rn(t, a);
    *e = __dadd_rn(__dsub_rn(a, __dsub_rn(t, bb)), __dsub_rn(b, bb));
    *s = t;
}

// atan2(y, x) as UCRT gives it where it matters for zero/non-zero: for x > 0 and a quotient |y|/x below the
// smallest normal, atan2 == y/x and UCRT rounds an exact half-quantum tie AWAY from zero (atan2(5e-324, 2) =
// 5e-324, found by probing), while IEEE division rounds it to even (0). Otherwise CUDA atan2.
__device__ __forceinline__ double tet_atan2(double y, double x) {
    if (x > 0.0 && isfinite(x) && isfinite(y) && y != 0.0) {
        double ay = fabs(y);
        double q = __ddiv_rn(ay, x);
        if (q < 0x1p-1022) {
            // exact tie at q + 2^-1075 (q rounded down to even)  <=>  (2q + 2^-1074) * x == 2|y|, tested scaled by
            // 2^600 so the fma difference cannot underflow to zero.
            double m = __dmul_rn(__dadd_rn(__dmul_rn(2.0, q), 0x1p-1074), 0x1p600);
            if (__fma_rn(m, x, -__dmul_rn(__dmul_rn(2.0, ay), 0x1p600)) == 0.0) q = __dadd_rn(q, 0x1p-1074);
            return copysign(q, y);
        }
    }
    return atan2(y, x);
}

// clog(a) = (log|a|, atan2(im, re)), accurate log|a| also near |a| == 1 and for huge/tiny |a|.
// (UCRT clog is a black box; only its value matters, compared within tau by the one-step test.)
__device__ tet_c tet_clog(double ar, double ai) {
    const double LN2_HI = 6.93147180369123816490e-01;  // fdlibm split of ln 2
    const double LN2_LO = 1.90821492927058770002e-10;
    double li = tet_atan2(ai, ar);
    double ax = fabs(ar), ay = fabs(ai);
    double lr;
    if (isinf(ax) || isinf(ay)) {
        lr = TET_INF;
    } else if (isnan(ax) || isnan(ay)) {
        lr = TET_NAN;
    } else if (ax == 0.0 && ay == 0.0) {
        lr = -TET_INF;
    } else {
        double am = ax > ay ? ax : ay, an = ax > ay ? ay : ax;
        int k = 0;
        if (am > 0x1p+1000) {
            k = 600;
        } else if (am < 0x1p-1000) {
            k = -600;
        }
        if (k != 0) {
            am = scalbn(am, -k);
            an = scalbn(an, -k);
        }
        double h = hypot(am, an);
        if (k == 0 && 0.71 <= h && h <= 1.73) {
            // s = am^2 + an^2 - 1 in double-double, then log|a| = log1p(s)/2
            double p1 = __dmul_rn(am, am), e1 = __fma_rn(am, am, -p1);
            double p2 = __dmul_rn(an, an), e2 = __fma_rn(an, an, -p2);
            double h1, l1, h2, l2;
            tet_two_sum(p1, -1.0, &h1, &l1);
            tet_two_sum(h1, p2, &h2, &l2);
            double lo = __dadd_rn(__dadd_rn(l1, l2), __dadd_rn(e1, e2));
            double s = __dadd_rn(h2, lo);
            lr = __dmul_rn(0.5, log1p(s));
        } else {
            lr = log(h);
            if (k != 0) {
                lr = __dadd_rn(__dadd_rn(lr, __dmul_rn((double)k, LN2_HI)), __dmul_rn((double)k, LN2_LO));
            }
        }
    }
    return tet_mk(lr, li);
}

// UCRT cexp as observed: naive exp(x)*(cos y, sin y) except the inf/NaN cases below.
__device__ __forceinline__ tet_c tet_cexp(double x, double y) {
    if (isnan(x)) {
        if (y == 0.0) return tet_mk(x, y);
    } else if (isinf(x)) {
        if (x > 0.0) {
            if (y == 0.0) return tet_mk(x, y);
            if (!isfinite(y)) return tet_mk(x, TET_NAN);
        } else {
            if (isnan(y)) return tet_mk(0.0, 0.0);  // UCRT: +0 whatever the NaN's sign (found by probing)
            if (isinf(y)) return tet_mk(0.0, copysign(0.0, y));
        }
    }
    double e = exp(x);
    double s, c;
    sincos(y, &s, &c);
    return tet_mk(__dmul_rn(e, c), __dmul_rn(e, s));
}

// a ** b exactly as npy_cpow + UCRT cpow are structured. loga must be tet_clog(a.re, a.im).
__device__ tet_c tet_cpow(tet_c a, tet_c b, tet_c loga) {
    if (b.re == 0.0 && b.im == 0.0) {
        return tet_mk(1.0, 0.0);
    }
    if (a.re == 0.0 && a.im == 0.0) {
        if (b.re > 0.0) return tet_mk(0.0, 0.0);
        return tet_mk(TET_NAN, TET_NAN);
    }
    if (b.im == 0.0 && b.re > -100.0 && b.re < 100.0) {
        long long n = (long long)b.re;
        if ((double)n == b.re) {
            if (n == 1) return a;
            if (n == 2) return tet_cmul(a, a);
            if (n == 3) return tet_cmul(a, tet_cmul(a, a));
            long long mask = 1;
            if (n < 0) n = -n;
            tet_c aa = tet_mk(1.0, 0.0);
            tet_c p = a;
            while (1) {
                if (n & mask) aa = tet_cmul(aa, p);
                mask <<= 1;
                if (n < mask || mask <= 0) break;
                p = tet_cmul(p, p);
            }
            if (b.re < 0.0) aa = tet_cdiv(tet_mk(1.0, 0.0), aa);
            return aa;
        }
    }
    // platform cpow (UCRT)
    if (a.im == 0.0 && b.im == 0.0 && a.re > 0.0) {
        return tet_mk(pow(a.re, b.re), 0.0);
    }
    double wr = __dsub_rn(__dmul_rn(b.re, loga.re), __dmul_rn(b.im, loga.im));
    double wi = __dadd_rn(__dmul_rn(b.re, loga.im), __dmul_rn(b.im, loga.re));
    return tet_cexp(wr, wi);
}

// c[i, j] = x[i] + (1j*y)[j], with w = 1j*y computed by NumPy on the host (same expression as the
// original). NumPy adds float + complex as (x + w.re, 0.0 + w.im).
__device__ __forceinline__ tet_c tet_grid_c(const double* x, const double* wr, const double* wi, long long i,
                                            long long j) {
    return tet_mk(__dadd_rn(x[i], wr[j]), __dadd_rn(0.0, wi[j]));
}

// One pixel. Returns 1 if diverged; *stop = number of powers computed (k+1 at divergence, else max_iter).
// With TET_RECORD, rec (if non-null) receives z_0 .. z_stop as (re, im) pairs.
__device__ __forceinline__ int tet_pixel(tet_c c, int max_iter, double escape_radius, int* stop, double* rec) {
    const tet_c lc = tet_clog(c.re, c.im);
    tet_c z = c;
#ifdef TET_RECORD
    if (rec) {
        rec[0] = z.re;
        rec[1] = z.im;
    }
#endif
    for (int k = 0; k < max_iter; ++k) {
        z = tet_cpow(c, z, lc);
#ifdef TET_RECORD
        if (rec) {
            rec[2 * (k + 1)] = z.re;
            rec[2 * (k + 1) + 1] = z.im;
        }
#endif
        if (tet_cabs(z.re, z.im) > escape_radius) {
            *stop = k + 1;
            return 1;
        }
    }
    *stop = max_iter;
    return 0;
}

// Product kernel: rows [row0, row0 + nrows) of the (nx, ny) map, row-major like divergence_map[i, j].
// iters may be null (product run); if given it receives the stop number per pixel.
extern "C" __global__ void tet_map_rows(const double* __restrict__ x, const double* __restrict__ wr,
                                        const double* __restrict__ wi, int ny, int row0, int nrows,
                                        int max_iter, double escape_radius, unsigned char* __restrict__ out,
                                        int* __restrict__ iters) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nrows * ny;
    if (t >= total) return;
    long long i = row0 + t / ny, j = t % ny;
    int stop;
    int d = tet_pixel(tet_grid_c(x, wr, wi, i, j), max_iter, escape_radius, &stop, (double*)0);
    long long idx = i * (long long)ny + j;
    out[idx] = (unsigned char)d;
    if (iters) iters[idx] = stop;
}

#ifdef TET_RECORD
// Trajectory kernel (same tet_pixel, recording on): pixels given as flat indices i*ny + j.
extern "C" __global__ void tet_record(const double* __restrict__ x, const double* __restrict__ wr,
                                      const double* __restrict__ wi, int ny, const long long* __restrict__ pix,
                                      int npix, int max_iter, double escape_radius,
                                      unsigned char* __restrict__ out, int* __restrict__ iters,
                                      double* __restrict__ traj) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npix) return;
    long long p = pix[t];
    int stop;
    double* rec = traj + (long long)t * 2 * ((long long)max_iter + 1);
    out[t] = (unsigned char)tet_pixel(tet_grid_c(x, wr, wi, p / ny, p % ny), max_iter, escape_radius, &stop, rec);
    iters[t] = stop;
}
#endif

// One-step test: out[k] = a[k] ** b[k] through the same device function the kernels use.
extern "C" __global__ void tet_onestep(const double* __restrict__ a, const double* __restrict__ b, int n,
                                       double* __restrict__ out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    tet_c av = tet_mk(a[2 * t], a[2 * t + 1]), bv = tet_mk(b[2 * t], b[2 * t + 1]);
    tet_c r = tet_cpow(av, bv, tet_clog(av.re, av.im));
    out[2 * t] = r.re;
    out[2 * t + 1] = r.im;
}

// abs test: out[k] = |z[k]| through the same device function the kernels use.
extern "C" __global__ void tet_abs(const double* __restrict__ z, int n, double* __restrict__ out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    out[t] = tet_cabs(z[2 * t], z[2 * t + 1]);
}

// Grid test: c for every pixel of rows [row0, row0+nrows).
extern "C" __global__ void tet_grid(const double* __restrict__ x, const double* __restrict__ wr,
                                    const double* __restrict__ wi, int ny, int row0, int nrows,
                                    double* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nrows * ny;
    if (t >= total) return;
    long long i = row0 + t / ny, j = t % ny;
    tet_c c = tet_grid_c(x, wr, wi, i, j);
    out[2 * t] = c.re;
    out[2 * t + 1] = c.im;
}

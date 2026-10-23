// Viewer fast mode: the product algorithm (mytet_cuda/kernel.cu) in float32, max_iter 500,
// escape 1e10. A float32 variant of the product kernel, kept
// in the viewer folder. IEEE single precision, NVRTC defaults (no fast-math). Deterministic: same inputs -> same bits.


#define F_INF __int_as_float(0x7f800000)
#define F_NAN __int_as_float(0x7fc00000)

struct fc {
    float re, im;
};

__device__ __forceinline__ fc fmk(float re, float im) {
    fc r;
    r.re = re;
    r.im = im;
    return r;
}

__device__ __forceinline__ fc f_cmul(fc a, fc b) {
    return fmk(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

__device__ __forceinline__ fc f_cdiv(fc a, fc b) {
    float abs_br = fabsf(b.re), abs_bi = fabsf(b.im);
    if (abs_br >= abs_bi) {
        if (abs_br == 0.0f && abs_bi == 0.0f) return fmk(a.re / abs_br, a.im / abs_bi);
        float rat = b.im / b.re;
        float scl = 1.0f / (b.re + b.im * rat);
        return fmk((a.re + a.im * rat) * scl, (a.im - a.re * rat) * scl);
    }
    float rat = b.re / b.im;
    float scl = 1.0f / (b.im + b.re * rat);
    return fmk((a.re * rat + a.im) * scl, (a.im * rat - a.re) * scl);
}

__device__ __forceinline__ float f_cabs(float re, float im) {
    re = fabsf(re);
    im = fabsf(im);
    if (isinf(re) || isinf(im)) return F_INF;
    if (isnan(re) || isnan(im)) return F_NAN;
    float larger = re > im ? re : im, smaller = im < re ? im : re;
    if (larger == 0.0f) return 0.0f;
    float ratio = smaller / larger;
    return sqrtf(fmaf(ratio, ratio, 1.0f)) * larger;
}

__device__ __forceinline__ fc f_clog(float ar, float ai) {
    float h = hypotf(ar, ai);
    float lr;
    if (0.71f <= h && h <= 1.73f) {
        float am = fmaxf(fabsf(ar), fabsf(ai)), an = fminf(fabsf(ar), fabsf(ai));
        lr = 0.5f * log1pf((am - 1.0f) * (am + 1.0f) + an * an);
    } else {
        lr = logf(h);
    }
    return fmk(lr, atan2f(ai, ar));
}

__device__ __forceinline__ fc f_cexp(float x, float y) {
    if (isnan(x)) {
        if (y == 0.0f) return fmk(x, y);
    } else if (isinf(x)) {
        if (x > 0.0f) {
            if (y == 0.0f) return fmk(x, y);
            if (!isfinite(y)) return fmk(x, F_NAN);
        } else {
            if (isnan(y)) return fmk(0.0f, 0.0f);
            if (isinf(y)) return fmk(0.0f, copysignf(0.0f, y));
        }
    }
    float e = expf(x), s, c;
    sincosf(y, &s, &c);
    return fmk(e * c, e * s);
}

__device__ fc f_cpow(fc a, fc b, fc loga) {
    if (b.re == 0.0f && b.im == 0.0f) return fmk(1.0f, 0.0f);
    if (a.re == 0.0f && a.im == 0.0f) return (b.re > 0.0f) ? fmk(0.0f, 0.0f) : fmk(F_NAN, F_NAN);
    if (b.im == 0.0f && b.re > -100.0f && b.re < 100.0f) {
        int n = (int)b.re;
        if ((float)n == b.re) {
            if (n == 1) return a;
            if (n == 2) return f_cmul(a, a);
            if (n == 3) return f_cmul(a, f_cmul(a, a));
            int mask = 1;
            if (n < 0) n = -n;
            fc aa = fmk(1.0f, 0.0f), p = a;
            while (1) {
                if (n & mask) aa = f_cmul(aa, p);
                mask <<= 1;
                if (n < mask || mask <= 0) break;
                p = f_cmul(p, p);
            }
            if (b.re < 0.0f) aa = f_cdiv(fmk(1.0f, 0.0f), aa);
            return aa;
        }
    }
    if (a.im == 0.0f && b.im == 0.0f && a.re > 0.0f) return fmk(powf(a.re, b.re), 0.0f);
    return f_cexp(b.re * loga.re - b.im * loga.im, b.re * loga.im + b.im * loga.re);
}

extern "C" __global__ void f32_map_rows(const float* __restrict__ x, const float* __restrict__ wr,
                                        const float* __restrict__ wi, int ny, int row0, int nrows, int max_iter,
                                        float escape_radius, unsigned char* __restrict__ out,
                                        float* __restrict__ coordinates) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nrows * ny;
    if (t >= total) return;
    long long i = row0 + t / ny, j = t % ny;
    fc c = fmk(x[i] + wr[j], 0.0f + wi[j]);
    if (coordinates) {
        coordinates[2 * (i * (long long)ny + j)] = c.re;
        coordinates[2 * (i * (long long)ny + j) + 1] = c.im;
    }
    fc lc = f_clog(c.re, c.im);
    fc z = c;
    unsigned char d = 0;
    for (int k = 0; k < max_iter; ++k) {
        z = f_cpow(c, z, lc);
        if (f_cabs(z.re, z.im) > escape_radius) {
            d = 1;
            break;
        }
    }
    out[i * (long long)ny + j] = d;
}

// Double fast path: original special-case structure, full-resolution double coordinates.
#define D_INF __longlong_as_double(0x7ff0000000000000LL)
#define D_NAN __longlong_as_double(0x7ff8000000000000LL)

struct dc {
    double re, im;
};

__device__ __forceinline__ dc dmk(double re, double im) {
    dc r;
    r.re = re;
    r.im = im;
    return r;
}

__device__ __forceinline__ dc d_cmul(dc a, dc b) {
    return dmk(__dsub_rn(__dmul_rn(a.re, b.re), __dmul_rn(a.im, b.im)),
               __dadd_rn(__dmul_rn(a.re, b.im), __dmul_rn(a.im, b.re)));
}

__device__ __forceinline__ dc d_cdiv(dc a, dc b) {
    double abs_br = fabs(b.re), abs_bi = fabs(b.im);
    if (abs_br >= abs_bi) {
        if (abs_br == 0.0 && abs_bi == 0.0) return dmk(a.re / abs_br, a.im / abs_bi);
        double rat = b.im / b.re;
        double scl = 1.0 / (b.re + b.im * rat);
        return dmk((a.re + a.im * rat) * scl, (a.im - a.re * rat) * scl);
    }
    double rat = b.re / b.im;
    double scl = 1.0 / (b.im + b.re * rat);
    return dmk((a.re * rat + a.im) * scl, (a.im * rat - a.re) * scl);
}

__device__ __forceinline__ double d_cabs(double re, double im) {
    re = fabs(re);
    im = fabs(im);
    if (isinf(re) || isinf(im)) return D_INF;
    if (isnan(re) || isnan(im)) return D_NAN;
    double larger = re > im ? re : im, smaller = im < re ? im : re;
    if (larger == 0.0) return 0.0;
    double ratio = smaller / larger;
    return sqrt(fma(ratio, ratio, 1.0)) * larger;
}

// Copied from mytet_cuda/kernel.cu; f64 module uses --fmad=false.
// Two-sum (exact sum as hi + lo).
__device__ __forceinline__ void d_log_two_sum(double a, double b, double* s, double* e) {
    double t = __dadd_rn(a, b);
    double bb = __dsub_rn(t, a);
    *e = __dadd_rn(__dsub_rn(a, __dsub_rn(t, bb)), __dsub_rn(b, bb));
    *s = t;
}

// atan2(y, x) as UCRT gives it where it matters for zero/non-zero: for x > 0 and a quotient |y|/x below the
// smallest normal, atan2 == y/x and UCRT rounds an exact half-quantum tie AWAY from zero (atan2(5e-324, 2) =
// 5e-324), while IEEE division rounds it to even (0). Otherwise CUDA atan2.
__device__ __forceinline__ double d_log_atan2(double y, double x) {
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
__device__ dc d_clog(double ar, double ai) {
    const double LN2_HI = 6.93147180369123816490e-01;  // fdlibm split of ln 2
    const double LN2_LO = 1.90821492927058770002e-10;
    double li = d_log_atan2(ai, ar);
    double ax = fabs(ar), ay = fabs(ai);
    double lr;
    if (isinf(ax) || isinf(ay)) {
        lr = D_INF;
    } else if (isnan(ax) || isnan(ay)) {
        lr = D_NAN;
    } else if (ax == 0.0 && ay == 0.0) {
        lr = -D_INF;
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
            d_log_two_sum(p1, -1.0, &h1, &l1);
            d_log_two_sum(h1, p2, &h2, &l2);
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
    return dmk(lr, li);
}

__device__ __forceinline__ dc d_cexp(double x, double y) {
    if (isnan(x)) {
        if (y == 0.0) return dmk(x, y);
    } else if (isinf(x)) {
        if (x > 0.0) {
            if (y == 0.0) return dmk(x, y);
            if (!isfinite(y)) return dmk(x, D_NAN);
        } else {
            if (isnan(y)) return dmk(0.0, 0.0);
            if (isinf(y)) return dmk(0.0, copysign(0.0, y));
        }
    }
    double e = exp(x), s, c;
    sincos(y, &s, &c);
    return dmk(e * c, e * s);
}

__device__ dc d_cpow(dc a, dc b, dc loga) {
    if (b.re == 0.0 && b.im == 0.0) return dmk(1.0, 0.0);
    if (a.re == 0.0 && a.im == 0.0) return (b.re > 0.0) ? dmk(0.0, 0.0) : dmk(D_NAN, D_NAN);
    if (b.im == 0.0 && b.re > -100.0 && b.re < 100.0) {
        int n = (int)b.re;
        if ((double)n == b.re) {
            if (n == 1) return a;
            if (n == 2) return d_cmul(a, a);
            if (n == 3) return d_cmul(a, d_cmul(a, a));
            int mask = 1;
            if (n < 0) n = -n;
            dc aa = dmk(1.0, 0.0), p = a;
            while (1) {
                if (n & mask) aa = d_cmul(aa, p);
                mask <<= 1;
                if (n < mask || mask <= 0) break;
                p = d_cmul(p, p);
            }
            if (b.re < 0.0) aa = d_cdiv(dmk(1.0, 0.0), aa);
            return aa;
        }
    }
    if (a.im == 0.0 && b.im == 0.0 && a.re > 0.0) return dmk(pow(a.re, b.re), 0.0);
    return d_cexp(__dsub_rn(__dmul_rn(b.re, loga.re), __dmul_rn(b.im, loga.im)),
                  __dadd_rn(__dmul_rn(b.re, loga.im), __dmul_rn(b.im, loga.re)));
}

extern "C" __global__ void f64_map_rows(const double* __restrict__ x, const double* __restrict__ wr,
                                        const double* __restrict__ wi, int ny, int row0, int nrows, int max_iter,
                                        double escape_radius, unsigned char* __restrict__ out,
                                        double* __restrict__ coordinates, int cycle_exit) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nrows * ny;
    if (t >= total) return;
    long long i = row0 + t / ny, j = t % ny;
    dc c = dmk(x[i] + wr[j], 0.0 + wi[j]);
    if (coordinates) {
        coordinates[2 * (i * (long long)ny + j)] = c.re;
        coordinates[2 * (i * (long long)ny + j) + 1] = c.im;
    }
    dc lc = d_clog(c.re, c.im);
    dc z = c;
    dc anchor = z;
    int power = 1, length = 0;
    const double escape2 = escape_radius * escape_radius;
    unsigned char d = 0;
    for (int k = 0; k < max_iter; ++k) {
        z = d_cpow(c, z, lc);
        // Squared norm avoids a double division/sqrt on every iteration.
        // f32's abs gives infinity precedence over NaN, so preserve it explicitly.
        if (z.re * z.re + z.im * z.im > escape2 || isinf(z.re) || isinf(z.im)) {
            d = 1;
            break;
        }
        if (cycle_exit) {
            // Brent's checkpoint schedule. Exact equality proves a deterministic
            // cycle; no tolerance and no reuse between pixels or frames.
            ++length;
            if (__double_as_longlong(z.re) == __double_as_longlong(anchor.re)
                && __double_as_longlong(z.im) == __double_as_longlong(anchor.im)) break;
            if (length == power) {
                anchor = z;
                power <<= 1;
                length = 0;
            }
        }
    }
    out[i * (long long)ny + j] = d;
}


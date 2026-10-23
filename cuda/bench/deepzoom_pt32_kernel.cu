// Bench-only perturbation experiment.
// Viewer fast mode: the product algorithm (mytet_cuda/kernel.cu) in float32, max_iter 500,
// escape 1e10. A bench copy, so the viewer does not depend on bench code. IEEE single precision, NVRTC defaults (no fast-math). Deterministic: same inputs -> same bits.


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

__device__ __forceinline__ dc d_clog(double ar, double ai) {
    double h = hypot(ar, ai);
    double lr;
    if (0.71 <= h && h <= 1.73) {
        double am = fmax(fabs(ar), fabs(ai)), an = fmin(fabs(ar), fabs(ai));
        lr = 0.5 * log1p((am - 1.0) * (am + 1.0) + an * an);
    } else {
        lr = log(h);
    }
    return dmk(lr, atan2(ai, ar));
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

// Reference candidates are ordered centre-outward on the host. Every candidate
// uses exactly the f64 path's double log/power and escape operations.
extern "C" __global__ void pt32_references(const double* x, const double* y,
        const int* candidates, int count, int max_iter, double escape_radius,
        dc* orbits, int* stops) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    dc c = dmk(x[candidates[2*t]], 0.0 + y[candidates[2*t+1]]);
    dc lc = d_clog(c.re, c.im), z = c;
    dc* orbit = orbits + t * (max_iter + 2);
    orbit[0] = c;
    orbit[max_iter+1] = lc;
    int stop = max_iter + 1;
    for (int k = 0; k < max_iter; ++k) {
        z = d_cpow(c, z, lc);
        orbit[k+1] = z;
        if (z.re*z.re + z.im*z.im > escape_radius*escape_radius
                || isinf(z.re) || isinf(z.im)) { stop = k+1; break; }
    }
    stops[t] = stop;
}

__device__ __forceinline__ fc f_expm1c(fc e) {
    float a = expm1f(e.re), s, c;
    sincosf(e.im, &s, &c);
    float h = sinf(0.5f * e.im);
    return fmk(a*c - 2.0f*h*h, (a+1.0f)*s);
}

extern "C" __global__ void pt32_map_rows(const double* x, const double* wr,
        const double* wi, int ny, int row0, int nrows, int max_iter,
        double escape_radius, unsigned char* out, float* coordinates,
        const dc* orbit, int k_ref, float rho, int cycle_exit, int* switches) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)nrows * ny) return;
    long long i = row0 + t / ny, j = t % ny, index = i*(long long)ny+j;
    dc ref = orbit[0];
    fc gamma = fmk((float)((x[i]+wr[j])-ref.re), (float)((0.0+wi[j])-ref.im));
    if (coordinates) { coordinates[2*index]=gamma.re; coordinates[2*index+1]=gamma.im; }
    // This is the represented c used by fallback as well as perturbation.
    dc c = dmk(ref.re + (double)gamma.re, ref.im + (double)gamma.im);
    // Compute the small logarithm against the double reference once per pixel,
    // then round lambda to float32 for the perturbation recurrence.
    dc q = d_cdiv(dmk((double)gamma.re, (double)gamma.im), ref);
    fc lambda = fmk((float)(0.5*log1p(2.0*q.re+q.re*q.re+q.im*q.im)),
                      (float)atan2(q.im, 1.0+q.re));
    fc l0 = fmk((float)orbit[max_iter+1].re, (float)orbit[max_iter+1].im);
    // log(c)-log(ref) differs from log1p(gamma/ref) by 2*pi*i
    // across the principal-log cut, even when gamma is arbitrarily small.
    bool crosses_log_cut = ref.re < 0.0 && c.re < 0.0 && signbit(ref.im) != signbit(c.im);
    fc delta = gamma;
    int k = 0, switched = -1;
    unsigned char d = 0;
    for (; k < max_iter; ++k) {
        if (k >= k_ref) break;
        fc zref = fmk((float)orbit[k].re, (float)orbit[k].im);
        float rn = zref.re*zref.re+zref.im*zref.im;
        float dn = delta.re*delta.re+delta.im*delta.im;
        fc z = fmk(zref.re+delta.re, zref.im+delta.im);
        float zn = z.re*z.re+z.im*z.im;
        if (crosses_log_cut || !isfinite(dn) || !isfinite(rn) || !isfinite(lambda.re) || !isfinite(lambda.im)
                || dn > rho*rho*rn || zn < 1e-6f*rn || rn == 0.0f) break;
        fc a = f_cmul(zref, lambda);
        fc b = f_cmul(delta, fmk(l0.re+lambda.re, l0.im+lambda.im));
        fc e = fmk(a.re+b.re, a.im+b.im);
        fc nextref = fmk((float)orbit[k+1].re, (float)orbit[k+1].im);
        fc next = f_cmul(nextref, f_expm1c(e));
        // Reconstruct in double rather than losing gamma at the escape test.
        dc nextz = dmk(orbit[k+1].re+(double)next.re, orbit[k+1].im+(double)next.im);
        if (!isfinite(next.re) || !isfinite(next.im)) break;
        delta = next;
        if (nextz.re*nextz.re+nextz.im*nextz.im > escape_radius*escape_radius
                || isinf(nextz.re) || isinf(nextz.im)) { d=1; k=max_iter; break; }
    }
    if (k < max_iter) {
        switched = k;
        dc z = dmk(orbit[k].re+(double)delta.re, orbit[k].im+(double)delta.im);
        dc lc = d_clog(c.re, c.im), anchor = z;
        int power=1, length=0;
        for (; k < max_iter; ++k) {
            z = d_cpow(c, z, lc);
            if (z.re*z.re+z.im*z.im > escape_radius*escape_radius
                    || isinf(z.re) || isinf(z.im)) { d=1; break; }
            if (cycle_exit) {
                ++length;
                if (__double_as_longlong(z.re)==__double_as_longlong(anchor.re)
                        && __double_as_longlong(z.im)==__double_as_longlong(anchor.im)) break;
                if (length==power) { anchor=z; power<<=1; length=0; }
            }
        }
    }
    if (switches) switches[index] = switched;
    out[index] = d;
}

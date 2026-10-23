// Extended singular orbit: Z[0]=0, Z[1]=1, Z[2]=C, e=n+2.
// Log's branch cut and |C|<1e-3 are rejected by the host, as in stage one.
// n>=1 only: starting c is not an escape test. NVRTC --fmad=false is mandatory.
// Cancellation is resolved by multiplication or DD log-polar reconstruction.
#define DBL_MIN 2.2250738585072014e-308
#define DBL_MAX 1.7976931348623157e308
struct pt_c { double re, im; };
__device__ __forceinline__ pt_c pt_mk(double r, double i) { return {r,i}; }
__device__ __forceinline__ pt_c pt_add(pt_c a, pt_c b) { return {a.re+b.re,a.im+b.im}; }
__device__ __forceinline__ pt_c pt_mul(pt_c a, pt_c b) {
    return {a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re};
}
__device__ pt_c pt_div(pt_c a, pt_c b) {
    if (fabs(b.re)>=fabs(b.im)) {
        double r=b.im/b.re, s=1.0/(b.re+b.im*r);
        return {(a.re+a.im*r)*s,(a.im-a.re*r)*s};
    }
    double r=b.re/b.im, s=1.0/(b.im+b.re*r);
    return {(a.re*r+a.im)*s,(a.im*r-a.re)*s};
}
__device__ pt_c pt_exp(pt_c a) { double e=exp(a.re); return {e*cos(a.im),e*sin(a.im)}; }
__device__ pt_c pt_naive_log1p(pt_c a) { return {log(hypot(1.0+a.re,a.im)),atan2(a.im,1.0+a.re)}; }
__device__ pt_c pt_clog1p(pt_c a) {
    double u=a.re, v=a.im, h=hypot(1.0+u,v);
    double s=2.0*u+(u*u+v*v);
    double re=fabs(h-1.0)<=0.5 ? 0.5*log1p(s) : log(h);
    return {re,atan2(v,1.0+u)};
}
__device__ pt_c pt_cexpm1(pt_c a) {
    double s=sin(a.im/2.0);
    return {expm1(a.re)*cos(a.im)-2.0*(s*s),exp(a.re)*sin(a.im)};
}
// Shared call sites also exercised by PT_TEST: fault replacement is unique.
__device__ pt_c pt_lambda(pt_c u) {
    return pt_clog1p(u);
}
__device__ pt_c pt_increment(pt_c w) {
    return pt_cexpm1(w);
}
__device__ bool pt_finite(pt_c z) { return isfinite(z.re)&&isfinite(z.im); }
__device__ bool pt_subnormal(pt_c z) {
    double h=hypot(z.re,z.im);
    return h!=0.0 && h<DBL_MIN;
}


struct pt_dd { double hi, lo; };
__device__ pt_dd pt_ddadd(double hi, double lo, double x) {
    double s=hi+x, v=s-hi;
    double t=((hi-(s-v))+(x-v))+lo;
    double h=s+t;
    return {h,t-(h-s)};
}
__device__ bool pt_same(pt_c a, pt_c b) {
    return __double_as_longlong(a.re)==__double_as_longlong(b.re)
        && __double_as_longlong(a.im)==__double_as_longlong(b.im);
}
__device__ pt_c pt_polar(pt_dd logz, pt_dd th) {
    double r=exp(logz.hi)*exp(logz.lo);
    double ch=cos(th.hi), sh=sin(th.hi), cl=cos(th.lo), sl=sin(th.lo);
    return {r*(ch*cl-sh*sl),r*(sh*cl+ch*sl)};
}
__device__ bool pt_cycle(pt_c delta, pt_c initial) {
    return pt_same(delta,initial); // FAULT_2: exact state recurrence only
}
__device__ bool pt_cancel(pt_c u) {
    return u.re < -0.5; // FAULT_3: cancellation rebases
}
__device__ bool pt_detect() {
    return true; // FAULT_4: cancellation and uncertain escape detection
}
extern "C" __global__ void pt_map(
    const double* gx, const double* gy, double cr, double ci, double lr, double li,
    const pt_c* zs, const double* logs, const double* loglo,
    const double* argz, const double* arglo, const double* margins, int length, int maxiter,
    const long long* pixels, int stride, int ny, int offset, int count,
    unsigned char* out, unsigned char* flags, int* stops, int* rebases, unsigned char* cycles) {
    int t=blockDim.x*blockIdx.x+threadIdx.x;
    if(t>=count) return;
    int index=offset+t;
    long long pixel=pixels ? pixels[index] : (long long)(index/ny*stride)*1080+index%ny*stride;
    int px=(int)(pixel/1080), py=(int)(pixel%1080);
    out[index]=0; flags[index]=0; stops[index]=0; rebases[index]=0; cycles[index]=0;
    pt_c gamma={gx[px],gy[py]}, delta={0.,0.}, initial={0.,0.};
    if(pt_subnormal(gamma)) { flags[index]=2; return; }
    pt_c lambda=pt_lambda(pt_div(gamma,{cr,ci}));
    pt_c a=pt_add({lr,li},lambda);
    int e=0;
    bool saved=false;
    for(int n=-1;n<=maxiter;n++) {
        int next=e+1;
        if(next>=length) { flags[index]=3; return; } // host always provides through final escape
        pt_c w=pt_add(pt_mul(delta,a),pt_mul(zs[e],lambda));
        pt_c zn=zs[next], u={0.,0.}, dn={0.,0.};
        bool direct=hypot(zn.re,zn.im)<DBL_MIN || !pt_finite(zn)
                     || !pt_finite(w) || w.re>log(DBL_MAX) || (next==length-1 && (n<maxiter || margins[next]>0.0));
        if(!direct) {
            u=pt_increment(w);
            dn=pt_mul(zn,u);
            direct=!pt_finite(dn);
        }
        if(!direct && n>=1) {
            double margin=margins[next]+2.0*(zn.re*dn.re+zn.im*dn.im)+(dn.re*dn.re+dn.im*dn.im);
            if(isfinite(margin)) {
                if(margin>0.0) { out[index]=1; stops[index]=n; return; }
            } else direct=true;
        }
        bool cancel=!direct && pt_detect() && pt_cancel(u);
        if(direct || cancel) {
            pt_dd logz=pt_ddadd(logs[next],loglo[next],w.re);
            // DD removes the O(eps*|ln Z|) input/sum rounding term. The double
            // products in w still cost <=8 eps*wmag; DD residual costs eps^2
            // times the input scale. 1e-12 covers exp/log/trig evaluation error.
            double wmag=hypot(delta.re,delta.im)*hypot(a.re,a.im)+hypot(zs[e].re,zs[e].im)*hypot(lambda.re,lambda.im);
            double tol=1e-12+8.0*2.220446049250313e-16*wmag
                       +8.0*4.930380657631324e-32*(fabs(logs[next])+fabs(w.re));
            if(!isfinite(logz.hi) || !pt_finite(w)) { flags[index]=3; return; }
            if(direct && n>=1) {
                double gap=(logz.hi-log(1e10))+logz.lo;
                if(gap>tol) { out[index]=1; stops[index]=n; return; }
                if(pt_detect() && fabs(gap)<=tol) { flags[index]=3; return; }
            }
            pt_c value;
            if(logz.hi+logz.lo<log(DBL_MIN)) value={0.,0.};
            else if(cancel && w.re<log(DBL_MAX) && w.re>log(DBL_MIN)) {
                // Multiplication preserves the small pixel; never form Z+delta.
                value=pt_mul(zn,pt_exp(w));
                if(!pt_finite(value) || hypot(value.re,value.im)<DBL_MIN) {
                    pt_dd th=pt_ddadd(argz[next],arglo[next],w.im);
                    value=pt_polar(logz,th);
                }
            } else {
                pt_dd th=pt_ddadd(argz[next],arglo[next],w.im);
                value=pt_polar(logz,th);
            }
            if(!pt_finite(value)) { flags[index]=3; return; }
            e=0; delta=value; rebases[index]++;
        } else { e=next; delta=dn; }
        // State and pixel constants fully determine every subsequent step.
        // Only an identical e=2 state (including signed zero bits) can repeat
        // the already non-escaping interval from n=0; tolerance is unsound.
        if(e==2) {
            if(!saved && n==0) { initial=delta; saved=true; }
            else if(saved && pt_cycle(delta,initial)) {
                stops[index]=maxiter; cycles[index]=1; return;
            }
        }
        if(n>=0 && hypot(delta.re,delta.im)<DBL_MIN && hypot(lambda.re,lambda.im)<0x1p-900
                && (gamma.re!=0.0 || gamma.im!=0.0)) { flags[index]=2; return; }
    }
    stops[index]=maxiter;
}

#ifdef PT_TEST
extern "C" __global__ void pt_functions(const pt_c* input, int count, pt_c* em, pt_c* lp) {
    int t=blockDim.x*blockIdx.x+threadIdx.x;
    if(t<count) { em[t]=pt_increment(input[t]); lp[t]=pt_lambda(input[t]); }
}
#endif

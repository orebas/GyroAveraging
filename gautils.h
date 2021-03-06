

#ifndef GYROAVERAGING_UTILS_H
#define GYROAVERAGING_UTILS_H

#include <iostream>
#include <cmath>

#include <functional>
#include <memory>
#include <utility>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/special_functions/next.hpp>
#include <chrono>

template <typename F>
class gsl_quad_old {
    F f;
    int limit;
    std::unique_ptr<gsl_integration_workspace,
                    std::function<void(gsl_integration_workspace *)> >
        workspace;

    static double gsl_wrapper(double x, void *p) {
        gsl_quad_old *t = reinterpret_cast<gsl_quad_old *>(p);
        return t->f(x);
    }

   public:
    gsl_quad_old(F f, int limit)
        : f(f), limit(limit), workspace(gsl_integration_workspace_alloc(limit), gsl_integration_workspace_free) {}

    double integrate(double min, double max, double epsabs, double epsrel) {
        gsl_function gsl_f;
        gsl_f.function = &gsl_wrapper;
        gsl_f.params = this;

        double result, error;
        if (!std::isinf(min) && !std::isinf(max)) {
            gsl_integration_qags(&gsl_f, min, max,
                                 epsabs, epsrel, limit,
                                 workspace.get(), &result, &error);
        } else if (std::isinf(min) && !std::isinf(max)) {
            gsl_integration_qagil(&gsl_f, max,
                                  epsabs, epsrel, limit,
                                  workspace.get(), &result, &error);
        } else if (!std::isinf(min) && std::isinf(max)) {
            gsl_integration_qagiu(&gsl_f, min,
                                  epsabs, epsrel, limit,
                                  workspace.get(), &result, &error);
        } else {
            gsl_integration_qagi(&gsl_f,
                                 epsabs, epsrel, limit,
                                 workspace.get(), &result, &error);
        }

        return result;
    }
};

template <typename F>
double quad_old(F func,
                std::pair<double, double> const &range,
                double epsabs = 1.49e-12, double epsrel = 1.49e-12,
                int limit = 10000) {
    return gsl_quad_old<F>(func, limit).integrate(range.first, range.second, epsabs, epsrel);
}  //end

template <typename F>
class gsl_quad {
    F f;
    int limit;
    std::unique_ptr<gsl_integration_cquad_workspace,
                    std::function<void(gsl_integration_cquad_workspace *)> >
        workspace;

    static double gsl_wrapper(double x, void *p) {
        gsl_quad *t = reinterpret_cast<gsl_quad *>(p);
        return t->f(x);
    }

   public:
    gsl_quad(F f, int limit)
        : f(f), limit(limit), workspace(gsl_integration_cquad_workspace_alloc(limit), gsl_integration_cquad_workspace_free) {}

    double integrate(double min, double max, double epsabs, double epsrel) {
        gsl_function gsl_f;
        gsl_f.function = &gsl_wrapper;
        gsl_f.params = this;

        double result = 0, error = 0;

        gsl_integration_cquad(&gsl_f, min, max,
                              epsabs, epsrel, workspace.get(), &result, &error, nullptr);

        /*if (!std::isinf(min) && !std::isinf(max)) {
            gsl_integration_qags(&gsl_f, min, max,
                                 epsabs, epsrel, limit,
                                 workspace.get(), &result, &error);
        } else if (std::isinf(min) && !std::isinf(max)) {
            gsl_integration_qagil(&gsl_f, max,
                                  epsabs, epsrel, limit,
                                  workspace.get(), &result, &error);
        } else if (!std::isinf(min) && std::isinf(max)) {
            gsl_integration_qagiu(&gsl_f, min,
                                  epsabs, epsrel, limit,
                                  workspace.get(), &result, &error);
        } else {
            gsl_integration_qagi(&gsl_f,
                                 epsabs, epsrel, limit,
                                 workspace.get(), &result, &error);
        }*/

        return result;
    }
};

template <typename F>
double quad(F func,
            std::pair<double, double> const &range,
            double epsabs = 1.49e-10, double epsrel = 1.49e-10,
            int limit = 10000) {
    return gsl_quad<F>(func, limit).integrate(range.first, range.second, epsabs, epsrel);
}

/*int main() {
    std::cout << "\\int_0^1 x^2 dx = "
              << quad([](double x) { return x * x; }, {0, 1}) << '\n'
              << "\\int_1^\\infty x^{-2} dx = "
              << quad([](double x) { return 1 / (x * x); }, {1, INFINITY}) << '\n'
              << "\\int_{-\\infty}^\\infty \\exp(-x^2) dx = "
              << quad([](double x) { return std::exp(-x * x); }, {-INFINITY, INFINITY}) << '\n';
}*/

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    if (!v.empty()) {
        out << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
        out << "\b\b]";
    }
    return out;
}

template <typename T, size_t s>
std::ostream &operator<<(std::ostream &out, const std::array<T, s> &v) {
    if (!v.empty()) {
        out << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
        out << "\b\b]";
    }
    return out;
}

namespace OOGA {

template <typename TimeT = std::chrono::milliseconds>
struct measure {
    template <typename F, typename... Args>
    static typename TimeT::rep execution(F func, Args &&... args) {
        auto start = std::chrono::steady_clock::now();

        // Now call the function with all the parameters you need.
        func(std::forward<Args>(args)...);

        auto duration = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start);

        return duration.count();
    }

    template <typename F, typename... Args>
    static typename TimeT::rep execution2(F func, Args &&... args) {
        auto start = std::chrono::steady_clock::now();

        // Now call the function with all the parameters you need.
        func(std::forward<Args>(args)...);

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);

        auto start2 = std::chrono::steady_clock::now();
        long iters = 1000.0 / (duration.count() + 1);
        iters = std::max(1L, iters);
        iters = std::min(iters, 50L);
        for (long i = 0; i < iters; ++i) {
            func(std::forward<Args>(args)...);
        }
        auto duration2 = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start2);
        return duration2.count() / iters;
    }
};

template <class RealT = double>
struct indexedPoint {
    RealT xvalue = 0;
    RealT yvalue = 0;
    RealT s = 0;  // from 0 to 2*Pi only please.
    explicit indexedPoint(RealT x = 0, RealT y = 0, RealT row = 0)
        : xvalue(x), yvalue(y), s(row) {}
};

template <class RealT = double>
inline void integrand(const RealT rho, const RealT xc, const RealT yc,
                      const RealT gamma, RealT &xn, RealT &yn) {
    xn = xc + rho * std::sin(gamma);
    yn = yc - rho * std::cos(gamma);
}

template <class RealT = double>
std::array<RealT, 4> arcIntegral(RealT rho, RealT xc, RealT yc, RealT s0, RealT s1) {
    std::array<RealT, 4> coeffs = {};
    RealT coss1 = std::cos(s1), coss0 = std::cos(s0), sins0 = std::sin(s0),
          sins1 = std::sin(s1);
    coeffs[0] = s1 - s0;
    coeffs[1] = (s1 * xc - rho * coss1) - (s0 * xc - rho * coss0);
    coeffs[2] = (s1 * yc - rho * sins1) - (s0 * yc - rho * sins0);
    coeffs[3] = (s1 * xc * yc - rho * yc * coss1 - rho * xc * sins1 +
                 rho * rho * coss1 * coss1 / 2.0) -
                (s0 * xc * yc - rho * yc * coss0 - rho * xc * sins0 +
                 rho * rho * coss0 * coss0 / 2.0);
    return coeffs;
}

constexpr double pi = 3.14159265358979323846264338327950288419716939937510;

double inline DCTBasisFunction(double p, double q, int i, int j, int N) {
    double a = 2, b = 2;
    //if (p == 0) a = 1;
    //if (q == 0) b = 1;
    return std::sqrt(a / N) * std::sqrt(b / N) *
           std::cos(pi * (2 * i + 1) * p / (2.0 * N)) *
           std::cos(pi * (2 * j + 1) * q / (2.0 * N));
}

double inline DCTBasisFunction2(double p, double q, double i, double j, int N) {  //TODO(orebas) MODIFY FOR NONSQUARE (N should be two paramters)
    double a = 2, b = 2;
    if (p == 0) {
        a = 1;
    }
    if (q == 0) {
        b = 1;
    }
    return a * b *
           std::cos(pi * (2 * i + 1) * p / (2.0 * N)) *
           std::cos(pi * (2 * j + 1) * q / (2.0 * N));
}

double inline fast_cheb_t(int n, double x) {
    return std::cos(n * std::acos(x));
}

double inline chebBasisFunction(int p, int q, double x, double y, int N) {
    double a = 1, b = 1;
    if (p == 0) {
        a = 0.5;
    }
    if (q == 0) {
        b = 0.5;
    }
    if (p == N - 1) {
        a = 0.5;
    }
    if (q == N - 1) {
        b = 0.5;
    }
    //return a * b * boost::math::chebyshev_t(p, -x) * boost::math::chebyshev_t(q, -y);
    return a * b * fast_cheb_t(p, -x) * fast_cheb_t(q, -y);
}  // namespace OOGA

/*void inline arcIntegralBicubic(
    std::array<double, 16> &coeffs,  // this function is being left in double,
                                     // intentionally, for now
    double rho, double xc, double yc, double s0, double s1);*/

template <typename TFunc, class RealT = double>
RealT TanhSinhIntegrate(RealT x, RealT y, TFunc f) {
    boost::math::quadrature::tanh_sinh<RealT> integrator;
    return integrator.integrate(f, x, y);
}

template <class RealT = double>
inline RealT BilinearInterpolation(RealT q11, RealT q12, RealT q21, RealT q22,
                                   RealT x1, RealT x2, RealT y1, RealT y2,
                                   RealT x, RealT y) {
    RealT x2x1, y2y1, x2x, y2y, yy1, xx1;
    x2x1 = x2 - x1;
    y2y1 = y2 - y1;
    x2x = x2 - x;
    y2y = y2 - y;
    yy1 = y - y1;
    xx1 = x - x1;
    return 1.0 / (x2x1 * y2y1) *
           (q11 * x2x * y2y + q21 * xx1 * y2y + q12 * x2x * yy1 +
            q22 * xx1 * yy1);
}

template <typename TFunc, class RealT = double>
inline RealT TrapezoidIntegrate(RealT x, RealT y, TFunc f) {
    using boost::math::quadrature::trapezoidal;
    return trapezoidal(f, x, y, 1e-12, 20);

    //boost::math::quadrature::tanh_sinh<RealT> integrator;
    //return integrator.integrate(f, x, y);  //TODO REPLACE
}

template <typename TFunc, class RealT = double>
inline RealT BOOSTGKIntegrate(RealT x, RealT y, TFunc f) {
    using boost::math::quadrature::gauss_kronrod;
    double error;
    double Q = gauss_kronrod<double, 15>::integrate(f, x, y, 16, 1e-12, &error);

    return Q;

    //boost::math::quadrature::tanh_sinh<RealT> integrator;
    //return integrator.integrate(f, x, y);  //TODO REPLACE
}

template <typename TFunc, class RealT = double>
inline RealT GSLIntegrate(RealT x, RealT y, TFunc f) {
    //double newq = quad(f, {x, y});
    //double oldq = quad_old(f, {x, y});
    //double trap = TrapezoidIntegrate(x, y, f);
    double GK = BOOSTGKIntegrate(x, y, f);
    //double tanhsinh = TanhSinhIntegrate(x, y, f);
    //std::array<double, 3> values = {0, oldq - newq, GK - newq};
    //double localmin = *std::min_element(values.begin(), values.end());
    //double localmax = *std::max_element(values.begin(), values.end());
    //if (localmax - localmin > 1e-6) {
    //   std::cout << "Quadrature accuracy in question: " << "value:" << newq << " " << localmax - localmin << " " << values << std::endl;
    //abort();
    //}
    return GK;
}

template <class T = double>
class Array3d {
   private:
    int w;
    int h;
    int d;

   public:
    std::vector<T> data;

    Array3d(int width, int height, int depth) : w(width), h(height), d(depth), data(w * h * d, 0) {}

    inline T &at(int x, int y, int z) { return data[x * h * d + y * d + z]; }

    inline T at(int x, int y, int z) const {
        return data[x * h * d + y * d + z];
    }

    inline T &operator()(int x, int y, int z) {
        return data[x * h * d + y * d + z];
    }

    inline T operator()(int x, int y, int z) const {
        return data[x * h * d + y * d + z];
    }
    inline int width() { return w; }
    inline int height() { return h; }
    inline int depth() { return d; }
};

template <class T = double>
class Array4d {
   private:
    int w;
    int h;
    int d;
    int l;

   public:
    std::vector<T> data;

    Array4d(int r, int s, int t, int u) : w(r), h(s), d(t), l(u), data(w * h * d * l, 0) {}

    inline T &at(int x, int y, int z, int t) {
        return data[x * h * d * l + y * d * l + z * l + t];
    }

    inline T at(int x, int y, int z, int t) const {
        return data[x * h * d * l + y * d * l + z * l + t];
    }

    inline T &operator()(int x, int y, int z, int t) {
        return data[x * h * d * l + y * d * l + z * l + t];
    }

    inline T operator()(int x, int y, int z, int t) const {
        return data[x * h * d * l + y * d * l + z * l + t];
    }

    inline int internalRef(int x, int y, int z, int t) {
        return x * h * d * l + y * d * l + z * l + t;
    }
};

template <typename TFunc, class RealT = double>
inline RealT FixedNTrapezoidIntegrate(RealT x, RealT y, TFunc f,
                                      int n) {  // used to be trapezoid rule.
    RealT h = (y - x) / (n - 1);
    RealT sum = 0;
    RealT fx1 = f(x);
    for (int i = 1; i <= n; i++) {
        RealT x2 = x + i * h;
        RealT fx2 = f(x2);
        sum += (fx1 + fx2) * h;
        fx1 = fx2;
    }
    return 0.5 * sum;
}

inline double BilinearInterpolation(double q11, double q12, double q21,
                                    double q22, double x1, double x2, double y1,
                                    double y2, double x, double y);

template <class RealT = double>
std::vector<RealT> LinearSpacedArray(RealT a, RealT b, int N) {
    RealT h = (b - a) / static_cast<RealT>(N - 1);
    std::vector<RealT> xs(N);
    auto x = xs.begin();
    RealT val = a;
    for (; x != xs.end(); ++x) {
        *x = val;
        val += h;
    }
    return xs;
}

template <class RealT = double>
struct sparseOffset {  // no constructor.
    int target, source;
    std::array<RealT, 4> coeffs;
};

template <class RealT = double>
struct LTOffset {
    int target, source;
    RealT coeff;
};

inline std::array<double, 4> operator+(const std::array<double, 4> &l, const std::array<double, 4> &r) {
    std::array<double, 4> ret = {};
    ret[0] = l[0] + r[0];
    ret[1] = l[1] + r[1];
    ret[2] = l[2] + r[2];
    ret[3] = l[3] + r[3];
    return ret;
}

inline std::array<float, 4> operator+(const std::array<float, 4> &l, const std::array<float, 4> &r) {
    std::array<float, 4> ret = {};
    ret[0] = l[0] + r[0];
    ret[1] = l[1] + r[1];
    ret[2] = l[2] + r[2];
    ret[3] = l[3] + r[3];
    return ret;
}

}  // namespace OOGA

constexpr int inline mymax(int a, int b) {
    if (a > b) {
        return a;
    }
    return b;
}

template <class RealT = double>
std::vector<RealT> chebPoints(int N) {
    std::vector<RealT> xs(N);
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = -1 * std::cos(i * OOGA::pi / (N - 1));
    }
    return xs;
}

#endif  //GYROAVERGING_UTILS_H
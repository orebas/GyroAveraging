// file foobar.h:
#ifndef GYROAVERAGING_GA_H
#define GYROAVERAGING_GA_H
// ... declarations ...
#endif  // GYROAVERAGING_GA_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <vector>

void inline arcIntegralBicubic(
    std::array<double, 16> &coeffs,  // this function is being left in double,
                                     // intentionally, for now
    double rho, double xc, double yc, double s0, double s1);

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
    return trapezoidal(f, x, y);
}

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

template <int w, int h, int d, class T = double>
class Array3d {
   public:
    std::vector<T> data;

   public:
    Array3d() : data(w * h * d, 0) {}

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
};

template <int w, int h, int d, int l, class T = double>
class Array4d {
   public:
    std::vector<T> data;

   public:
    Array4d() : data(w * h * d * l, 0) {}

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
};

constexpr double pi = 3.1415926535897932384626433832795028841971;

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
    RealT val;
    for (val = a; x != xs.end(); ++x, val += h) {
        *x = val;
    }
    return xs;
}

template <class RealT = double>
struct sparseOffset {  // no constructor.
    int target, source;
    RealT coeffs[4];
};

template <class RealT = double>
struct LTOffset {
    int target, source;
    RealT coeff;
};

std::array<double, 4> operator+(const std::array<double, 4> &l,
                                const std::array<double, 4> &r) {
    std::array<double, 4> ret;
    ret[0] = l[0] + r[0];
    ret[1] = l[1] + r[1];
    ret[2] = l[2] + r[2];
    ret[3] = l[3] + r[3];
    return ret;
}

std::array<float, 4> operator+(const std::array<float, 4> &l,
                               const std::array<float, 4> &r) {
    std::array<float, 4> ret;
    ret[0] = l[0] + r[0];
    ret[1] = l[1] + r[1];
    ret[2] = l[2] + r[2];
    ret[3] = l[3] + r[3];
    return ret;
}

void testInterpImprovement();

template <int rhocount, int xcount, int ycount, class RealT = double>
class GyroAveragingGrid {
   public:
    typedef Array3d<rhocount, xcount, ycount, RealT> fullgrid;
    typedef Array4d<rhocount, xcount, ycount, 4, RealT> fullgridInterp;
    typedef Array4d<rhocount, xcount, ycount, 4, RealT>
        derivsGrid;  // at each rho,x,y calculate [f,f_x,f_y,f_xy]
    typedef Array4d<rhocount, xcount, ycount, 16, RealT> bicubicParameterGrid;
    typedef Eigen::SparseMatrix<RealT, Eigen::RowMajor> SpM;
    typedef Eigen::Triplet<RealT> SpT;

   private:
    std::vector<RealT> rhoset;
    std::vector<RealT> xset;
    std::vector<RealT> yset;
    fullgrid gridValues;     // input values of f
    fullgrid almostExactGA;  // stores value of GA calculated as trapezoid rule
                             // on input f
    fullgrid truncatedAlmostExactGA;  // above, except f hard truncated to 0
                                      // outside grid
    fullgrid trapezoidInterp;         // GA calculated as trapezoid rule on
                                      // interpolated, truncated f
    fullgrid bicubicInterp;
    fullgrid fastGALTResult;
    fullgrid BCResult;
    fullgrid
        analytic_averages;  // stores value of expected GA computed analytically
    fullgridInterp
        interpParameters;  // will store the bilinear interp parameters.
    bicubicParameterGrid bicubicParameters;
    SpM LTOffsetTensor;
    SpM BCOffsetTensor;
    SpM FDTensor;  // finite difference tensor, used to populate derivs from
                   // gridValues
    derivsGrid derivs;

    void csvPrinter(const fullgrid &m, int rho) {
        for (int j = 0; j < xcount; j++) {
            for (int k = 0; k < ycount; k++) {
                std::cout << m(rho, j, k) << ",";
            }
            std::cout << std::endl;
        }
    }

    void csvPrinterDiff(const fullgrid &m, const fullgrid &n, int rho) {
        for (int j = 0; j < xcount; j++) {
            for (int k = 0; k < ycount; k++) {
                std::cout << m(rho, j, k) - n(rho, j, k) << ",";
            }
            std::cout << std::endl;
        }
    }

    void clearGrid(fullgrid &m) {
        for (auto i = 0; i < rhocount; i++) {
            for (auto j = 0; j < xcount; j++)
                for (auto k = 0; k < ycount; k++) {
                    m(i, j, k) = 0;
                }
        }
    }
    RealT RMSNorm(const fullgrid &m, int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++)
                result += m(rho, j, k) * m(rho, j, k);
        return std::sqrt(result / (xcount * ycount));
    }
    RealT maxNorm(const fullgrid &m, int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++)
                result = std::max(result, std::abs(m(rho, j, k)));
        return result;
    }

    RealT RMSNormDiff(const fullgrid &m1, const fullgrid &m2, int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++) {
                RealT t = m1(rho, j, k) - m2(rho, j, k);
                result += t * t;
            }
        return std::sqrt(result / (xcount * ycount));
    }
    RealT maxNormDiff(const fullgrid &m1, const fullgrid &m2, int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++) {
                RealT t = m1(rho, j, k) - m2(rho, j, k);
                result = std::max(result, std::abs(t));
            }
        return result;
    }

    // below fills a grid, given a function of rho, x, and y
    template <typename TFunc>
    void fill(fullgrid &m, TFunc f) {
#pragma omp parallel for
        for (auto i = 0; i < rhocount; i++) {
            for (auto j = 0; j < xcount; j++)
                for (auto k = 0; k < ycount; k++) {
                    m(i, j, k) = f(rhoset[i], xset[j], yset[k]);
                }
        }
    }
    // below fills a grid, given a function of i,j,k
    template <typename TFunc>
    void fillbyindex(fullgrid &m, TFunc f) {
#pragma omp parallel for
        for (int i = 0; i < rhocount; i++)
            for (int j = 0; j < xcount; j++)
                for (int k = 0; k < ycount; k++) {
                    m(i, j, k) = f(i, j, k);
                }
    }
    template <typename TFunc1>
    void fillAlmostExactGA(fullgrid &m, TFunc1 f) {
        fillbyindex(
            m,
            [&](int i, int j, int k)
                -> RealT {  // adaptive trapezoid rule on actual input function.
                RealT xc = xset[j];
                RealT yc = yset[k];
                if (rhoset[i] == 0) return f(i, xc, yc);
                auto new_f = [&](RealT x) -> RealT {
                    return f(rhoset[i], xc + rhoset[i] * std::sin(x),
                             yc - rhoset[i] * std::cos(x));
                };
                RealT result =
                    TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
                return result;
            });
    }

    template <typename TFunc1>
    void fillTruncatedAlmostExactGA(fullgrid &m, TFunc1 f) {
        fillbyindex(m, [&](int i, int j, int k) -> RealT {
            RealT xc = xset[j];
            RealT yc = yset[k];
            if (rhoset[i] == 0) return f(i, xc, yc);
            auto new_f = [&](RealT x) -> RealT {
                RealT ex = xc + rhoset[i] * std::sin(x);
                RealT why = yc - rhoset[i] * std::cos(x);
                if ((ex < xset[0]) || (ex > xset.back())) return 0;
                if ((why < yset[0]) || (why > yset.back())) return 0;
                return f(rhoset[i], ex, why);
            };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
    }

    template <typename TFunc1>
    void fillTrapezoidInterp(
        fullgrid &m, TFunc1 f) {  // this calls interp2d. gridValues must be
                                  // filled, but we don't need setupInterpGrid.
        fillbyindex(m, [&](int i, int j, int k) -> RealT {
            RealT xc = xset[j];
            RealT yc = yset[k];
            if (rhoset[i] == 0) return f(rhoset[i], xc, yc);
            auto new_f = [&](RealT x) -> RealT {
                return interp2d(i, xc + rhoset[i] * std::sin(x),
                                yc - rhoset[i] * std::cos(x));
            };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
    }
    // below requires the bicubic parameter grid to be populated.
    void fillBicubicInterp(fullgrid &m) {
        fillbyindex(m, [&](int i, int j, int k) -> RealT {
            RealT xc = xset[j];
            RealT yc = yset[k];
            if (rhoset[i] == 0) return gridValues(i, j, k);

            auto new_f = [&](RealT x) -> RealT {
                return interpNaiveBicubic(i, xc + rhoset[i] * std::sin(x),
                                          yc - rhoset[i] * std::cos(x));
            };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
    }

   public:
    GyroAveragingGrid(const std::vector<RealT> &rhos,
                      const std::vector<RealT> &xes,
                      const std::vector<RealT> &yies)
        : rhoset(rhos), xset(xes), yset(yies) {
        assert(rhocount == rhos.size());
        assert(xcount == xes.size());
        assert(ycount == yies.size());

        std::sort(xset.begin(), xset.end());
        std::sort(yset.begin(), yset.end());
        std::sort(rhoset.begin(), rhoset.end());
    }
    void interpIndexSearch(const RealT x, const RealT y, int &xindex,
                           int &yindex);
    inline void integrand(const RealT rho, const RealT xc, const RealT yc,
                          const RealT gamma, RealT &xn, RealT &yn) {
        xn = xc + rho * std::sin(gamma);
        yn = yc - rho * std::cos(gamma);
    }
    void setupInterpGrid();
    void setupDerivsGrid();  // make accuracy a variable later
    void setupBicubicGrid();
    // void setupBicubicGrid2();
    void assembleFastGACalc(void);

    void assembleFastBCCalc(void);
    void fastLTCalcOffset();
    void fastBCCalcOffset();
    std::array<RealT, 4> arcIntegral(RealT rho, RealT xc, RealT yc, RealT s0,
                                     RealT s1);
    // std::array<RealT, 16> arcIntegralBicubic(RealT rho, RealT xc, RealT yc,
    // RealT s0, RealT s1);
    template <typename TFunc1, typename TFunc2>
    void GyroAveragingTestSuite(TFunc1 f, TFunc2 analytic);

    template <typename TFunc1, typename TFunc2>
    void compactErrorAnalysis(TFunc1 f, TFunc2 analytic);

    template <typename TFunc1, typename TFunc2>
    void GPUTestSuite(TFunc1 f, TFunc2 analytic);

    template <typename TFunc1, typename TFunc2>
    void GPUTestSuiteBC(TFunc1 f, TFunc2 analytic);

    template <typename TFunc1, typename TFunc2>
    void InterpErrorAnalysis(TFunc1 f, TFunc2 analytic);

    template <typename TFunc1, typename TFunc2>
    void errorAnalysis(TFunc1 f, TFunc2 analytic);

    template <typename TFunc1, typename TFunc2, typename TFunc3,
              typename TFunc4>
    void derivsErrorAnalysis(TFunc1 f, TFunc2 f_x, TFunc3 f_y, TFunc4 f_xy);

    RealT interp2d(int rhoindex, const RealT x, const RealT y);
    RealT interpNaiveBicubic(int rhoindex, const RealT x, const RealT y);
    RealT interpNaiveBicubic2(int rhoindex, const RealT x, const RealT y);

    friend void testInterpImprovement();
};

template <class RealT = double>
struct gridDomain {
    RealT xmin = 0, xmax = 0, ymin = 0, ymax = 0, rhomin = 0, rhomax = 0;
};

template <class RealT = double>
struct indexedPoint {
    RealT xvalue = 0;
    RealT yvalue = 0;
    RealT s = 0;  // from 0 to 2*Pi only please.
    indexedPoint(RealT x = 0, RealT y = 0, RealT row = 0)
        : xvalue(x), yvalue(y), s(row) {}
};

template <typename TFunc1, typename TFunc2, class RealT = double>
void interpAnalysis(const gridDomain<RealT> &g, TFunc1 f, TFunc2 analytic);

template <typename TFunc1, typename TFunc2, class RealT = double>
void errorAnalysis(const gridDomain<RealT> &g, TFunc1 f, TFunc2 analytic);

template <int i, typename TFunc1, typename TFunc2, class RealT = double>
void interpAnalysisInnerLoop(const gridDomain<RealT> &g, TFunc1 f,
                             TFunc2 analytic);

template <int i, typename TFunc1, typename TFunc2, class RealT = double>
void errorAnalysisInnerLoop(const gridDomain<RealT> &g, TFunc1 f,
                            TFunc2 analytic);

template <typename TFunc1, typename TFunc2, typename TFunc3, typename TFunc4,
          class RealT = double>
void derivTest(const gridDomain<RealT> &g, TFunc1 f, TFunc2 f_x, TFunc3 f_y,
               TFunc4 f_xy);

template <class RealT = double>
void inline arcIntegralBicubic(std::array<RealT, 16> &coeffs, RealT rho,
                               RealT xc, RealT yc, RealT s0, RealT s1);

template <class RealT = double>
void testArcIntegralBicubic();

/* setupInterpGrid is going to fill a 4d array with the A,B,C,D coefficients
such that for each i,j,k with j,k not on the top or right edge for every x,y
inside the box xset[j],xset[j+1],yset[k],yset[k+1] interp2d(i,x,y,etc) =
A+Bx+Cy+Dxy (approx)
*/
template <int rhocount, int xcount, int ycount, class RealT>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::setupInterpGrid() {
    // using namespace Eigen;
    for (int i = 0; i < rhocount; i++) {
        interpParameters(i, xcount - 1, ycount - 1, 0) =
            0;  // we set the top-right grid points to 0
        interpParameters(i, xcount - 1, ycount - 1, 1) =
            0;  // so we can point to them later.
        interpParameters(i, xcount - 1, ycount - 1, 2) =
            0;  // otherwise the right and top edges are not used.
        interpParameters(i, xcount - 1, ycount - 1, 3) = 0;
        for (int j = 0; j < xcount - 1; j++)
            for (int k = 0; k < ycount - 1; k++) {
                RealT Q11 = gridValues(i, j, k), Q12 = gridValues(i, j + 1, k),
                      Q21 = gridValues(i, j, k + 1),
                      Q22 = gridValues(i, j + 1, k + 1);

                RealT x = xset[j], a = xset[j + 1], y = yset[k],
                      b = yset[k + 1];
                RealT denom = (a - x) * (b - y);
                interpParameters(i, j, k, 0) =
                    (a * b * Q11 - a * y * Q12 - b * x * Q21 + x * y * Q22) /
                    denom;
                interpParameters(i, j, k, 1) =
                    (-b * Q11 + y * Q12 + b * Q21 - y * Q22) / denom;
                interpParameters(i, j, k, 2) =
                    (-a * Q11 + a * Q12 + x * Q21 - x * Q22) / denom;
                interpParameters(i, j, k, 3) = (Q11 - Q12 - Q21 + Q22) / denom;
            }
    }
}

// Using an existing derivs grid, the below computes bicubic interpolation
// parameters.
// 16 parameters per patch, and the bicubic is of the form a_{ij} x^i y^j for 0
// \leq x,y \leq 3
template <int rhocount, int xcount, int ycount, class RealT>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::setupBicubicGrid() {
    using namespace Eigen;
    bicubicParameterGrid &b = bicubicParameters;
    derivsGrid &d = derivs;
    for (int i = 0; i < rhocount; i++) {
        // we explicitly rely on parameters being initialized to 0, including
        // the top and right sides.
        for (int j = 0; j < xcount - 1; j++)
            for (int k = 0; k < ycount - 1; k++) {
                RealT x0 = xset[j], x1 = xset[j + 1];
                RealT y0 = yset[k], y1 = yset[k + 1];
                Matrix<RealT, 4, 4> X, Y, RHS, A, temp1, temp2;

                RHS << d(i, j, k, 0), d(i, j, k + 1, 0), d(i, j, k, 2),
                    d(i, j, k + 1, 2), d(i, j + 1, k, 0), d(i, j + 1, k + 1, 0),
                    d(i, j + 1, k, 2), d(i, j + 1, k + 1, 2), d(i, j, k, 1),
                    d(i, j, k + 1, 1), d(i, j, k, 3), d(i, j, k + 1, 3),
                    d(i, j + 1, k, 1), d(i, j + 1, k + 1, 1), d(i, j + 1, k, 3),
                    d(i, j + 1, k + 1, 3);
                X << 1, x0, x0 * x0, x0 * x0 * x0, 1, x1, x1 * x1, x1 * x1 * x1,
                    0, 1, 2 * x0, 3 * x0 * x0, 0, 1, 2 * x1, 3 * x1 * x1;
                Y << 1, 1, 0, 0, y0, y1, 1, 1, y0 * y0, y1 * y1, 2 * y0, 2 * y1,
                    y0 * y0 * y0, y1 * y1 * y1, 3 * y0 * y0, 3 * y1 * y1;

                // temp1 = X.fullPivLu().inverse(); //this line crashes on my
                // home machine without optimization turned on. temp2 =
                // Y.fullPivLu().inverse(); // we should take out the Eigen
                // dependency methinks.  TODO

                A = X.inverse() * RHS * Y.inverse();
                for (int t = 0; t < 16; ++t) b(i, j, k, t) = A(t % 4, t / 4);
            }
    }
}

// setupDerivsGrid assumes the values of f are already in gridValues
// then populates derivs with vectors of the form [f,f_x,f_y, f_xy]
// we are using finite difference
// the below is a linear transform, and we will hopefully get it into a (sparse)
// matrix soon. the below uses 5-point stencils (including at edges) for f_x and
// f_y 16-point stencils for f_xy where derivatives are available
// 4-point stencils for f_xy one row or column from edges
// f_xy at edges is hardcoded to 0.
template <int rhocount, int xcount, int ycount, class RealT>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::setupDerivsGrid() {
    RealT ydenom = yset[1] - yset[0];
    RealT xdenom = xset[1] - xset[0];
    fullgrid &g = gridValues;
    for (int i = 0; i < rhocount; i++) {
        for (int j = 0; j < xcount; j++) {
            for (int k = 0; k < ycount; k++) {
                derivs(i, j, k, 0) = g(i, j, k);
            }
        }
        for (int k = 0; k < ycount; k++) {
            derivs(i, 0, k, 1) = 0;
            derivs(i, xcount - 1, k, 1) = 0;
            derivs(i, 1, k, 1) =
                (-3.0 * g(i, 0, k) + -10.0 * g(i, 1, k) + 18 * g(i, 2, k) +
                 -6 * g(i, 3, k) + 1 * g(i, 4, k)) /
                (12.0 * xdenom);

            derivs(i, xcount - 2, k, 1) =
                (3.0 * g(i, xcount - 1, k) + 10.0 * g(i, xcount - 2, k) +
                 -18.0 * g(i, xcount - 3, k) + 6.0 * g(i, xcount - 4, k) +
                 -1.0 * g(i, xcount - 5, k)) /
                (12.0 * xdenom);
            for (int j = 2; j <= xcount - 3; j++)
                derivs(i, j, k, 1) =
                    (1.0 * g(i, j - 2, k) + -8.0 * g(i, j - 1, k) +
                     0.0 * g(i, j, k) + 8.0 * g(i, j + 1, k) +
                     -1.0 * g(i, j + 2, k)) /
                    (12.0 * xdenom);
        }
        for (int j = 0; j < xcount; j++) {
            derivs(i, j, 0, 2) = 0;
            derivs(i, j, ycount - 1, 2) = 0;
            derivs(i, j, 1, 2) =
                (-3.0 * g(i, j, 0) + -10.0 * g(i, j, 1) + 18.0 * g(i, j, 2) +
                 -6.0 * g(i, j, 3) + 1.0 * g(i, j, 4)) /
                (12.0 * ydenom);
            derivs(i, j, ycount - 2, 2) =
                (3.0 * g(i, j, ycount - 1) + 10.0 * g(i, j, ycount - 2) +
                 -18.0 * g(i, j, ycount - 3) + 6.0 * g(i, j, ycount - 4) +
                 -1 * g(i, j, ycount - 5)) /
                (12.0 * ydenom);
            for (int k = 2; k < ycount - 3; k++) {
                derivs(i, j, k, 2) =
                    (1.0 * g(i, j, k - 2) + -8.0 * g(i, j, k - 1) +
                     0.0 * g(i, j, k) + 8.0 * g(i, j, k + 1) +
                     -1.0 * g(i, j, k + 2)) /
                    (12.0 * ydenom);
            }
        };
        for (int j = 2; j < xcount - 2; ++j) {
            for (int k = 2; k < ycount - 2; ++k) {
                derivs(i, j, k, 3) =
                    (8 * (g(i, j + 1, k - 2) + g(i, j + 2, k - 1) +
                          g(i, j - 2, k + 1) + g(i, j - 1, k + 2)) +
                     -8 * (g(i, j - 1, k - 2) + g(i, j - 2, k - 1) +
                           g(i, j + 1, k + 2) + g(i, j + 2, k + 1)) +
                     -1 * (g(i, j + 2, k - 2) + g(i, j - 2, k + 2) -
                           g(i, j - 2, k - 2) - g(i, j + 2, k + 2)) +
                     64 * (g(i, j - 1, k - 1) + g(i, j + 1, k + 1) -
                           g(i, j + 1, k - 1) - g(i, j - 1, k + 1))) /
                    (144 * xdenom * ydenom);
            }
        }
        for (int j = 1; j < xcount - 1; j++) {
            derivs(i, j, 1, 3) = (g(i, j - 1, 0) + g(i, j + 1, 1 + 1) -
                                  g(i, j + 1, 1 - 1) - g(i, j - 1, 1 + 1)) /
                                 (4 * xdenom * ydenom);
            derivs(i, j, ycount - 2, 3) =
                (g(i, j - 1, ycount - 2 - 1) + g(i, j + 1, ycount - 2 + 1) -
                 g(i, j + 1, ycount - 2 - 1) - g(i, j - 1, ycount - 2 + 1)) /
                (4 * xdenom * ydenom);
        }
        for (int k = 1; k < ycount - 1; k++) {
            derivs(i, 1, k, 3) = (g(i, 1 - 1, k - 1) + g(i, 1 + 1, k + 1) -
                                  g(i, 1 + 1, k - 1) - g(i, 1 - 1, k + 1)) /
                                 (4 * xdenom * ydenom);

            derivs(i, xcount - 2, k, 3) =
                (g(i, xcount - 2 - 1, k - 1) + g(i, xcount - 2 + 1, k + 1) -
                 g(i, xcount - 2 + 1, k - 1) - g(i, xcount - 2 - 1, k + 1)) /
                (4 * xdenom * ydenom);
        }
    }
}

// arcIntegral computes the analytic integral of a bilinear function over an arc
// of a circle the circle is centered at xc,yc with radius rho, and the function
// is determined by a previously setup interp grid.

template <int rhocount, int xcount, int ycount, class RealT>
std::array<RealT, 4> GyroAveragingGrid<rhocount, xcount, ycount,
                                       RealT>::arcIntegral(RealT rho, RealT xc,
                                                           RealT yc, RealT s0,
                                                           RealT s1) {
    std::array<RealT, 4> coeffs;
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

// add handling for rho =0.
template <int rhocount, int xcount, int ycount, class RealT>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::assembleFastGACalc(
    void) {
    std::vector<std::vector<SpT>> TripletVecVec(rhocount);
#pragma omp parallel for
    for (auto i = 0; i < rhocount; i++) {
        for (auto j = 0; j < xcount; j++)
            for (auto k = 0; k < ycount; k++) {
                std::vector<indexedPoint<RealT>> intersections;
                RealT rho = rhoset[i];
                RealT xc = xset[j];
                RealT yc = yset[k];
                RealT xmin = xset[0], xmax = xset.back();
                RealT ymin = yset[0], ymax = yset.back();
                std::vector<RealT> xIntersections, yIntersections;
                // these two loops calculate all potential intersection points
                // between GA circle and the grid.
                for (auto v : xset)
                    if (std::abs(v - xc) <= rho) {
                        RealT deltax = v - xc;
                        RealT deltay = std::sqrt(rho * rho - deltax * deltax);
                        if ((yc + deltay >= ymin) && (yc + deltay <= ymax))
                            intersections.push_back(
                                indexedPoint<RealT>(v, yc + deltay, 0));
                        if ((yc - deltay >= ymin) && (yc - deltay <= ymax))
                            intersections.push_back(
                                indexedPoint<RealT>(v, yc - deltay, 0));
                    }
                for (auto v : yset)
                    if (std::abs(v - yc) <= rho) {
                        RealT deltay = v - yc;
                        RealT deltax = std::sqrt(rho * rho - deltay * deltay);
                        if ((xc + deltax >= xmin) && (xc + deltax <= xmax))
                            intersections.push_back(
                                indexedPoint<RealT>(xc + deltax, v, 0));
                        if ((xc - deltax >= xmin) && (xc - deltax <= xmax))
                            intersections.push_back(
                                indexedPoint<RealT>(xc - deltax, v, 0));
                    }
                for (auto &v : intersections) {
                    v.s = std::atan2(v.xvalue - xc, yc - v.yvalue);
                    if (v.s < 0) v.s += 2 * pi;
                    assert((0 <= v.s) && (v.s < 2 * pi));
                    assert((xc + std::sin(v.s) * rho) - v.xvalue < 1e-10);
                    assert((yc - std::cos(v.s) * rho) - v.yvalue < 1e-10);
                }
                indexedPoint<RealT> temp;
                temp.s = 0;
                intersections.push_back(temp);
                temp.s = 2 * pi;
                intersections.push_back(temp);
                std::sort(intersections.begin(), intersections.end(),
                          [](indexedPoint<RealT> a, indexedPoint<RealT> b) {
                              return a.s < b.s;
                          });
                assert(intersections.size() > 0);
                assert(intersections[0].s == 0);
                assert(intersections.back().s == (2 * pi));
                for (size_t p = 0; p < intersections.size() - 1; p++) {
                    RealT s0 = intersections[p].s, s1 = intersections[p + 1].s;
                    RealT xmid, ymid;
                    std::array<RealT, 4> coeffs;
                    int xInterpIndex = 0, yInterpIndex = 0;
                    if (s1 - s0 < 1e-12)
                        continue;  // if two of our points are equal or very
                                   // close to one another, we make the arc
                                   // larger. this will probably happen for s=0
                                   // and s=pi/2, but doesn't cost us anything.
                    integrand(rho, xc, yc, (s0 + s1) / 2, xmid,
                              ymid);  // this just calculates into (xmid,ymid)
                                      // the point half through the arc.
                    coeffs = arcIntegral(rho, xc, yc, s0, s1);
                    interpIndexSearch(xmid, ymid, xInterpIndex, yInterpIndex);

                    // begin look-thru code
                    if (!((xInterpIndex == (xcount - 1)) &&
                          (yInterpIndex == (ycount - 1)))) {
                        RealT x = xset[xInterpIndex],
                              a = xset[xInterpIndex + 1];
                        RealT y = yset[yInterpIndex],
                              b = yset[yInterpIndex + 1];
                        RealT c1 = coeffs[0] / (2 * pi);
                        RealT c2 = coeffs[1] / (2 * pi);
                        RealT c3 = coeffs[2] / (2 * pi);
                        RealT c4 = coeffs[3] / (2 * pi);
                        RealT denom = (a - x) * (b - y);

                        std::array<int, 4> LTSources({0, 0, 0, 0}),
                            LTTargets({0, 0, 0, 0});
                        std::array<RealT, 4> LTCoeffs({0, 0, 0, 0});
                        LTSources[0] =
                            &(gridValues(i, xInterpIndex, yInterpIndex)) -
                            &(gridValues(0, 0, 0));
                        LTSources[1] =
                            &(gridValues(i, xInterpIndex + 1, yInterpIndex)) -
                            &(gridValues(0, 0, 0));
                        LTSources[2] =
                            &(gridValues(i, xInterpIndex, yInterpIndex + 1)) -
                            &(gridValues(0, 0, 0));
                        LTSources[3] = &(gridValues(i, xInterpIndex + 1,
                                                    yInterpIndex + 1)) -
                                       &(gridValues(0, 0, 0));
                        LTTargets[0] =
                            &(fastGALTResult(i, j, k)) -
                            &(fastGALTResult(0, 0,
                                             0));  // eh these are all the same,
                                                   // delete the extras.
                        LTTargets[1] = &(fastGALTResult(i, j, k)) -
                                       &(fastGALTResult(0, 0, 0));
                        LTTargets[2] = &(fastGALTResult(i, j, k)) -
                                       &(fastGALTResult(0, 0, 0));
                        LTTargets[3] = &(fastGALTResult(i, j, k)) -
                                       &(fastGALTResult(0, 0, 0));

                        LTCoeffs[0] = (c1 * a * b - b * c2 - a * c3 + c4) /
                                      denom;  // coeff of Q11
                        LTCoeffs[1] = (-c4 - c1 * a * y + a * c3 + y * c2) /
                                      denom;  // etc
                        LTCoeffs[2] =
                            (b * c2 - c1 * b * x + x * c3 - c4) / denom;
                        LTCoeffs[3] =
                            (c1 * x * y - y * c2 - x * c3 + c4) / denom;
                        for (int l = 0; l < 4; l++) {
                            TripletVecVec[i].emplace_back(
                                SpT(LTTargets[l], LTSources[l], LTCoeffs[l]));
                        }
                    }
                }
            }
    }
    std::vector<SpT> Triplets;

    for (int i = 0; i < rhocount; i++) {
        for (auto iter = TripletVecVec[i].begin();
             iter != TripletVecVec[i].end(); ++iter) {
            Triplets.emplace_back(*iter);
        }
    }

    LTOffsetTensor.resize(rhocount * xcount * ycount,
                          rhocount * xcount * ycount);
    LTOffsetTensor.setFromTriplets(Triplets.begin(), Triplets.end());

    // std::cout << "Number of RealT  products needed for LT calc: " <<
    // LTOffsetTensor.nonZeros() << " and rough memory usage is " <<
    // LTOffsetTensor.nonZeros() * (sizeof(RealT) + sizeof(long)) << std::endl;
}

template <int rhocount, int xcount, int ycount, class RealT>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::assembleFastBCCalc(
    void) {  // bicubic version of the above.
    std::vector<std::vector<SpT>> TripletVecVec(rhocount);
#pragma omp parallel for
    for (auto i = 0; i < rhocount; i++) {
        for (auto j = 0; j < xcount; j++)
            for (auto k = 0; k < ycount; k++) {
                std::vector<indexedPoint<RealT>> intersections;
                RealT rho = rhoset[i];
                RealT xc = xset[j];
                RealT yc = yset[k];
                RealT xmin = xset[0], xmax = xset.back();
                RealT ymin = yset[0], ymax = yset.back();
                std::vector<RealT> xIntersections, yIntersections;
                // these two loops calculate all potential intersection points
                // between GA circle and the grid.
                for (auto v : xset)
                    if (std::abs(v - xc) <= rho) {
                        RealT deltax = v - xc;
                        RealT deltay = std::sqrt(rho * rho - deltax * deltax);
                        if ((yc + deltay >= ymin) && (yc + deltay <= ymax))
                            intersections.push_back(
                                indexedPoint<RealT>(v, yc + deltay, 0));
                        if ((yc - deltay >= ymin) && (yc - deltay <= ymax))
                            intersections.push_back(
                                indexedPoint<RealT>(v, yc - deltay, 0));
                    }
                for (auto v : yset)
                    if (std::abs(v - yc) <= rho) {
                        RealT deltay = v - yc;
                        RealT deltax = std::sqrt(rho * rho - deltay * deltay);
                        if ((xc + deltax >= xmin) && (xc + deltax <= xmax))
                            intersections.push_back(
                                indexedPoint<RealT>(xc + deltax, v, 0));
                        if ((xc - deltax >= xmin) && (xc - deltax <= xmax))
                            intersections.push_back(
                                indexedPoint<RealT>(xc - deltax, v, 0));
                    }
                for (auto &v : intersections) {
                    v.s = std::atan2(v.xvalue - xc, yc - v.yvalue);
                    if (v.s < 0) v.s += 2 * pi;
                    assert((0 <= v.s) && (v.s < 2 * pi));
                    assert((xc + std::sin(v.s) * rho) - v.xvalue < 1e-10);
                    assert((yc - std::cos(v.s) * rho) - v.yvalue < 1e-10);
                }
                indexedPoint<RealT> temp;
                temp.s = 0;
                intersections.push_back(temp);
                temp.s = 2 * pi;
                intersections.push_back(temp);
                std::sort(intersections.begin(), intersections.end(),
                          [](indexedPoint<RealT> a, indexedPoint<RealT> b) {
                              return a.s < b.s;
                          });
                assert(intersections.size() > 0);
                assert(intersections[0].s == 0);
                assert(intersections.back().s == (2 * pi));
                for (size_t p = 0; p < intersections.size() - 1; p++) {
                    RealT s0 = intersections[p].s, s1 = intersections[p + 1].s;
                    RealT xmid, ymid;
                    std::array<RealT, 16> coeffs;
                    int xInterpIndex = 0, yInterpIndex = 0;
                    if (s1 - s0 <
                        1e-12)  // TODO MAGIC NUMBER (here and another place )
                        continue;  // if two of our points are equal or very
                                   // close to one another, we make the arc
                                   // larger. this will probably happen for s=0
                                   // and s=pi/2, but doesn't cost us anything.
                    integrand(rho, xc, yc, (s0 + s1) / 2, xmid,
                              ymid);  // this just calculates into (xmid,ymid)
                                      // the point half through the arc. 
                    arcIntegralBicubic(coeffs, rho, xc, yc, s0, s1);
                    interpIndexSearch(xmid, ymid, xInterpIndex, yInterpIndex);
                    // begin look-thru code
                    if (!((xInterpIndex == (xcount - 1)) &&
                          (yInterpIndex == (ycount - 1)))) {
                        std::array<int, 16> LTSources, LTTargets;
                        std::array<RealT, 16> LTCoeffs;
                        for (int l = 0; l < 16; l++) {
                            LTSources[l] =
                                &(bicubicParameters(i, xInterpIndex,
                                                    yInterpIndex, l)) -
                                &(bicubicParameters(0, 0, 0, 0));
                            LTTargets[l] =
                                &(BCResult(i, j, k)) - &(BCResult(0, 0, 0));
                            LTCoeffs[l] = coeffs[l] / (2.0 * pi);

                            TripletVecVec[i].emplace_back(
                                SpT(LTTargets[l], LTSources[l], LTCoeffs[l]));
                        }
                    }
                }
            }
    }
    std::vector<SpT> Triplets;
    for (int i = 0; i < rhocount; i++) {
        for (auto iter = TripletVecVec[i].begin();
             iter != TripletVecVec[i].end(); ++iter) {
            SpT lto(*iter);
            Triplets.emplace_back(*iter);
        }
    }
    BCOffsetTensor.resize(rhocount * xcount * ycount,
                          rhocount * xcount * ycount * 16);
    BCOffsetTensor.setFromTriplets(Triplets.begin(), Triplets.end());
    // std::cout << "Number of RealT  products needed for BC calc: " <<
    // BCOffsetTensor.nonZeros() << " and rough memory usage is " <<
    // BCOffsetTensor.nonZeros() * (sizeof(RealT) + sizeof(long)) << std::endl;
}

template <int rhocount, int xcount, int ycount, class RealT>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::fastLTCalcOffset() {
    clearGrid(fastGALTResult);
    Eigen::Map<Eigen::Matrix<RealT, rhocount * xcount * ycount, 1>> source(
        gridValues.data.data());
    Eigen::Map<Eigen::Matrix<RealT, rhocount * xcount * ycount, 1>> target(
        fastGALTResult.data.data());
    target = LTOffsetTensor * source;
}

template <int rhocount, int xcount, int ycount, class RealT>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::fastBCCalcOffset() {
    clearGrid(BCResult);
    Eigen::Map<Eigen::Matrix<RealT, rhocount * xcount * ycount * 16, 1>> source(
        bicubicParameters.data.data());
    Eigen::Map<Eigen::Matrix<RealT, rhocount * xcount * ycount, 1>> target(
        BCResult.data.data());
    target = BCOffsetTensor * source;
}

template <int rhocount, int xcount, int ycount, class RealT>
template <typename TFunc1, typename TFunc2>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::GPUTestSuite(
    TFunc1 f, TFunc2 analytic) {
    // boost::timer::auto_cpu_timer t;
    SpM &cpu_sparse_matrix = LTOffsetTensor;
    viennacl::compressed_matrix<RealT> vcl_sparse_matrix(
        xcount * ycount * rhocount, xcount * ycount * rhocount);
    // t.start();

    viennacl::copy(cpu_sparse_matrix, vcl_sparse_matrix);

    viennacl::backend::finish();
    // t.report();
    std::cout
        << "That was the time to create the CPU matrix and copy it once to GPU."
        << std::endl;
    // t.start();

    viennacl::compressed_matrix<RealT, 1> vcl_compressed_matrix_1(
        xcount * ycount * rhocount, xcount * ycount * rhocount);
    viennacl::compressed_matrix<RealT, 4> vcl_compressed_matrix_4(
        xcount * ycount * rhocount, xcount * ycount * rhocount);
    viennacl::compressed_matrix<RealT, 8> vcl_compressed_matrix_8(
        xcount * ycount * rhocount, xcount * ycount * rhocount);
    // viennacl::coordinate_matrix<RealT> vcl_coordinate_matrix_128(xcount *
    // ycount * rhocount, xcount * ycount * rhocount);
    // viennacl::ell_matrix<RealT, 1> vcl_ell_matrix_1();
    // viennacl::hyb_matrix<RealT, 1> vcl_hyb_matrix_1();
    // viennacl::sliced_ell_matrix<RealT> vcl_sliced_ell_matrix_1(xcount *
    // ycount * rhocount, xcount * ycount * rhocount);

    viennacl::vector<RealT> gpu_source(gridValues.data.size());
    viennacl::vector<RealT> gpu_target(gridValues.data.size());
    copy(gridValues.data.begin(), gridValues.data.end(), gpu_source.begin());
    copy(gridValues.data.begin(), gridValues.data.end(), gpu_target.begin());
    // we are going to compute each product once and then sync, to compile all
    // kernels. this will feel like a ~1 second delay in user space.

    viennacl::copy(cpu_sparse_matrix, vcl_compressed_matrix_1);
    viennacl::copy(cpu_sparse_matrix, vcl_compressed_matrix_4);
    viennacl::copy(cpu_sparse_matrix, vcl_compressed_matrix_8);
    // viennacl::copy(cpu_sparse_matrix, vcl_coordinate_matrix_128);
    //    viennacl::copy(cpu_sparse_matrix,vcl_ell_matrix_1);
    // viennacl::copy(ublas_matrix,vcl_hyb_matrix_1);
    // viennacl::copy(cpu_sparse_matrix, vcl_sliced_ell_matrix_1);
    viennacl::backend::finish();
    // t.report();
    // t.start();
    std::cout << "That was the time to copy everything onto the GPU."
              << std::endl;

    fullgrid cpu_results[8];

    gpu_target = viennacl::linalg::prod(vcl_sparse_matrix, gpu_source);
    viennacl::backend::finish();

    gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_1, gpu_source);
    viennacl::copy(gpu_target.begin(), gpu_target.end(),
                   cpu_results[0].data.begin());

    gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_4, gpu_source);
    gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_8, gpu_source);
    // gpu_target = viennacl::linalg::prod(vcl_coordinate_matrix_128,
    // gpu_source); gpu_target =
    // viennacl::linalg::prod(vcl_ell_matrix_1,gpu_source); gpu_target =
    // viennacl::linalg::prod(vcl_hyb_matrix_1,gpu_source); gpu_target =
    // viennacl::linalg::prod(vcl_sliced_ell_matrix_1, gpu_source);

    viennacl::backend::finish();
    viennacl::copy(gpu_target.begin(), gpu_target.end(),
                   cpu_results[0].data.begin());
    viennacl::backend::finish();

    // t.report();
    std::cout << "That was the time to do all of the products, and copy the "
                 "result back twice."
              << std::endl;

    constexpr int gputimes = 1;
    // At this point everything has been done once.  We start benchmarking.  We
    // are going to include cost of vectors transfers back and forth.

    // t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(gridValues.data.begin(), gridValues.data.end(),
             gpu_source.begin());
        viennacl::backend::finish();
        gpu_target = viennacl::linalg::prod(vcl_sparse_matrix, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       cpu_results[0].data.begin());
        viennacl::backend::finish();
    }
    // t.report();
    std::cout << "That was the full cycle time to do " << gputimes
              << "  products using default sparse matrix." << std::endl;

    // t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(gridValues.data.begin(), gridValues.data.end(),
             gpu_source.begin());
        viennacl::backend::finish();
        gpu_target =
            viennacl::linalg::prod(vcl_compressed_matrix_1, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       cpu_results[1].data.begin());
        viennacl::backend::finish();
    }
    // t.report();
    std::cout << "That was the full cycle time to do " << gputimes
              << "  products using compressed_matrix_1 matrix." << std::endl;

    // t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(gridValues.data.begin(), gridValues.data.end(),
             gpu_source.begin());
        viennacl::backend::finish();
        gpu_target =
            viennacl::linalg::prod(vcl_compressed_matrix_4, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       cpu_results[2].data.begin());
        viennacl::backend::finish();
    }
    // t.report();
    std::cout << "That was the full cycle time to do " << gputimes
              << "  products using compressed_matrix_4 matrix." << std::endl;

    // t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(gridValues.data.begin(), gridValues.data.end(),
             gpu_source.begin());
        viennacl::backend::finish();
        gpu_target =
            viennacl::linalg::prod(vcl_compressed_matrix_8, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       cpu_results[3].data.begin());
        viennacl::backend::finish();
    }
    // t.report();
    std::cout << "That was the full cycle time to do " << gputimes
              << "  products using compressed_matrix_8 matrix." << std::endl;

    /* t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(gridValues.data.begin(), gridValues.data.end(),
    gpu_source.begin()); viennacl::backend::finish(); gpu_target =
    viennacl::linalg::prod(vcl_coordinate_matrix_128, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
    cpu_results[4].data.begin()); viennacl::backend::finish();
    }
    t.report();
*/
    // std::cout << "That was the full cycle time to do " << gputimes * 0 << "
    // products using coordinate_matrix_128 matrix." << std::endl;

    /*t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(gridValues.data.begin(), gridValues.data.end(),
    gpu_source.begin()); viennacl::backend::finish(); gpu_target =
    viennacl::linalg::prod(vcl_sliced_ell_matrix_1, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
    cpu_results[4].data.begin()); viennacl::backend::finish();
    }
    t.report();
    std::cout << "That was the full cycle time to do " << gputimes << " products
    using sliced_ell_matrix_1 matrix." << std::endl;
*/
    std::cout << "Next we report errors for each GPU calc (in above order) vs "
                 "CPU dot-product calc.  Here we only report maxabs norm"
              << std::endl;
    for (int i = 0; i < rhocount; i++) {
        std::cout.precision(5);
        std::cout << std::fixed << rhoset[i] << std::scientific << std::setw(15)
                  << maxNormDiff(fastGALTResult, cpu_results[0], i) << "\t"
                  << maxNormDiff(fastGALTResult, cpu_results[1], i) << "\t"
                  << maxNormDiff(fastGALTResult, cpu_results[2], i) << "\t"
                  << maxNormDiff(fastGALTResult, cpu_results[3], i)
                  << std::endl;
    }

    // end ViennaCL calc
}

template <int rhocount, int xcount, int ycount, class RealT>
template <typename TFunc1, typename TFunc2>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::GPUTestSuiteBC(
    TFunc1 f, TFunc2 analytic) {
    // TODO the below cheats and doesn't yet recompute derivs/params.  Need to
    // add that and benchmark.
    std::cout << "Beginning GPU test of BC calc.\n";
    // boost::timer::auto_cpu_timer t;
    viennacl::compressed_matrix<RealT> vcl_sparse_matrix(
        xcount * ycount * rhocount, xcount * ycount * rhocount * 16);
    // t.start();
    SpM &cpu_sparse_matrix = BCOffsetTensor;
    viennacl::copy(cpu_sparse_matrix, vcl_sparse_matrix);
    viennacl::backend::finish();
    // t.report();
    std::cout
        << "That was the time to create the CPU matrix and copy it once to GPU."
        << std::endl;
    // t.start();
    viennacl::backend::finish();

    viennacl::compressed_matrix<RealT, 1> vcl_compressed_matrix_1(
        xcount * ycount * rhocount, xcount * ycount * rhocount * 16);
    viennacl::compressed_matrix<RealT, 4> vcl_compressed_matrix_4(
        xcount * ycount * rhocount, xcount * ycount * rhocount * 16);
    viennacl::compressed_matrix<RealT, 8> vcl_compressed_matrix_8(
        xcount * ycount * rhocount, xcount * ycount * rhocount * 16);
    // viennacl::coordinate_matrix<RealT> vcl_coordinate_matrix_128(xcount *
    // ycount * rhocount, xcount * ycount * rhocount * 16);
    // viennacl::ell_matrix<RealT, 1> vcl_ell_matrix_1();
    // viennacl::hyb_matrix<RealT, 1> vcl_hyb_matrix_1();
    // viennacl::sliced_ell_matrix<RealT> vcl_sliced_ell_matrix_1(xcount *
    // ycount * rhocount, xcount * ycount * rhocount *16);
    viennacl::vector<RealT> gpu_source(bicubicParameters.data.size());
    viennacl::vector<RealT> gpu_target(BCResult.data.size());
    copy(cpu_sparse_matrix,
         vcl_sparse_matrix);  // default alignment, benchmark different options.
    copy(bicubicParameters.data.begin(), bicubicParameters.data.end(),
         gpu_source.begin());
    viennacl::backend::finish();
    copy(gridValues.data.begin(), gridValues.data.end(),
         gpu_target.begin());  // this is garbage data, I just want to make sure
                               // it's allocated.
    // we are going to compute each product once and then sync, to compile all
    // kernels. this will feel like a ~1 second delay in user space.
    viennacl::copy(cpu_sparse_matrix, vcl_compressed_matrix_1);
    viennacl::copy(cpu_sparse_matrix, vcl_compressed_matrix_4);
    viennacl::copy(cpu_sparse_matrix, vcl_compressed_matrix_8);
    // viennacl::copy(cpu_sparse_matrix, vcl_coordinate_matrix_128);
    //    viennacl::copy(cpu_sparse_matrix,vcl_ell_matrix_1);
    // viennacl::copy(ublas_matrix,vcl_hyb_matrix_1);
    // viennacl::copy(cpu_sparse_matrix, vcl_sliced_ell_matrix_1);
    viennacl::backend::finish();
    // t.report();
    // t.start();
    std::cout << "That was the time to copy everything onto the GPU."
              << std::endl;

    fullgrid cpu_results[8];

    gpu_target = viennacl::linalg::prod(vcl_sparse_matrix, gpu_source);
    viennacl::backend::finish();
    gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_1, gpu_source);
    viennacl::copy(gpu_target.begin(), gpu_target.end(),
                   cpu_results[0].data.begin());
    gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_4, gpu_source);
    gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_8, gpu_source);
    // gpu_target = viennacl::linalg::prod(vcl_coordinate_matrix_128,
    // gpu_source); gpu_target =
    // viennacl::linalg::prod(vcl_ell_matrix_1,gpu_source); gpu_target =
    // viennacl::linalg::prod(vcl_hyb_matrix_1,gpu_source); gpu_target =
    // viennacl::linalg::prod(vcl_sliced_ell_matrix_1, gpu_source);
    viennacl::backend::finish();
    viennacl::copy(gpu_target.begin(), gpu_target.end(),
                   cpu_results[0].data.begin());
    viennacl::backend::finish();
    // t.report();
    std::cout << "That was the time to do all of the products, and copy the "
                 "result back."
              << std::endl;

    constexpr int gputimes = 1;
    // At this point everything has been done once.  We start benchmarking.  We
    // are going to include cost of vectors transfers back and forth.

    // t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(bicubicParameters.data.begin(), bicubicParameters.data.end(),
             gpu_source.begin());
        viennacl::backend::finish();
        gpu_target = viennacl::linalg::prod(vcl_sparse_matrix, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       cpu_results[0].data.begin());
        viennacl::backend::finish();
    }
    // t.report();
    std::cout << "That was the full cycle time to do " << gputimes
              << "  products using default sparse matrix." << std::endl;

    // t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(bicubicParameters.data.begin(), bicubicParameters.data.end(),
             gpu_source.begin());
        viennacl::backend::finish();
        gpu_target =
            viennacl::linalg::prod(vcl_compressed_matrix_1, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       cpu_results[1].data.begin());
        viennacl::backend::finish();
    }
    // t.report();
    std::cout << "That was the full cycle time to do " << gputimes
              << "  products using compressed_matrix_1 matrix." << std::endl;

    // t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(bicubicParameters.data.begin(), bicubicParameters.data.end(),
             gpu_source.begin());
        viennacl::backend::finish();
        gpu_target =
            viennacl::linalg::prod(vcl_compressed_matrix_4, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       cpu_results[2].data.begin());
        viennacl::backend::finish();
    }
    // t.report();
    std::cout << "That was the full cycle time to do " << gputimes
              << "  products using compressed_matrix_4 matrix." << std::endl;

    // t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(bicubicParameters.data.begin(), bicubicParameters.data.end(),
             gpu_source.begin());
        viennacl::backend::finish();
        gpu_target =
            viennacl::linalg::prod(vcl_compressed_matrix_8, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       cpu_results[3].data.begin());
        viennacl::backend::finish();
    }
    //  t.report();
    std::cout << "That was the full cycle time to do " << gputimes
              << "  products using compressed_matrix_8 matrix." << std::endl;

    // t.start();
    /*    for(int count =0; count<gputimes;++count){
      copy(bicubicParameters.data.begin(),bicubicParameters.data.end(),gpu_source.begin());
      viennacl::backend::finish();
      gpu_target = viennacl::linalg::prod(vcl_coordinate_matrix_128,gpu_source);
      viennacl::backend::finish();
      viennacl::copy(gpu_target.begin(),gpu_target.end(),cpu_results[4].data.begin());
      viennacl::backend::finish();
    }
    t.report();*/
    // std::cout << "That was the full cycle time to do " << gputimes * 0 << "
    // products using coordinate_matrix_128 matrix." << std::endl;

    // t.start();
    /*for (int count = 0; count < gputimes; ++count) {
        copy(bicubicParameters.data.begin(), bicubicParameters.data.end(),
       gpu_source.begin()); viennacl::backend::finish(); gpu_target =
       viennacl::linalg::prod(vcl_sliced_ell_matrix_1, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
       cpu_results[4].data.begin()); viennacl::backend::finish();
        }*/
    // t.report();
    // std::cout << "That was the full cycle time to do " << gputimes << "
    // products using sliced_ell_matrix_1 matrix." << std::endl;

    // t.start();
    for (int count = 0; count < gputimes; ++count) {
        setupDerivsGrid();
        setupBicubicGrid();
        copy(bicubicParameters.data.begin(), bicubicParameters.data.end(),
             gpu_source.begin());
        viennacl::backend::finish();
        gpu_target = viennacl::linalg::prod(vcl_sparse_matrix, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       cpu_results[0].data.begin());
        viennacl::backend::finish();
    }
    // t.report();
    std::cout << "That was the full cycle time to do " << gputimes
              << "  products using default sparse matrix, and recalculated "
                 "derivatives and BC parameters."
              << std::endl;

    std::cout << "Next we report errors for each GPU calc (in above order) vs "
                 "CPU dot-product calc.  Here we only report maxabs norm"
              << std::endl;
    for (int i = 0; i < rhocount; i++) {
        std::cout.precision(5);
        std::cout << std::fixed << rhoset[i] << std::scientific << std::setw(15)
                  << maxNormDiff(BCResult, cpu_results[0], i) << "\t"
                  << maxNormDiff(BCResult, cpu_results[1], i) << "\t"
                  << maxNormDiff(BCResult, cpu_results[2], i) << "\t"
                  << maxNormDiff(BCResult, cpu_results[3], i) << std::endl;
    }
}

/* Run test suite.  We expect the test suite to:
1)     Calculate the gyroaverage transform of f, using f on a full grid, at each
grid point 2)     above, using f but truncating it to be 0 outside of our grid
3)     above, using bilinear interp of f (explicitly only gridpoint valuations
are allowed) and we trapezoid rule the interp-ed, truncated function  (vs (2),
only interp error introduced) 4)     fast dot-product calc: see
AssembleFastGACalc for details We report evolution of errorand sample timings.
   */

template <int rhocount, int xcount, int ycount, class RealT>
template <typename TFunc1, typename TFunc2>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::GyroAveragingTestSuite(
    TFunc1 f, TFunc2 analytic) {
    // boost::timer::auto_cpu_timer t;
    fill(gridValues,
         f);  // This is the base grid of values we will interpolate.
              //  t.start();
    fill(analytic_averages, analytic);  // analytic formula for gyroaverages
                                        // t.report();
    std::cout
        << "That was the time required to calculate analytic gyroaverages.\n";
    setupInterpGrid();
    //  t.start();
    fillAlmostExactGA(almostExactGA, f);
    //  t.report();
    std::cout << "That was the time required to calculate gyroaverages from "
                 "the definition, with the trapezoid rule.\n";
    //   t.start();
    fillTruncatedAlmostExactGA(truncatedAlmostExactGA, f);
    //   t.report();
    std::cout << "That was the time required to calculate gryoaverages by def "
                 "(as above), except we hard truncated f() to 0 off-grid.\n";
    //  t.start();
    fillTrapezoidInterp(trapezoidInterp, f);
    //  t.report();
    std::cout << "That was the time required to calc gyroaverages by def, "
                 "replacing f() by its bilinear interpolant."
              << std::endl;
    //  t.start();
    assembleFastGACalc();
    //  t.report();
    std::cout << "That was the time required to assemble the sparse matrix in "
                 "the fast-GA dot product calculation."
              << std::endl;
    //  t.start();
    setupDerivsGrid();
    //  t.report();
    //  t.start();
    setupBicubicGrid();
    //  t.report();
    //  t.start();
    assembleFastBCCalc();
    //  t.report();
    std::cout << "That was the time required to assemble the sparse matrix in "
                 "the fast-BC dot product calculation."
              << std::endl;

    //  t.start();
    int times = 1;
    for (int counter = 0; counter < times; counter++) {
        fastLTCalcOffset();
    }

    // t.report();
    std::cout << "The was the time require to run LT gyroaverage calc " << times
              << " times. \n " << std::endl;

    // t.start();
    for (int counter = 0; counter < times; counter++) {
        setupDerivsGrid();
        setupBicubicGrid();
        fastBCCalcOffset();
    }

    // t.report();
    std::cout << "The was the time require to run BC gyroaverage calc " << times
              << " times. \n " << std::endl;

    GPUTestSuite(f, analytic);
    setupDerivsGrid();
    setupBicubicGrid();
    fullgrid bicubicResults;
    fillBicubicInterp(bicubicResults);
    GPUTestSuiteBC(f, analytic);

    std::cout << "Below are some summary statistics for various grid.  Under "
                 "each header is a pair of values.  The first is the RMS of a "
                 "matrix, the second is the max absolute entry in a matrix.\n";

    std::cout
        << "rho        Input Grid                   Analytic Estimates         "
           "     Trapezoid Rule                  Trap Rule truncation          "
           "  Trapezoid rule of bilin interp  Fast dot-product GA\n";

    for (int i = 0; i < rhocount; i++) {
        std::cout.precision(5);
        std::cout << std::fixed << rhoset[i] << std::scientific << std::setw(15)
                  << RMSNorm(gridValues, i) << "\t" << maxNorm(gridValues, i)
                  << "\t" << RMSNorm(analytic_averages, i) << "\t"
                  << maxNorm(analytic_averages, i) << "\t"
                  << RMSNorm(almostExactGA, i) << "\t"
                  << maxNorm(almostExactGA, i) << "\t"
                  << RMSNorm(truncatedAlmostExactGA, i) << "\t"
                  << maxNorm(truncatedAlmostExactGA, i) << "\t"
                  << RMSNorm(trapezoidInterp, i) << "\t"
                  << maxNorm(trapezoidInterp, i) << "\t"
                  << RMSNorm(fastGALTResult, i) << "\t"
                  << maxNorm(fastGALTResult, i) << "\t"
                  << RMSNorm(bicubicResults, i) << "\t"
                  << maxNorm(bicubicResults, i) << "\t" << RMSNorm(BCResult, i)
                  << "\t" << maxNorm(BCResult, i) << "\n";
    }
    std::cout << "Diffs:\n";
    std::cout << "rho        Analytic vs Quadrature       Error due to "
                 "truncation         Error due to interp             Analytic "
                 "vs interp              interp vs DP        bicub trap vs "
                 "analytic          bicubic DP vs analytic \n";
    for (int i = 0; i < rhocount; i++) {
        std::cout.precision(5);
        std::cout << std::fixed << rhoset[i] << std::scientific << std::setw(15)
                  << RMSNormDiff(analytic_averages, almostExactGA, i) << "\t"
                  << maxNormDiff(analytic_averages, almostExactGA, i) << "\t"
                  << RMSNormDiff(almostExactGA, truncatedAlmostExactGA, i)
                  << "\t"
                  << maxNormDiff(almostExactGA, truncatedAlmostExactGA, i)
                  << "\t"
                  << RMSNormDiff(truncatedAlmostExactGA, trapezoidInterp, i)
                  << "\t"
                  << maxNormDiff(truncatedAlmostExactGA, trapezoidInterp, i)
                  << "\t" << RMSNormDiff(analytic_averages, trapezoidInterp, i)
                  << "\t" << maxNormDiff(analytic_averages, trapezoidInterp, i)
                  << "\t" << RMSNormDiff(trapezoidInterp, fastGALTResult, i)
                  << "\t" << maxNormDiff(trapezoidInterp, fastGALTResult, i)
                  << "\t" << RMSNormDiff(bicubicResults, analytic_averages, i)
                  << "\t" << maxNormDiff(bicubicResults, analytic_averages, i)
                  << "\t" << RMSNormDiff(BCResult, analytic_averages, i) << "\t"
                  << maxNormDiff(BCResult, analytic_averages, i) << "\n";
        //     << RMSNormDiff(fastGALTResult, cpu_results[0], i) << "\t"
        //    << maxNormDiff(fastGALTResult, cpu_results[0], i) << "\n";

        //<< RMSNormDiff(fastGALTResult, fastGACalcResultOffset, i) << "\t"
        //<< maxNormDiff(fastGALTResult, fastGACalcResultOffset, i) << "\n";
    }
};

template <int rhocount, int xcount, int ycount, class RealT>
template <typename TFunc1, typename TFunc2>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::compactErrorAnalysis(
    TFunc1 f, TFunc2 analytic) {
    fill(gridValues, f);  // This is the base grid of values we will
                          // interpolate.
    fill(analytic_averages, analytic);  // analytic formula for gyroaverages
    setupInterpGrid();
    fillAlmostExactGA(almostExactGA, f);
    fillTruncatedAlmostExactGA(truncatedAlmostExactGA, f);
    fillTrapezoidInterp(trapezoidInterp, f);
    assembleFastGACalc();
    setupDerivsGrid();
    setupBicubicGrid();
    assembleFastBCCalc();
    fastLTCalcOffset();
    fastBCCalcOffset();
    fullgrid bicubicResults;
    fillBicubicInterp(bicubicResults);

    for (int i = 0; i < rhocount; i++) {
        std::cout.precision(5);
        std::cout << std::fixed << xcount << std::scientific << std::setw(15)
                  << RMSNorm(gridValues, i) << "\t" << maxNorm(gridValues, i)
                  << "\t" << RMSNorm(analytic_averages, i) << "\t"
                  << maxNorm(analytic_averages, i) << "\t"
                  << RMSNorm(fastGALTResult, i) << "\t"
                  << maxNorm(fastGALTResult, i) << "\t" << RMSNorm(BCResult, i)
                  << "\t" << maxNorm(BCResult, i) << "\t"
                  << maxNormDiff(fastGALTResult, analytic_averages, i) /
                         maxNorm(analytic_averages, i)
                  << "\t"
                  << maxNormDiff(BCResult, analytic_averages, i) /
                         maxNorm(analytic_averages, i)
                  << "\n";
    }
}

// below function returns the indices referring to lower left point of the grid
// box containing (x,y) it is not (yet) efficient.  In particular, we should
// probably explicitly assume equispaced grids and use that fact.
template <int rhocount, int xcount, int ycount, class RealT>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::interpIndexSearch(
    const RealT x, const RealT y, int &xindex, int &yindex) {
    if ((x < xset[0]) || (y < yset[0]) || (x > xset.back()) ||
        (y > yset.back())) {
        xindex = xcount - 1;  // the top right corner should have zeros.
        yindex = ycount - 1;
        return;
    }
    xindex = std::upper_bound(xset.begin(), xset.end(), x) - xset.begin() - 1;
    yindex = std::upper_bound(yset.begin(), yset.end(), y) - yset.begin() - 1;
    xindex = std::min(std::max(xindex, 0), xcount - 2);
    yindex = std::min(ycount - 2, std::max(yindex, 0));
    assert((xset[xindex]) <= x && (xset[xindex + 1] >= x));
    assert((yset[yindex]) <= y && (yset[yindex + 1] >= y));
}

template <int rhocount, int xcount, int ycount, class RealT>
RealT GyroAveragingGrid<rhocount, xcount, ycount, RealT>::interp2d(
    int rhoindex, const RealT x, const RealT y) {
    assert((rhoindex >= 0) && (rhoindex < rhocount));
    if ((x <= xset[0]) || (y <= yset[0]) || (x >= xset.back()) ||
        (y >= yset.back()))
        return 0;
    int xindex = 0, yindex = 0;
    interpIndexSearch(x, y, xindex, yindex);
    RealT result = BilinearInterpolation<RealT>(
        gridValues(rhoindex, xindex, yindex),
        gridValues(rhoindex, xindex + 1, yindex),
        gridValues(rhoindex, xindex, yindex + 1),
        gridValues(rhoindex, xindex + 1, yindex + 1), xset[xindex],
        xset[xindex + 1], yset[yindex], yset[yindex + 1], x, y);

    return result;
}

// Below returns bicubic interp f(x,y), in the dumb way.  We should do
// multivariate horners method soon.
template <int rhocount, int xcount, int ycount, class RealT>
RealT GyroAveragingGrid<rhocount, xcount, ycount, RealT>::interpNaiveBicubic(
    int rhoindex, const RealT x, const RealT y) {
    assert((rhoindex >= 0) && (rhoindex < rhocount));
    if ((x <= xset[0]) || (y <= yset[0]) || (x >= xset.back()) ||
        (y >= yset.back()))
        return 0;
    int xindex = 0, yindex = 0;
    RealT result = 0;
    interpIndexSearch(x, y, xindex, yindex);
    RealT xns[4] = {1, x, x * x, x * x * x};
    RealT yns[4] = {1, y, y * y, y * y * y};
    for (int i = 0; i <= 3; ++i)
        for (int j = 0; j <= 3; ++j) {
            result += bicubicParameters(rhoindex, xindex, yindex, j * 4 + i) *
                      xns[i] * yns[j];
        }
    return result;
}

template <int rhocount, int xcount, int ycount, class RealT>
template <typename TFunc1, typename TFunc2, typename TFunc3, typename TFunc4>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::derivsErrorAnalysis(
    TFunc1 f, TFunc2 f_x, TFunc3 f_y, TFunc4 f_xy) {
    std::cout << "Starting derivative accuracy check:\n";
    fill(gridValues, f);
    setupInterpGrid();
    setupDerivsGrid();
    fullgrid analytic[4];
    fullgrid numeric[4];
    for (int i = 0; i < rhocount; i++)
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++) {
                analytic[0](i, j, k) = f(rhoset[i], xset[j], yset[k]);
                numeric[0](i, j, k) = derivs(i, j, k, 0);
                analytic[1](i, j, k) = f_x(rhoset[i], xset[j], yset[k]);
                numeric[1](i, j, k) = derivs(i, j, k, 1);
                analytic[2](i, j, k) = f_y(rhoset[i], xset[j], yset[k]);
                numeric[2](i, j, k) = derivs(i, j, k, 2);
                analytic[3](i, j, k) = f_xy(rhoset[i], xset[j], yset[k]);
                numeric[3](i, j, k) = derivs(i, j, k, 3);
            }

    for (int l = 0; l < 4; l++) {
        csvPrinter(analytic[l], 1);
        std::cout << std::endl;
        csvPrinter(numeric[l], 1);
        std::cout << std::endl;
    }

    for (int i = 0; i < rhocount; i++) {
        std::cout.precision(5);
        std::cout << std::fixed << rhoset[i] << std::scientific << std::setw(15)
                  << RMSNormDiff(analytic[0], numeric[0], i) << "\t"
                  << maxNormDiff(analytic[0], numeric[0], i) << "\t"
                  << RMSNormDiff(analytic[1], numeric[1], i) << "\t"
                  << maxNormDiff(analytic[1], numeric[1], i) << "\t"
                  << RMSNormDiff(analytic[2], numeric[2], i) << "\t"
                  << maxNormDiff(analytic[2], numeric[2], i) << "\n";
    }
}

template <int rhocount, int xcount, int ycount, class RealT>
template <typename TFunc1, typename TFunc2>
void GyroAveragingGrid<rhocount, xcount, ycount, RealT>::InterpErrorAnalysis(
    TFunc1 f, TFunc2 analytic) {
    fill(gridValues, f);  // This is the base grid of values we will
                          // interpolate.
    fill(analytic_averages, analytic);  // analytic formula for gyroaverages
    fullgrid bicubictest;

    setupInterpGrid();
    setupDerivsGrid();
    setupBicubicGrid();
    for (auto i = 0; i < rhocount; i++)
        for (auto j = 0; j < xcount; j++)
            for (auto k = 0; k < ycount; k++) {
                bicubictest(i, j, k) = interpNaiveBicubic(i, xset[j], yset[k]);
            }
    fillAlmostExactGA(almostExactGA, f);
    fillTruncatedAlmostExactGA(truncatedAlmostExactGA, f);
    fillTrapezoidInterp(trapezoidInterp, f);
    fillBicubicInterp(bicubicInterp);
    std::cout << "There were " << trapezoidInterp.data.size() << " entries. \n";
    std::cout << "Diffs:\n";
    std::cout
        << "rho        Analytic vs Quadrature       Error due to truncation    "
           "     Error due to interp             Analytic vs interp            "
           "  interp vs dot-product GA          GPU vs CPU\n";

    for (int i = 0; i < rhocount; i++) {
        std::cout.precision(5);
        std::cout << std::fixed << rhoset[i] << std::scientific << std::setw(15)
                  << RMSNormDiff(analytic_averages, almostExactGA, i) << "\t"
                  << maxNormDiff(analytic_averages, almostExactGA, i) << "\t"
                  << RMSNormDiff(almostExactGA, truncatedAlmostExactGA, i)
                  << "\t"
                  << maxNormDiff(almostExactGA, truncatedAlmostExactGA, i)
                  << "\t"
                  << RMSNormDiff(truncatedAlmostExactGA, trapezoidInterp, i)
                  << "\t"
                  << maxNormDiff(truncatedAlmostExactGA, trapezoidInterp, i)
                  << "\t" << RMSNormDiff(analytic_averages, trapezoidInterp, i)
                  << "\t" << maxNormDiff(analytic_averages, trapezoidInterp, i)
                  << "\t"
                  << RMSNormDiff(truncatedAlmostExactGA, bicubicInterp, i)
                  << "\t"
                  << maxNormDiff(truncatedAlmostExactGA, bicubicInterp, i)
                  << "\n";
    }
}
// below function is for testing and will be refactored later.

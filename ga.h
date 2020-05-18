// file foobar.h:
#ifndef GYROAVERAGING_GA_H
#define GYROAVERAGING_GA_H
// ... declarations ...
#endif // GYROAVERAGING_GA_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <vector>


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
    Array3d() : data(w * h * d, 0) {
    }

    inline T &at(int x, int y, int z) {
        return data[x * h * d + y * d + z];
    }

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
    Array4d() : data(w * h * d * l, 0) {
    }

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
inline RealT FixedNTrapezoidIntegrate(RealT x, RealT y, TFunc f, int n) { //used to be trapezoid rule.
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

inline double
BilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y);

template <class RealT = double>
std::vector<RealT> LinearSpacedArray(RealT a, RealT b, int N) {
    RealT h = (b - a) / static_cast<RealT>(N - 1);
    std::vector<RealT> xs(N);
    auto x=xs.begin();
    RealT val;
    for (val = a; x != xs.end(); ++x, val += h) {
        *x = val;
    }
    return xs;
}

//change this to realT
template <class RealT = double>
struct sparseOffset { //no constructor.
    int target, source;
    RealT coeffs[4];
};

template <class RealT = double>
struct LTOffset {
    int target, source;
    RealT coeff;
};

std::array<double, 4>
operator+(const std::array<double, 4> &l, const std::array<double, 4> &r) {
    std::array<double, 4> ret;
    ret[0] = l[0] + r[0];
    ret[1] = l[1] + r[1];
    ret[2] = l[2] + r[2];
    ret[3] = l[3] + r[3];
    return ret;
}


std::array<float, 4>
operator+(const std::array<float, 4> &l, const std::array<float, 4> &r) {
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
    typedef Array4d<rhocount, xcount, ycount, 4, RealT> derivsGrid; //at each rho,x,y calculate [f,f_x,f_y,f_xy]
    typedef Array4d<rhocount, xcount, ycount, 16, RealT> bicubicParameterGrid;
    typedef Eigen::SparseMatrix<RealT, Eigen::RowMajor> SpM;
    typedef Eigen::Triplet<RealT> SpT;

private:
    std::vector<RealT> rhoset;
    std::vector<RealT> xset;
    std::vector<RealT> yset;
    fullgrid gridValues;             //input values of f
    fullgrid almostExactGA;          //stores value of GA calculated as trapezoid rule on input f
    fullgrid truncatedAlmostExactGA; //above, except f hard truncated to 0 outside grid
    fullgrid trapezoidInterp;        //GA calculated as trapezoid rule on interpolated, truncated f
    fullgrid bicubicInterp;
    fullgrid fastGALTResult;
    fullgrid BCResult;
    fullgrid analytic_averages;      // stores value of expected GA computed analytically
    fullgridInterp interpParameters; //will store the bilinear interp parameters.
    bicubicParameterGrid bicubicParameters;
    SpM LTOffsetTensor;
    SpM BCOffsetTensor;
    SpM FDTensor; //finite difference tensor, used to populate derivs from gridValues
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

    //below fills a grid, given a function of rho, x, and y
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
    //below fills a grid, given a function of i,j,k
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
        fillbyindex(m, [&](int i, int j, int k) -> RealT { //adaptive trapezoid rule on actual input function.
            RealT xc = xset[j];
            RealT yc = yset[k];
            if (rhoset[i] == 0)
                return f(i, xc, yc);
            auto new_f = [&](RealT x) -> RealT { return f(rhoset[i], xc + rhoset[i] * std::sin(x), yc - rhoset[i] * std::cos(x)); };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
    }

    template <typename TFunc1>
    void fillTruncatedAlmostExactGA(fullgrid &m, TFunc1 f) {
        fillbyindex(m, [&](int i, int j, int k) -> RealT {
            RealT xc = xset[j];
            RealT yc = yset[k];
            if (rhoset[i] == 0)
                return f(i, xc, yc);
            auto new_f = [&](RealT x) -> RealT {
                RealT ex = xc + rhoset[i] * std::sin(x);
                RealT why = yc - rhoset[i] * std::cos(x);
                if ((ex < xset[0]) || (ex > xset.back()))
                    return 0;
                if ((why < yset[0]) || (why > yset.back()))
                    return 0;
                return f(rhoset[i], ex, why);
            };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
    }

    template <typename TFunc1>
    void fillTrapezoidInterp(fullgrid &m, TFunc1 f) { //this calls interp2d. gridValues must be filled, but we don't need setupInterpGrid.
        fillbyindex(m, [&](int i, int j, int k) -> RealT {
            RealT xc = xset[j];
            RealT yc = yset[k];
            if (rhoset[i] == 0)
                return f(rhoset[i], xc, yc);
            auto new_f = [&](RealT x) -> RealT { return interp2d(i, xc + rhoset[i] * std::sin(x), yc - rhoset[i] * std::cos(x)); };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
    }
    //below requires the bicubic parameter grid to be populated.
    void fillBicubicInterp(fullgrid &m) {
        fillbyindex(m, [&](int i, int j, int k) -> RealT {
            RealT xc = xset[j];
            RealT yc = yset[k];
            if (rhoset[i] == 0)
                return gridValues(i, j, k);

            auto new_f = [&](RealT x) -> RealT { return interpNaiveBicubic(i, xc + rhoset[i] * std::sin(x), yc - rhoset[i] * std::cos(x)); };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
    }

public:
    GyroAveragingGrid(const std::vector<RealT> &rhos,
                      const std::vector<RealT> &xes,
                      const std::vector<RealT> &yies) : rhoset(rhos), xset(xes), yset(yies) {
        assert(rhocount == rhos.size());
        assert(xcount == xes.size());
        assert(ycount == yies.size());

        std::sort(xset.begin(), xset.end());
        std::sort(yset.begin(), yset.end());
        std::sort(rhoset.begin(), rhoset.end());
    }
    void interpIndexSearch(const RealT x, const RealT y, int &xindex, int &yindex);
    inline void integrand(const RealT rho, const RealT xc, const RealT yc, const RealT gamma, RealT &xn, RealT &yn) {
        xn = xc + rho * std::sin(gamma);
        yn = yc - rho * std::cos(gamma);
    }
    void setupInterpGrid();
    void setupDerivsGrid(); //make accuracy a variable later
    void setupBicubicGrid();
    //void setupBicubicGrid2();
    void assembleFastGACalc(void);

    void assembleFastBCCalc(void);
    void fastLTCalcOffset();
    void fastBCCalcOffset();
    std::array<RealT, 4> arcIntegral(RealT rho, RealT xc, RealT yc, RealT s0, RealT s1);
    //std::array<RealT, 16> arcIntegralBicubic(RealT rho, RealT xc, RealT yc, RealT s0, RealT s1);
    template <typename TFunc1, typename TFunc2>
    void GyroAveragingTestSuite(TFunc1 f,
                                TFunc2 analytic);

    
    template <typename TFunc1, typename TFunc2>
    void compactErrorAnalysis(TFunc1 f,
                              TFunc2 analytic);

    template <typename TFunc1, typename TFunc2>
    void GPUTestSuite(TFunc1 f,
                      TFunc2 analytic);

    template <typename TFunc1, typename TFunc2>
    void GPUTestSuiteBC(TFunc1 f,
                        TFunc2 analytic);
    template <typename TFunc1, typename TFunc2>
    void InterpErrorAnalysis(TFunc1 f,
                             TFunc2 analytic);
    template <typename TFunc1, typename TFunc2>
    void errorAnalysis(TFunc1 f,
                       TFunc2 analytic);

    template <typename TFunc1, typename TFunc2, typename TFunc3, typename TFunc4>
    void derivsErrorAnalysis(TFunc1 f,
                             TFunc2 f_x, TFunc3 f_y, TFunc4 f_xy);

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
    RealT s = 0; // from 0 to 2*Pi only please.
    indexedPoint(RealT x = 0, RealT y = 0, RealT row = 0) : xvalue(x), yvalue(y), s(row) {}
};

template <typename TFunc1, typename TFunc2, class RealT = double>
void interpAnalysis(const gridDomain<RealT> &g, TFunc1 f,
                    TFunc2 analytic);

template <typename TFunc1, typename TFunc2, class RealT = double>
void errorAnalysis(const gridDomain<RealT> &g, TFunc1 f,
                   TFunc2 analytic);

template <int i, typename TFunc1, typename TFunc2, class RealT = double>
void interpAnalysisInnerLoop(const gridDomain<RealT> &g, TFunc1 f,
                             TFunc2 analytic);

template <int i, typename TFunc1, typename TFunc2, class RealT = double>
void errorAnalysisInnerLoop(const gridDomain<RealT> &g, TFunc1 f,
                            TFunc2 analytic);

template <typename TFunc1, typename TFunc2, typename TFunc3, typename TFunc4, class RealT = double>
void derivTest(const gridDomain<RealT>  &g, TFunc1 f,
               TFunc2 f_x, TFunc3 f_y, TFunc4 f_xy);

template <class RealT = double>
void inline arcIntegralBicubic(std::array<RealT, 16> &coeffs,
                               RealT rho, RealT xc, RealT yc, RealT s0, RealT s1);
void testArcIntegralBicubic();

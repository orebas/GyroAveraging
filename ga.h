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

template <typename TFunc>
inline double FixedNTrapezoidIntegrate(double x, double y, TFunc f, int n) { //used to be trapezoid rule.

    double h = (y - x) / (n - 1);
    double sum = 0;
    double fx1 = f(x);
    for (int i = 1; i <= n; i++) {
        double x2 = x + i * h;
        double fx2 = f(x2);
        sum += (fx1 + fx2) * h;
        fx1 = fx2;
    }
    return 0.5 * sum;
}

inline double
BilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y);
std::vector<double> LinearSpacedArray(double a, double b, int N);

struct sparseEntry {
    std::array<int, 3> target; //target entry i,j,k
    std::array<double, 4> coeffs;
    std::array<int, 3> source;
    sparseEntry(std::array<int, 3> t, std::array<double, 4> c, std::array<int, 3> s) : target(t), coeffs(c), source(s) {}
};

struct sparseOffset { //no constructor.
    int target, source;
    double coeffs[4];
};

std::array<double, 4> operator+(const std::array<double, 4> &l, const std::array<double, 4> &r) {
    std::array<double, 4> ret;
    ret[0] = l[0] + r[0];
    ret[1] = l[1] + r[1];
    ret[2] = l[2] + r[2];
    ret[3] = l[3] + r[3];
    return ret;
}

template <int rhocount, int xcount, int ycount>
class GyroAveragingGrid {
public:
    typedef Array3d<rhocount, xcount, ycount, double> fullgrid;
    typedef Array4d<rhocount, xcount, ycount, 4, double> fullgridInterp;

private:
    std::vector<double> rhoset;
    std::vector<double> xset;
    std::vector<double> yset;
    fullgrid gridValues;             //input values of f
    fullgrid almostExactGA;          //stores value of GA calculated as trapezoid rule on input f
    fullgrid truncatedAlmostExactGA; //above, except f hard truncated to 0 outside grid
    fullgrid trapezoidInterp;        //GA calculated as trapezoid rule on interpolated, truncated f
    fullgrid fastGACalcResult;
    fullgrid fastGACalcResultOffset;
    fullgrid analytic_averages; // stores value of expected GA computed analytically
    fullgrid exactF;
    fullgridInterp interpParameters; //will store the bilinear interp parameters.
    std::vector<sparseEntry> GATensor;
    std::vector<sparseOffset> GAOffsetTensor;

    void csvPrinter(const fullgrid &m, int rho) {
        for (int j = 0; j < xcount; j++) {
            for (int k = 0; k < ycount; k++) {
                std::cout << m(rho, j, k) << ",";
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
    double FrobNorm(const fullgrid &m, int rho) {
        double result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++)
                result += m(rho, j, k) * m(rho, j, k);
        return std::sqrt(result);
    }
    double maxNorm(const fullgrid &m, int rho) {
        double result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++)
                result = std::max(result, std::abs(m(rho, j, k)));
        return result;
    }

    double FrobNormDiff(const fullgrid &m1, const fullgrid &m2, int rho) {
        double result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++) {
                double t = m1(rho, j, k) - m2(rho, j, k);
                result += t * t;
            }
        return std::sqrt(result);
    }
    double maxNormDiff(const fullgrid &m1, const fullgrid &m2, int rho) {
        double result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++) {
                double t = m1(rho, j, k) - m2(rho, j, k);
                result = std::max(result, std::abs(t));
            }
        return result;
    }

    //below fills a grid, given a function of rho, x, and y
    template <typename TFunc>
    void fill(fullgrid &m, TFunc f) {
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
        for (int i = 0; i < rhocount; i++)
            for (int j = 0; j < xcount; j++)
                for (int k = 0; k < ycount; k++) {
                    m(i, j, k) = f(i, j, k);
                }
    }

public:
    GyroAveragingGrid(const std::vector<double> &rhos,
                      const std::vector<double> &xes,
                      const std::vector<double> &yies) : rhoset(rhos), xset(xes), yset(yies) {
        assert(rhocount == rhos.size());
        assert(xcount == xes.size());
        assert(ycount == yies.size());

        std::sort(xset.begin(), xset.end());
        std::sort(yset.begin(), yset.end());
        std::sort(rhoset.begin(), rhoset.end());
    }
    void interpIndexSearch(const int rhoindex, const double x, const double y, int &xindex, int &yindex);
    inline void integrand(const double rho, const double xc, const double yc, const double gamma, double &xn, double &yn) {
        xn = xc + rho * std::sin(gamma);
        yn = yc - rho * std::cos(gamma);
    }
    void setupInterpGrid();
    void assembleFastGACalc(void);
    void fastGACalc();
    void fastGACalcOffset();
    std::array<double, 4> arcIntegral(double rho, double xc, double yc, double s0, double s1);
    template <typename TFunc1, typename TFunc2>
    void GyroAveragingTestSuite(TFunc1 f,
                                TFunc2 analytic);

    double interp2d(int rhoindex, const double x, const double y);
};

struct indexedPoint {
    double xvalue = 0;
    double yvalue = 0;
    double s = 0; // from 0 to 2*Pi only please.
    indexedPoint(double x = 0, double y = 0, double row = 0) : xvalue(x), yvalue(y), s(row) {}
};

// GyroDebugging.cpp : This file contains the 'main' function. Program execution begins and ends there.
////#include "pch.h"

#ifndef NDEBUG
#define NDEBUG
#endif

//#undef NDEBUG

#define VIENNACL_WITH_UBLAS 1

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/sliced_ell_matrix.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/vector.hpp"

#include <eigen3/eigen/Eigen>

#include "ga.h"
#include <algorithm>
#include <array>
//#include <boost/math/differentiaton/finite_difference.hpp>  //this needs a fairly recent version of boost.
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/timer/timer.hpp>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <math.h>
#include <omp.h>
#include <vector>

template <typename TFunc>
double TanhSinhIntegrate(double x, double y, TFunc f) {
    boost::math::quadrature::tanh_sinh<double> integrator;
    return integrator.integrate(f, x, y);
}

template <typename TFunc>
inline double TrapezoidIntegrate(double x, double y, TFunc f) {
    using boost::math::quadrature::trapezoidal;
    return trapezoidal(f, x, y);
}

inline double
BilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y) {
    double x2x1, y2y1, x2x, y2y, yy1, xx1;
    x2x1 = x2 - x1;
    y2y1 = y2 - y1;
    x2x = x2 - x;
    y2y = y2 - y;
    yy1 = y - y1;
    xx1 = x - x1;
    return 1.0 / (x2x1 * y2y1) * (q11 * x2x * y2y + q21 * xx1 * y2y + q12 * x2x * yy1 + q22 * xx1 * yy1);
}

std::vector<double> LinearSpacedArray(double a, double b, int N) {
    double h = (b - a) / static_cast<double>(N - 1);
    std::vector<double> xs(N);
    std::vector<double>::iterator x;
    double val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
        *x = val;
    }
    return xs;
}

/* setupInterpGrid is going to fill a 4d array with the A,B,C,D coefficients such that
for each i,j,k with j,k not on the top or right edge
for every x,y inside the box xset[j],xset[j+1],yset[k],yset[k+1]
interp2d(i,x,y,etc) = A+Bx+Cy+Dxy (approx)
*/
template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::setupInterpGrid() {
    //using namespace Eigen;
    for (int i = 0; i < rhocount; i++) {
        interpParameters(i, xcount - 1, ycount - 1, 0) = 0; // we set the top-right grid points to 0
        interpParameters(i, xcount - 1, ycount - 1, 1) = 0; //so we can point to them later.
        interpParameters(i, xcount - 1, ycount - 1, 2) = 0; //otherwise the right and top edges are not used.
        interpParameters(i, xcount - 1, ycount - 1, 3) = 0;
        for (int j = 0; j < xcount - 1; j++)
            for (int k = 0; k < ycount - 1; k++) {
                double Q11 = gridValues(i, j, k),
                       Q12 = gridValues(i, j + 1, k),
                       Q21 = gridValues(i, j, k + 1),
                       Q22 = gridValues(i, j + 1, k + 1);

                double x = xset[j],
                       a = xset[j + 1],
                       y = yset[k],
                       b = yset[k + 1];
                double denom = (a - x) * (b - y);
                interpParameters(i, j, k, 0) = (a * b * Q11 - a * y * Q12 - b * x * Q21 + x * y * Q22) / denom;
                interpParameters(i, j, k, 1) = (-b * Q11 + y * Q12 + b * Q21 - y * Q22) / denom;
                interpParameters(i, j, k, 2) = (-a * Q11 + a * Q12 + x * Q21 - x * Q22) / denom;
                interpParameters(i, j, k, 3) = (Q11 - Q12 - Q21 + Q22) / denom;
            }
    }
}

// Using an existing derivs grid, the below computes bicubic interpolation parameters.
//16 parameters per patch, and the bicubic is of the form a_{ij} x^i y^j for 0 \leq x,y \leq 3
template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::setupBicubicGrid() {
    using namespace Eigen;
    bicubicParameterGrid &b = bicubicParameters;
    derivsGrid &d = derivs;
    for (int i = 0; i < rhocount; i++) {
        //we explicitly rely on parameters being initialized to 0, including the top and right sides.
        for (int j = 0; j < xcount - 1; j++)
            for (int k = 0; k < ycount - 1; k++) {
                double x0 = xset[j], x1 = xset[j + 1];
                double y0 = yset[k], y1 = yset[k + 1];
                Matrix<double, 4, 4> X, Y, RHS, A;
                RHS << d(i, j, k, 0), d(i, j, k + 1, 0), d(i, j, k, 2), d(i, j, k + 1, 2),
                    d(i, j + 1, k, 0), d(i, j + 1, k + 1, 0), d(i, j + 1, k, 2), d(i, j + 1, k + 1, 2),
                    d(i, j, k, 1), d(i, j, k + 1, 1), d(i, j, k, 3), d(i, j, k + 1, 3),
                    d(i, j + 1, k, 1), d(i, j + 1, k + 1, 1), d(i, j + 1, k, 3), d(i, j + 1, k + 1, 3);
                X << 1, x0, x0 * x0, x0 * x0 * x0,
                    1, x1, x1 * x1, x1 * x1 * x1,
                    0, 1, 2 * x0, 3 * x0 * x0,
                    0, 1, 2 * x1, 3 * x1 * x1;
                Y << 1, 1, 0, 0,
                    y0, y1, 1, 1,
                    y0 * y0, y1 * y1, 2 * y0, 2 * y1,
                    y0 * y0 * y0, y1 * y1 * y1, 3 * y0 * y0, 3 * y1 * y1;

                A = X.inverse() * RHS * Y.inverse();
                for (int t = 0; t < 16; ++t)
                    b(i, j, k, t) = A(t % 4, t / 4); // PLEASE ADD SOME ASSERTS HERE
            }
    }
}

// setupDerivsGrid assumes the values of f are already in gridValues
// then populates derivs with vectors of the form [f,f_x,f_y, f_xy]
// we are using finite difference
//the below is a linear transform, and we will hopefully get it into a (sparse) matrix soon
//the below uses 5-point stencils (including at edges) for f_x and f_y
//16-point stencils for f_xy where are derivatives are available
// 4-point stencils for f_xy one row or column from edges
// f_xy at edges is hardcoded to 0.
template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::setupDerivsGrid() {
    double ydenom = yset[1] - yset[0];
    double xdenom = xset[1] - xset[0];
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
            derivs(i, 1, k, 1) = (-3.0 * g(i, 0, k) +
                                  -10.0 * g(i, 1, k) +
                                  18 * g(i, 2, k) +
                                  -6 * g(i, 3, k) +
                                  1 * g(i, 4, k)) /
                                 (12.0 * xdenom);

            derivs(i, xcount - 2, k, 1) = (3.0 * g(i, xcount - 1, k) +
                                           10.0 * g(i, xcount - 2, k) +
                                           -18.0 * g(i, xcount - 3, k) +
                                           6.0 * g(i, xcount - 4, k) +
                                           -1.0 * g(i, xcount - 5, k)) /
                                          (12.0 * xdenom);
            for (int j = 2; j <= xcount - 3; j++)
                derivs(i, j, k, 1) = (1.0 * g(i, j - 2, k) +
                                      -8.0 * g(i, j - 1, k) +
                                      0.0 * g(i, j, k) +
                                      8.0 * g(i, j + 1, k) +
                                      -1.0 * g(i, j + 2, k)) /
                                     (12.0 * xdenom);
        }
        for (int j = 0; j < xcount; j++) {

            derivs(i, j, 0, 2) = 0;
            derivs(i, j, ycount - 1, 2) = 0;
            derivs(i, j, 1, 2) = (-3.0 * g(i, j, 0) +
                                  -10.0 * g(i, j, 1) +
                                  18.0 * g(i, j, 2) +
                                  -6.0 * g(i, j, 3) +
                                  1.0 * g(i, j, 4)) /
                                 (12.0 * ydenom);
            derivs(i, j, ycount - 2, 2) = (3.0 * g(i, j, ycount - 1) +
                                           10.0 * g(i, j, ycount - 2) +
                                           -18.0 * g(i, j, ycount - 3) +
                                           6.0 * g(i, j, ycount - 4) +
                                           -1 * g(i, j, ycount - 5)) /
                                          (12.0 * ydenom);
            for (int k = 2; k < ycount - 3; k++) {
                derivs(i, j, k, 2) = (1.0 * g(i, j, k - 2) +
                                      -8.0 * g(i, j, k - 1) +
                                      0.0 * g(i, j, k) +
                                      8.0 * g(i, j, k + 1) +
                                      -1.0 * g(i, j, k + 2)) /
                                     (12.0 * ydenom);
            }
        };
        for (int j = 2; j < xcount - 2; ++j) {
            for (int k = 2; k < ycount - 2; ++k) {
                derivs(i, j, k, 3) = (8 * (g(i, j + 1, k - 2) + g(i, j + 2, k - 1) + g(i, j - 2, k + 1) + g(i, j - 1, k + 2)) +
                                      -8 * (g(i, j - 1, k - 2) + g(i, j - 2, k - 1) + g(i, j + 1, k + 2) + g(i, j + 2, k + 1)) +
                                      -1 * (g(i, j + 2, k - 2) + g(i, j - 2, k + 2) - g(i, j - 2, k - 2) - g(i, j + 2, k + 2)) +
                                      64 * (g(i, j - 1, k - 1) + g(i, j + 1, k + 1) - g(i, j + 1, k - 1) - g(i, j - 1, k + 1))) /
                                     (144 * xdenom * ydenom);
            }
        }
        for (int j = 1; j < xcount - 1; j++) {
            derivs(i, j, 1, 3) = (g(i, j - 1, 0) + g(i, j + 1, 1 + 1) - g(i, j + 1, 1 - 1) - g(i, j - 1, 1 + 1)) /
                                 (4 * xdenom * ydenom);
            derivs(i, j, ycount - 2, 3) = (g(i, j - 1, ycount - 2 - 1) + g(i, j + 1, ycount - 2 + 1) -
                                           g(i, j + 1, ycount - 2 - 1) - g(i, j - 1, ycount - 2 + 1)) /
                                          (4 * xdenom * ydenom);
        }
        for (int k = 1; k < ycount - 1; k++) {
            derivs(i, 1, k, 3) = (g(i, 1 - 1, k - 1) + g(i, 1 + 1, k + 1) -
                                  g(i, 1 + 1, k - 1) - g(i, 1 - 1, k + 1)) /
                                 (4 * xdenom * ydenom);

            derivs(i, xcount - 2, k, 3) = (g(i, xcount - 2 - 1, k - 1) + g(i, xcount - 2 + 1, k + 1) -
                                           g(i, xcount - 2 + 1, k - 1) - g(i, xcount - 2 - 1, k + 1)) /
                                          (4 * xdenom * ydenom);
        }
    }
}

//arcIntegral computes the analytic integral of a bilinear function over an arc of a circle
//the circle is centered at xc,yc with radius rho, and the function is determined
//by a previously setup interp grid.

template <int rhocount, int xcount, int ycount>
std::array<double, 4> GyroAveragingGrid<rhocount, xcount, ycount>::arcIntegral(
    double rho, double xc, double yc, double s0, double s1) {
    std::array<double, 4> coeffs;
    double coss1 = std::cos(s1), coss0 = std::cos(s0), sins0 = std::sin(s0), sins1 = std::sin(s1);
    coeffs[0] = s1 - s0;
    coeffs[1] = (s1 * xc - rho * coss1) - (s0 * xc - rho * coss0);
    coeffs[2] = (s1 * yc - rho * sins1) - (s0 * yc - rho * sins0);
    coeffs[3] = (s1 * xc * yc - rho * yc * coss1 - rho * xc * sins1 + rho * rho * coss1 * coss1 / 2.0) -
                (s0 * xc * yc - rho * yc * coss0 - rho * xc * sins0 + rho * rho * coss0 * coss0 / 2.0);
    return coeffs;
}
/* This comment is to help recall precisely how to index the a_ij coefficients, can be deleted later.
for (int i = 0; i <= 3; ++i)
    for (int j = 0; j <= 3; ++j) {
        result += bicubicParameters(rhoindex, xindex, yindex, j * 4 + i) * xns[i] * yns[j];
    }
return result;
*/
std::array<double, 16> arcIntegralBicubic(
    double rho, double xc, double yc, double s0, double s1) {
    //we are going to write down indefinite integrals of (xc + rho*sin(x))^i (xc-rho*cos(x))^j
    // it seems like in c++, we can't make a 2d-array of lambdas that bind (rho, xc, yc)
    //without paying a performance penalty.
    //The below is not meant to be so human readable.  It was generated partly by mathematica.

    //crazy late-night thought - can this whole thing be constexpr?
    const double c = xc;
    const double d = yc;
    const double r = rho;
    using std::cos;
    using std::sin;
    auto f00 = [](double x) -> double { return x; };
    auto f10 = [r, c, d](double x) -> double { return c * x - r * cos(x); };
    auto f20 = [r, c, d](double x) -> double { return c * c * x - 2 * c * r * cos(x) + r * r * x / 2 - r * r * sin(2 * x) / 4; };
    auto f30 = [r, c, d](double x) -> double { return (1.0 / 12.0) *
                                                      (3 * c * (4 * c * c * x + 6 * r * r * x - 3 * r * r * sin(2 * x)) -
                                                       9 * r * (4 * c * c + r * r) * cos(x) + r * r * r * cos(3 * x)); };
    auto f01 = [r, c, d](double x) -> double { return d * x - r * sin(x); };
    auto f11 = [r, c, d](double x) -> double { return c * d * x - c * r * sin(x) - d * r * cos(x) + r * r * cos(x) * cos(x) / 2.0; };
    auto f21 = [r, c, d](double x) -> double { return (1.0 / 12.0) * (12.0 * c * c * d * x - 12 * c * c * r * sin(x) -
                                                                      24 * c * d * r * cos(x) + 6 * c * r * r * cos(2 * x) + 6 * d * r * r * x -
                                                                      3 * d * r * r * sin(2 * x) - 3 * r * r * r * sin(x) + r * r * r * sin(3 * x)); };
    auto f31 = [r, c, d](double x) -> double { return (1.0 / 96.0) * (48 * c * d * x * (2 * c * c + 3 * r * r) - 72 * d * r * (4 * c * c + r * r) * cos(x) -
                                                                      24 * c * r * (4 * c * c + 3 * r * r) * sin(x) + 12 * r * r * (6 * c * c + r * r) * cos(2 * x) -
                                                                      72 * c * d * r * r * sin(2 * x) + 24 * c * r * r * r * sin(3 * x) + 8 * d * r * r * r * cos(3 * x) -
                                                                      3 * r * r * r * r * cos(4 * x)); };

    auto f02 = [r, c, d](double x) -> double { return (1.0 / 2.0) * (x * (2 * d * d + r * r) + r * sin(x) * (r * cos(x) - 4 * d)); };
    auto f12 = [r, c, d](double x) -> double { return (1.0 / 12.0) * (6 * c * x * (2 * d * d + r * r) - 24 * c * d * r * sin(x) +
                                                                      3 * c * r * r * sin(2 * x) - 3 * r * (4 * d * d + r * r) * cos(x) + 6 * d * r * r * cos(2 * x) +
                                                                      r * r * r * (-1.0 * cos(3 * x))); };
    auto f22 = [r, c, d](double x) -> double { return (1.0 / 96.0) *
                                                      (24 * r * r * (c * c - d * d) * sin(2 * x) + 12 * x * (4 * c * c * (2 * d * d + r * r) + 4 * d * d * r * r + r * r * r * r) -
                                                       48 * d * r * (4 * c * c + r * r) * sin(x) - 48 * c * r * (4 * d * d + r * r) * cos(x) + 96 * c * d * r * r * cos(2 * x) -
                                                       16 * c * r * r * r * cos(3 * x) + 16 * d * r * r * r * sin(3 * x) - 3 * r * r * r * r * sin(4 * x)); };
    auto f32 = [r, c, d](double x) -> double { return (1.0 / 480.0) *
                                                      (120 * c * r * r * (c * c - 3 * d * d) * sin(2 * x) +
                                                       60 * c * x * (4 * c * c * (2 * d * d + r * r) + 3 * (4 * d * d * r * r + r * r * r * r)) -
                                                       60 * r * cos(x) * (6 * c * c * (4 * d * d + r * r) + 6 * d * d * r * r + r * r * r * r) -
                                                       10 * r * r * r * cos(3 * x) * (12 * c * c - 4 * d * d + r * r) -
                                                       240 * c * d * r * (4 * c * c + 3 * r * r) * sin(x) +
                                                       120 * d * r * r * (6 * c * c + r * r) * cos(2 * x) +
                                                       240 * c * d * r * r * r * sin(3 * x) - 45 * c * r * r * r * r * sin(4 * x) -
                                                       30 * d * r * r * r * r * cos(4 * x) + 6 * r * r * r * r * r * cos(5 * x)); };

    auto f03 = [r, c, d](double x) -> double { return (1.0 / 12.0) * (12 * d * d * d * x - 9 * r * (4 * d * d + r * r) * sin(x) +
                                                                      18 * d * r * r * x + 9 * d * r * r * sin(2 * x) - r * r * r * sin(3 * x)); };

    auto f13 = [r, c, d](double x) -> double { return (1.0 / 96.0) * (48 * c * d * x * (2 * d * d + 3 * r * r) -
                                                                      72 * c * r * (4 * d * d + r * r) * sin(x) +
                                                                      72 * c * d * r * r * sin(2 * x) -
                                                                      8 * c * r * r * r * sin(3 * x) +
                                                                      12 * r * r * (6 * d * d + r * r) * cos(2 * x) -
                                                                      24 * d * r * (4 * d * d + 3 * r * r) * cos(x) -
                                                                      24 * d * r * r * r * cos(3 * x) +
                                                                      3 * r * r * r * r * cos(4 * x)); };
    auto f23 = [r, c, d](double x) -> double { return (1.0 / 480.0) * (480 * c * c * d * d * d * x -
                                                                       1440 * c * c * d * d * r * sin(x) +
                                                                       720 * c * c * d * r * r * x +
                                                                       360 * c * c * d * r * r * sin(2 * x) -
                                                                       360 * c * c * r * r * r * sin(x) -
                                                                       40 * c * c * r * r * r * sin(3 * x) +
                                                                       120 * c * r * r * (6 * d * d + r * r) * cos(2 * x) -
                                                                       240 * c * d * r * (4 * d * d + 3 * r * r) * cos(x) -
                                                                       240 * c * d * r * r * r * cos(3 * x) +
                                                                       30 * c * r * r * r * r * cos(4 * x) +
                                                                       240 * d * d * d * r * r * x -
                                                                       120 * d * d * d * r * r * sin(2 * x) -
                                                                       360 * d * d * r * r * r * sin(x) +
                                                                       120 * d * d * r * r * r * sin(3 * x) +
                                                                       180 * d * r * r * r * r * x -
                                                                       45 * d * r * r * r * r * sin(4 * x) -
                                                                       60 * r * r * r * r * r * sin(x) +
                                                                       10 * r * r * r * r * r * sin(3 * x) +
                                                                       6 * r * r * r * r * r * sin(5 * x)); };
    auto f33 = [r, c, d](double x) -> double { return (1.0 / 960.0) * (960 * c * c * c * d * d * d * x -
                                                                       2880 * c * c * c * d * d * r * sin(x) +
                                                                       1440 * c * c * c * d * r * r * x +
                                                                       720 * c * c * c * d * r * r * sin(2 * x) -
                                                                       720 * c * c * c * r * r * r * sin(x) -
                                                                       80 * c * c * c * r * r * r * sin(3 * x) +
                                                                       45 * r * r * cos(2 * x) * (8 * c * c * (6 * d * d + r * r) + 8 * d * d * r * r + r * r * r * r) -
                                                                       360 * d * r * cos(x) * (c * c * (8 * d * d + 6 * r * r) + 2 * d * d * r * r + r * r * r * r) -
                                                                       720 * c * c * d * r * r * r * cos(3 * x) +
                                                                       90 * c * c * r * r * r * r * cos(4 * x) +
                                                                       1440 * c * d * d * d * r * r * x -
                                                                       720 * c * d * d * d * r * r * sin(2 * x) -
                                                                       2160 * c * d * d * r * r * r * sin(x) +
                                                                       720 * c * d * d * r * r * r * sin(3 * x) +
                                                                       1080 * c * d * r * r * r * r * x -
                                                                       270 * c * d * r * r * r * r * sin(4 * x) -
                                                                       360 * c * r * r * r * r * r * sin(x) +
                                                                       60 * c * r * r * r * r * r * sin(3 * x) +
                                                                       36 * c * r * r * r * r * r * sin(5 * x) +
                                                                       80 * d * d * d * r * r * r * cos(3 * x) -
                                                                       90 * d * d * r * r * r * r * cos(4 * x) -
                                                                       60 * d * r * r * r * r * r * cos(3 * x) +
                                                                       36 * d * r * r * r * r * r * cos(5 * x) -
                                                                       5 * r * r * r * r * r * r * cos(6 * x)); };

    std::array<double, 16> coeffs;
    double coss1 = std::cos(s1), coss0 = std::cos(s0), sins0 = std::sin(s0), sins1 = std::sin(s1);

    //result += bicubicParameters(rhoindex, xindex, yindex, j * 4 + i) * xns[i] * yns[j];
    coeffs[0 * 4 + 0] = f00(s1) - f00(s0);
    coeffs[0 * 4 + 1] = f10(s1) - f10(s0);
    coeffs[0 * 4 + 2] = f20(s1) - f20(s0);
    coeffs[0 * 4 + 3] = f30(s1) - f30(s0);
    coeffs[1 * 4 + 0] = f01(s1) - f01(s0);
    coeffs[1 * 4 + 1] = f11(s1) - f11(s0);
    coeffs[1 * 4 + 2] = f21(s1) - f21(s0);
    coeffs[1 * 4 + 3] = f31(s1) - f31(s0);
    coeffs[2 * 4 + 0] = f02(s1) - f02(s0);
    coeffs[2 * 4 + 1] = f12(s1) - f12(s0);
    coeffs[2 * 4 + 2] = f22(s1) - f22(s0);
    coeffs[2 * 4 + 3] = f32(s1) - f32(s0);
    coeffs[3 * 4 + 0] = f03(s1) - f03(s0);
    coeffs[3 * 4 + 1] = f13(s1) - f13(s0);
    coeffs[3 * 4 + 2] = f23(s1) - f23(s0);
    coeffs[3 * 4 + 3] = f33(s1) - f33(s0);

    return coeffs;
}

//add handling for rho =0.
template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::assembleFastGACalc(void) {
    //std::map<std::pair<int, int>, std::array<double, 4>> GAOffsetMap;

    std::vector<std::map<std::pair<int, int>, double>> GALookThruOffsetMapArray(rhocount);

#pragma omp parallel for
    for (auto i = 0; i < rhocount; i++) {
        std::map<std::pair<int, int>, double> &GALookThruOffsetMap = GALookThruOffsetMapArray[i];
        for (auto j = 0; j < xcount; j++)
            for (auto k = 0; k < ycount; k++) {
                std::vector<indexedPoint> intersections;
                double rho = rhoset[i];
                double xc = xset[j];
                double yc = yset[k];
                double xmin = xset[0], xmax = xset.back();
                double ymin = yset[0], ymax = yset.back();
                std::vector<double> xIntersections, yIntersections;
                //these two loops calculate all potential intersection points
                //between GA circle and the grid.
                for (auto v : xset)
                    if (std::abs(v - xc) <= rho) {
                        double deltax = v - xc;
                        double deltay = std::sqrt(rho * rho - deltax * deltax);
                        if ((yc + deltay >= ymin) && (yc + deltay <= ymax))
                            intersections.push_back(indexedPoint(v, yc + deltay, 0));
                        if ((yc - deltay >= ymin) && (yc - deltay <= ymax))
                            intersections.push_back(indexedPoint(v, yc - deltay, 0));
                    }
                for (auto v : yset)
                    if (std::abs(v - yc) <= rho) {
                        double deltay = v - yc;
                        double deltax = std::sqrt(rho * rho - deltay * deltay);
                        if ((xc + deltax >= xmin) && (xc + deltax <= xmax))
                            intersections.push_back(indexedPoint(xc + deltax, v, 0));
                        if ((xc - deltax >= xmin) && (xc - deltax <= xmax))
                            intersections.push_back(indexedPoint(xc - deltax, v, 0));
                    }
                for (auto &v : intersections) {
                    v.s = std::atan2(v.xvalue - xc, yc - v.yvalue);
                    if (v.s < 0)
                        v.s += 2 * pi;
                    assert((0 <= v.s) && (v.s < 2 * pi));
                    assert((xc + std::sin(v.s) * rho) - v.xvalue < 1e-10);
                    assert((yc - std::cos(v.s) * rho) - v.yvalue < 1e-10);
                }
                indexedPoint temp;
                temp.s = 0;
                intersections.push_back(temp);
                temp.s = 2 * pi;
                intersections.push_back(temp);
                std::sort(intersections.begin(), intersections.end(), [](indexedPoint a, indexedPoint b) { return a.s < b.s; });
                assert(intersections.size() > 0);
                assert(intersections[0].s == 0);
                assert(intersections.back().s == (2 * pi));
                for (size_t p = 0; p < intersections.size() - 1; p++) {
                    double s0 = intersections[p].s, s1 = intersections[p + 1].s;
                    double xmid, ymid;
                    std::array<double, 4> coeffs;
                    int xInterpIndex = 0, yInterpIndex = 0;
                    if (s1 - s0 < 1e-12)
                        continue;                                      //if two of our points are equal or very close to one another, we make the arc larger.
                                                                       //this will probably happen for s=0 and s=pi/2, but doesn't cost us anything.
                    integrand(rho, xc, yc, (s0 + s1) / 2, xmid, ymid); //this just calculates into (xmid,ymid) the point half through the arc.
                    coeffs = arcIntegral(rho, xc, yc, s0, s1);
                    interpIndexSearch(xmid, ymid, xInterpIndex, yInterpIndex);

                    //begin look-thru code
                    if (!((xInterpIndex == (xcount - 1)) && (yInterpIndex == (ycount - 1)))) {
                        double x = xset[xInterpIndex], a = xset[xInterpIndex + 1];
                        double y = yset[yInterpIndex], b = yset[yInterpIndex + 1];
                        double c1 = coeffs[0] / (2 * pi);
                        double c2 = coeffs[1] / (2 * pi);
                        double c3 = coeffs[2] / (2 * pi);
                        double c4 = coeffs[3] / (2 * pi);
                        double denom = (a - x) * (b - y);

                        std::array<int, 4> LTSources({0, 0, 0, 0}), LTTargets({0, 0, 0, 0});
                        std::array<double, 4> LTCoeffs({0, 0, 0, 0});
                        LTSources[0] = &(gridValues(i, xInterpIndex, yInterpIndex)) - &(gridValues(0, 0, 0));
                        LTSources[1] = &(gridValues(i, xInterpIndex + 1, yInterpIndex)) - &(gridValues(0, 0, 0));
                        LTSources[2] = &(gridValues(i, xInterpIndex, yInterpIndex + 1)) - &(gridValues(0, 0, 0));
                        LTSources[3] = &(gridValues(i, xInterpIndex + 1, yInterpIndex + 1)) - &(gridValues(0, 0, 0));
                        LTTargets[0] = &(fastGALTResult(i, j, k)) - &(fastGALTResult(0, 0, 0)); //eh these are all the same, delete the extras.
                        LTTargets[1] = &(fastGALTResult(i, j, k)) - &(fastGALTResult(0, 0, 0));
                        LTTargets[2] = &(fastGALTResult(i, j, k)) - &(fastGALTResult(0, 0, 0));
                        LTTargets[3] = &(fastGALTResult(i, j, k)) - &(fastGALTResult(0, 0, 0));

                        LTCoeffs[0] = (c1 * a * b - b * c2 - a * c3 + c4) / denom;  //coeff of Q11
                        LTCoeffs[1] = (-c4 - c1 * a * y + a * c3 + y * c2) / denom; //etc
                        LTCoeffs[2] = (b * c2 - c1 * b * x + x * c3 - c4) / denom;
                        LTCoeffs[3] = (c1 * x * y - y * c2 - x * c3 + c4) / denom;
                        for (int l = 0; l < 4; l++)
                            GALookThruOffsetMap[std::pair<int, int>(LTSources[l], LTTargets[l])] =
                                GALookThruOffsetMap[std::pair<int, int>(LTSources[l], LTTargets[l])] + LTCoeffs[l];
                    }
                }
            }
    }

    for (int i = 0; i < rhocount; i++) {
        auto &GALookThruOffsetMap = GALookThruOffsetMapArray[i];
        for (auto iter = GALookThruOffsetMap.begin(); iter != GALookThruOffsetMap.end(); ++iter) {
            LTOffset lto;
            lto.source = (iter->first).first;
            lto.target = (iter->first).second;
            lto.coeff = (iter->second);
            LTOffsetTensor.push_back(lto);
        }
    }
    std::sort(LTOffsetTensor.begin(), LTOffsetTensor.end(), [](LTOffset a, LTOffset b) -> bool { 
				if(a.source == b.source)
                    return a.target < b.target;
                else
					return a.source < b.source; });
    std::cout << "Number of double  products needed for LT calc: " << LTOffsetTensor.size() << " and rough memory usage is " << LTOffsetTensor.size() * sizeof(LTOffset) << std::endl;
}

template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::assembleFastBCCalc(void) { //bicubic version of the above.
    //std::map<std::pair<int, int>, std::array<double, 4>> GAOffsetMap;

    std::vector<std::map<std::pair<int, int>, double>> GALookThruOffsetMapArray(rhocount);

#pragma omp parallel for
    for (auto i = 0; i < rhocount; i++) {
        std::map<std::pair<int, int>, double> &GALookThruOffsetMap = GALookThruOffsetMapArray[i];
        for (auto j = 0; j < xcount; j++)
            for (auto k = 0; k < ycount; k++) {
                std::vector<indexedPoint> intersections;
                double rho = rhoset[i];
                double xc = xset[j];
                double yc = yset[k];
                double xmin = xset[0], xmax = xset.back();
                double ymin = yset[0], ymax = yset.back();
                std::vector<double> xIntersections, yIntersections;
                //these two loops calculate all potential intersection points
                //between GA circle and the grid.
                for (auto v : xset)
                    if (std::abs(v - xc) <= rho) {
                        double deltax = v - xc;
                        double deltay = std::sqrt(rho * rho - deltax * deltax);
                        if ((yc + deltay >= ymin) && (yc + deltay <= ymax))
                            intersections.push_back(indexedPoint(v, yc + deltay, 0));
                        if ((yc - deltay >= ymin) && (yc - deltay <= ymax))
                            intersections.push_back(indexedPoint(v, yc - deltay, 0));
                    }
                for (auto v : yset)
                    if (std::abs(v - yc) <= rho) {
                        double deltay = v - yc;
                        double deltax = std::sqrt(rho * rho - deltay * deltay);
                        if ((xc + deltax >= xmin) && (xc + deltax <= xmax))
                            intersections.push_back(indexedPoint(xc + deltax, v, 0));
                        if ((xc - deltax >= xmin) && (xc - deltax <= xmax))
                            intersections.push_back(indexedPoint(xc - deltax, v, 0));
                    }
                for (auto &v : intersections) {
                    v.s = std::atan2(v.xvalue - xc, yc - v.yvalue);
                    if (v.s < 0)
                        v.s += 2 * pi;
                    assert((0 <= v.s) && (v.s < 2 * pi));
                    assert((xc + std::sin(v.s) * rho) - v.xvalue < 1e-10);
                    assert((yc - std::cos(v.s) * rho) - v.yvalue < 1e-10);
                }
                indexedPoint temp;
                temp.s = 0;
                intersections.push_back(temp);
                temp.s = 2 * pi;
                intersections.push_back(temp);
                std::sort(intersections.begin(), intersections.end(), [](indexedPoint a, indexedPoint b) { return a.s < b.s; });
                assert(intersections.size() > 0);
                assert(intersections[0].s == 0);
                assert(intersections.back().s == (2 * pi));
                for (size_t p = 0; p < intersections.size() - 1; p++) {
                    double s0 = intersections[p].s, s1 = intersections[p + 1].s;
                    double xmid, ymid;
                    std::array<double, 16> coeffs;
                    int xInterpIndex = 0, yInterpIndex = 0;
                    if (s1 - s0 < 1e-12)                               //TODO MAGIC NUMBER (here and another place )
                        continue;                                      //if two of our points are equal or very close to one another, we make the arc larger.
                                                                       //this will probably happen for s=0 and s=pi/2, but doesn't cost us anything.
                    integrand(rho, xc, yc, (s0 + s1) / 2, xmid, ymid); //this just calculates into (xmid,ymid) the point half through the arc.
                    coeffs = arcIntegralBicubic(rho, xc, yc, s0, s1);
                    interpIndexSearch(xmid, ymid, xInterpIndex, yInterpIndex);

                    //begin look-thru code
                    if (!((xInterpIndex == (xcount - 1)) && (yInterpIndex == (ycount - 1)))) {
                        double x = xset[xInterpIndex], a = xset[xInterpIndex + 1];
                        double y = yset[yInterpIndex], b = yset[yInterpIndex + 1];

                        std::array<int, 16> LTSources, LTTargets;
                        std::array<double, 16> LTCoeffs;
                        LTSources.fill(0);
                        LTTargets.fill(0);
                        LTCoeffs.fill(0);
                        for (int l = 0; l < 16; l++) {
                            LTSources[l] = &(bicubicParameters(i, xInterpIndex, yInterpIndex, l)) - &(bicubicParameters(0, 0, 0, 0));
                            LTTargets[l] = &(BCResult(i, j, k)) - &(BCResult(0, 0, 0));
                            LTCoeffs[l] = coeffs[l] / (2.0 * pi);

                            GALookThruOffsetMap[std::pair<int, int>(LTSources[l], LTTargets[l])] =
                                GALookThruOffsetMap[std::pair<int, int>(LTSources[l], LTTargets[l])] + LTCoeffs[l];
                        }
                    }
                }
            }
    }

    for (int i = 0; i < rhocount; i++) {
        auto &GALookThruOffsetMap = GALookThruOffsetMapArray[i];
        for (auto iter = GALookThruOffsetMap.begin(); iter != GALookThruOffsetMap.end(); ++iter) {
            LTOffset lto;
            lto.source = (iter->first).first;
            lto.target = (iter->first).second;
            lto.coeff = (iter->second);
            BCOffsetTensor.push_back(lto);
        }
    }
    std::sort(BCOffsetTensor.begin(), BCOffsetTensor.end(), [](LTOffset a, LTOffset b) -> bool { 
				if(a.source == b.source)
                    return a.target < b.target;
                else
					return a.source < b.source; });
    std::cout << "Number of double  products needed for BC calc: " << BCOffsetTensor.size() << " and rough memory usage is " << BCOffsetTensor.size() * sizeof(LTOffset) << std::endl;
}

template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::fastLTCalcOffset() {
    clearGrid(fastGALTResult);
    int size = LTOffsetTensor.size();
    //#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        fastGALTResult.data[LTOffsetTensor[i].target] += gridValues.data[LTOffsetTensor[i].source] * LTOffsetTensor[i].coeff;
    }
}

template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::fastBCCalcOffset() {
    clearGrid(BCResult);
    long int size = BCOffsetTensor.size();
    //#pragma omp parallel for
    for (long int i = 0; i < size; ++i) {
        BCResult.data[BCOffsetTensor[i].target] += bicubicParameters.data[BCOffsetTensor[i].source] * BCOffsetTensor[i].coeff;
    }
}

template <int rhocount, int xcount, int ycount>
template <typename TFunc1, typename TFunc2>
void GyroAveragingGrid<rhocount, xcount, ycount>::GPUTestSuite(TFunc1 f, TFunc2 analytic) {

    boost::timer::auto_cpu_timer t;

    //begin ViennaCL calc

    std::vector<std::map<int, double>>
        cpu_sparse_matrix(xcount * ycount * rhocount);
    viennacl::compressed_matrix<double> vcl_sparse_matrix(xcount * ycount * rhocount, xcount * ycount * rhocount);
    boost::numeric::ublas::compressed_matrix<double> ublas_matrix(xcount * ycount * rhocount, xcount * ycount * rhocount);
    t.start();

    for (int m = 0; m < LTOffsetTensor.size(); ++m) {
        cpu_sparse_matrix[LTOffsetTensor[m].target][LTOffsetTensor[m].source] = LTOffsetTensor[m].coeff;
    }
    viennacl::copy(cpu_sparse_matrix, vcl_sparse_matrix);

    viennacl::backend::finish();
    t.report();
    std::cout << "That was the time to create the CPU matrix and copy it once to GPU." << std::endl;
    t.start();
    viennacl::copy(vcl_sparse_matrix, ublas_matrix);
    viennacl::backend::finish();
    t.report();
    t.start();
    std::cout << "That was the time just to setup the ublas matrix." << std::endl;

    viennacl::compressed_matrix<double, 1> vcl_compressed_matrix_1(xcount * ycount * rhocount, xcount * ycount * rhocount);
    viennacl::compressed_matrix<double, 4> vcl_compressed_matrix_4(xcount * ycount * rhocount, xcount * ycount * rhocount);
    viennacl::compressed_matrix<double, 8> vcl_compressed_matrix_8(xcount * ycount * rhocount, xcount * ycount * rhocount);
    viennacl::coordinate_matrix<double> vcl_coordinate_matrix_128(xcount * ycount * rhocount, xcount * ycount * rhocount);
    viennacl::ell_matrix<double, 1> vcl_ell_matrix_1();
    viennacl::hyb_matrix<double, 1> vcl_hyb_matrix_1();
    viennacl::sliced_ell_matrix<double> vcl_sliced_ell_matrix_1(xcount * ycount * rhocount, xcount * ycount * rhocount);

    viennacl::vector<double> gpu_source(gridValues.data.size());
    viennacl::vector<double> gpu_target(gridValues.data.size());
    copy(cpu_sparse_matrix, vcl_sparse_matrix); //default alignment, benchmark different options.
    copy(gridValues.data.begin(), gridValues.data.end(), gpu_source.begin());
    copy(gridValues.data.begin(), gridValues.data.end(), gpu_target.begin());
    //we are going to compute each product once and then sync, to compile all kernels.
    //this will feel like a ~1 second delay in user space.

    viennacl::copy(ublas_matrix, vcl_compressed_matrix_1);
    viennacl::copy(ublas_matrix, vcl_compressed_matrix_4);
    viennacl::copy(ublas_matrix, vcl_compressed_matrix_8);
    viennacl::copy(ublas_matrix, vcl_coordinate_matrix_128);
    //    viennacl::copy(cpu_sparse_matrix,vcl_ell_matrix_1);
    //viennacl::copy(ublas_matrix,vcl_hyb_matrix_1);
    viennacl::copy(cpu_sparse_matrix, vcl_sliced_ell_matrix_1);
    viennacl::backend::finish();
    t.report();
    t.start();
    std::cout << "That was the time to copy everything onto the GPU." << std::endl;

    fullgrid cpu_results[8];

    gpu_target = viennacl::linalg::prod(vcl_sparse_matrix, gpu_source);
    viennacl::backend::finish();

    gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_1, gpu_source);
    viennacl::copy(gpu_target.begin(), gpu_target.end(), cpu_results[0].data.begin());

    gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_4, gpu_source);
    gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_8, gpu_source);
    gpu_target = viennacl::linalg::prod(vcl_coordinate_matrix_128, gpu_source);
    //gpu_target = viennacl::linalg::prod(vcl_ell_matrix_1,gpu_source);
    //gpu_target = viennacl::linalg::prod(vcl_hyb_matrix_1,gpu_source);
    gpu_target = viennacl::linalg::prod(vcl_sliced_ell_matrix_1, gpu_source);

    viennacl::backend::finish();
    viennacl::copy(gpu_target.begin(), gpu_target.end(), cpu_results[0].data.begin());
    viennacl::backend::finish();

    t.report();
    std::cout << "That was the time to do all of the products, and copy the result back only once." << std::endl;

    constexpr int gputimes = 3;
    //At this point everything has been done once.  We start benchmarking.  We are going to include cost of vectors transfers back and forth.

    t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(gridValues.data.begin(), gridValues.data.end(), gpu_source.begin());
        viennacl::backend::finish();
        gpu_target = viennacl::linalg::prod(vcl_sparse_matrix, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(), cpu_results[0].data.begin());
        viennacl::backend::finish();
    }
    t.report();
    std::cout << "That was the full cycle time to do " << gputimes << "  products using default sparse matrix." << std::endl;

    t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(gridValues.data.begin(), gridValues.data.end(), gpu_source.begin());
        viennacl::backend::finish();
        gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_1, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(), cpu_results[1].data.begin());
        viennacl::backend::finish();
    }
    t.report();
    std::cout << "That was the full cycle time to do " << gputimes << "  products using compressed_matrix_1 matrix." << std::endl;

    t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(gridValues.data.begin(), gridValues.data.end(), gpu_source.begin());
        viennacl::backend::finish();
        gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_4, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(), cpu_results[2].data.begin());
        viennacl::backend::finish();
    }
    t.report();
    std::cout << "That was the full cycle time to do " << gputimes << "  products using compressed_matrix_4 matrix." << std::endl;

    t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(gridValues.data.begin(), gridValues.data.end(), gpu_source.begin());
        viennacl::backend::finish();
        gpu_target = viennacl::linalg::prod(vcl_compressed_matrix_8, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(), cpu_results[3].data.begin());
        viennacl::backend::finish();
    }
    t.report();
    std::cout << "That was the full cycle time to do " << gputimes << "  products using compressed_matrix_8 matrix." << std::endl;

    t.start();
    /*    for(int count =0; count<gputimes;++count){
      copy(gridValues.data.begin(),gridValues.data.end(),gpu_source.begin());
      viennacl::backend::finish();
      gpu_target = viennacl::linalg::prod(vcl_coordinate_matrix_128,gpu_source);
      viennacl::backend::finish();
      viennacl::copy(gpu_target.begin(),gpu_target.end(),cpu_results[4].data.begin());
      viennacl::backend::finish();
    }
    t.report();*/
    std::cout << "That was the full cycle time to do " << gputimes * 0 << "  products using coordinate_matrix_128 matrix." << std::endl;

    t.start();
    for (int count = 0; count < gputimes; ++count) {
        copy(gridValues.data.begin(), gridValues.data.end(), gpu_source.begin());
        viennacl::backend::finish();
        gpu_target = viennacl::linalg::prod(vcl_sliced_ell_matrix_1, gpu_source);
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(), cpu_results[4].data.begin());
        viennacl::backend::finish();
    }
    t.report();
    std::cout << "That was the full cycle time to do " << gputimes << "  products using sliced_ell_matrix_1 matrix." << std::endl;

    std::cout << "Next we report errors for each GPU calc (in above order) vs CPU dot-product calc.  Here we only report maxabs norm" << std::endl;
    for (int i = 0; i < rhocount; i++) {
        std::cout.precision(5);
        std::cout << std::fixed << rhoset[i] << std::scientific << std::setw(15) << maxNormDiff(fastGALTResult, cpu_results[0], i) << "\t" << maxNormDiff(fastGALTResult, cpu_results[1], i) << "\t" << maxNormDiff(fastGALTResult, cpu_results[2], i) << "\t" << maxNormDiff(fastGALTResult, cpu_results[3], i) << "\t" << maxNormDiff(fastGALTResult, cpu_results[4], i) << std::endl;
    }

    //end ViennaCL calc
}

/* Run test suite.  We expect the test suite to:
1)     Calculate the gyroaverage transform of f, using f on a full grid, at each grid point
2)     above, using f but truncating it to be 0 outside of our grid
3)     above, using bilinear interp of f (explicitly only gridpoint valuations are allowed)
		and we trapezoid rule the interp-ed, truncated function  (vs (2),  only interp error introduced)
4)     fast dot-product calc: see AssembleFastGACalc for details
We report evolution of errorand sample timings.
   */

template <int rhocount, int xcount, int ycount>
template <typename TFunc1, typename TFunc2>
void GyroAveragingGrid<rhocount, xcount, ycount>::GyroAveragingTestSuite(TFunc1 f, TFunc2 analytic) {
    boost::timer::auto_cpu_timer t;
    fill(gridValues, f); //This is the base grid of values we will interpolate.
    t.start();
    fill(analytic_averages, analytic); //analytic formula for gyroaverages
    t.report();
    std::cout << "That was the time required to calculate analytic gyroaverages.\n";
    setupInterpGrid();
    t.start();
    fillAlmostExactGA(almostExactGA, f);
    t.report();
    std::cout << "That was the time required to calculate gyroaverages from the definition, with the trapezoid rule.\n";
    t.start();
    fillTruncatedAlmostExactGA(truncatedAlmostExactGA, f);
    t.report();
    std::cout << "That was the time required to calculate gryoaverages by def (as above), except we hard truncated f() to 0 off-grid.\n";
    t.start();
    fillTrapezoidInterp(trapezoidInterp, f);
    t.report();
    std::cout << "That was the time required to calc gyroaverages by def, replacing f() by its bilinear interpolant." << std::endl;
    t.start();
    assembleFastGACalc();
    t.report();
    std::cout << "That was the time required to assemble the sparse matrix in the fast-GA dot product calculation." << std::endl;
    t.start();
    setupDerivsGrid();
    setupBicubicGrid();
    assembleFastBCCalc();
    t.report();
    std::cout << "That was the time required to assemble the sparse matrix in the fast-BC dot product calculation." << std::endl;

    t.start();
    int times = 10;
    for (int counter = 0; counter < times; counter++) {
        fastLTCalcOffset();
    }

    t.report();
    std::cout << "The was the time require to run LT gyroaverage calc " << times << " times. \n " << std::endl;

    t.start();
    for (int counter = 0; counter < times; counter++) {
        setupDerivsGrid();
        setupBicubicGrid();
        fastBCCalcOffset();
    }

    t.report();
    std::cout << "The was the time require to run BC gyroaverage calc " << times << " times. \n " << std::endl;

    GPUTestSuite(f, analytic);
    setupDerivsGrid();
    setupBicubicGrid();
    fullgrid bicubicResults;
    fillBicubicInterp(bicubicResults);
    std::cout
        << "Below are some summary statistics for various grid.  Under each header is a pair of values.  The first is the RMS of a matrix, the second is the max absolute entry in a matrix.\n";

    std::cout << "rho        Input Grid                   Analytic Estimates              Trapezoid Rule                  Trap Rule truncation            Trapezoid rule of bilin interp  Fast dot-product GA\n";

    for (int i = 0; i < rhocount; i++) {

        std::cout.precision(5);
        std::cout << std::fixed << rhoset[i] << std::scientific << std::setw(15) << RMSNorm(gridValues, i) << "\t"
                  << maxNorm(gridValues, i) << "\t"
                  << RMSNorm(analytic_averages, i) << "\t"
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
                  << maxNorm(bicubicResults, i) << "\t"
                  << RMSNorm(BCResult, i) << "\t"
                  << maxNorm(BCResult, i) << "\n";
    }
    std::cout << "Diffs:\n";
    std::cout << "rho        Analytic vs Quadrature       Error due to truncation         Error due to interp             Analytic vs interp              interp vs dot-product GA          GPU vs CPU\n";
    for (int i = 0; i < rhocount; i++) {
        std::cout.precision(5);
        std::cout << std::fixed << rhoset[i] << std::scientific << std::setw(15)
                  << RMSNormDiff(analytic_averages, almostExactGA, i) << "\t"
                  << maxNormDiff(analytic_averages, almostExactGA, i) << "\t"
                  << RMSNormDiff(almostExactGA, truncatedAlmostExactGA, i) << "\t"
                  << maxNormDiff(almostExactGA, truncatedAlmostExactGA, i) << "\t"
                  << RMSNormDiff(truncatedAlmostExactGA, trapezoidInterp, i) << "\t"
                  << maxNormDiff(truncatedAlmostExactGA, trapezoidInterp, i) << "\t"
                  << RMSNormDiff(analytic_averages, trapezoidInterp, i) << "\t"
                  << maxNormDiff(analytic_averages, trapezoidInterp, i) << "\t"
                  << RMSNormDiff(trapezoidInterp, fastGALTResult, i) << "\t"
                  << maxNormDiff(trapezoidInterp, fastGALTResult, i) << "\t"
                  << RMSNormDiff(bicubicResults, truncatedAlmostExactGA, i) << "\t"
                  << maxNormDiff(bicubicResults, truncatedAlmostExactGA, i) << "\t"
                  << RMSNormDiff(BCResult, truncatedAlmostExactGA, i) << "\t"
                  << maxNormDiff(BCResult, truncatedAlmostExactGA, i) << "\n";
        //     << RMSNormDiff(fastGALTResult, cpu_results[0], i) << "\t"
        //    << maxNormDiff(fastGALTResult, cpu_results[0], i) << "\n";

        //<< RMSNormDiff(fastGALTResult, fastGACalcResultOffset, i) << "\t"
        //<< maxNormDiff(fastGALTResult, fastGACalcResultOffset, i) << "\n";
    }
}

//below function returns the indices referring to lower left point of the grid box containing (x,y)
//it is not (yet) efficient.  In particular, we should probably explicitly assume equispaced grids and use that fact.
template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::interpIndexSearch(const double x, const double y, int &xindex, int &yindex) {

    if ((x < xset[0]) || (y < yset[0]) || (x > xset.back()) || (y > yset.back())) {
        xindex = xcount - 1; // the top right corner should have zeros.
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

template <int rhocount, int xcount, int ycount>
double GyroAveragingGrid<rhocount, xcount, ycount>::interp2d(int rhoindex, const double x, const double y) {
    assert((rhoindex >= 0) && (rhoindex < rhocount));
    if ((x <= xset[0]) || (y <= yset[0]) || (x >= xset.back()) || (y >= yset.back()))
        return 0;
    int xindex = 0, yindex = 0;
    interpIndexSearch(x, y, xindex, yindex);
    double result = BilinearInterpolation(gridValues(rhoindex, xindex, yindex), gridValues(rhoindex, xindex + 1, yindex),
                                          gridValues(rhoindex, xindex, yindex + 1), gridValues(rhoindex, xindex + 1, yindex + 1),
                                          xset[xindex], xset[xindex + 1], yset[yindex], yset[yindex + 1],
                                          x, y);

    return result;
}

//Below returns bicubic interp f(x,y), in the dumb way.  We should do multivariate horners method soon.
template <int rhocount, int xcount, int ycount>
double GyroAveragingGrid<rhocount, xcount, ycount>::interpNaiveBicubic(int rhoindex, const double x, const double y) {
    assert((rhoindex >= 0) && (rhoindex < rhocount));
    if ((x <= xset[0]) || (y <= yset[0]) || (x >= xset.back()) || (y >= yset.back()))
        return 0;
    int xindex = 0, yindex = 0;
    double result = 0;
    interpIndexSearch(x, y, xindex, yindex);
    double xns[4] = {1, x, x * x, x * x * x};
    double yns[4] = {1, y, y * y, y * y * y};
    for (int i = 0; i <= 3; ++i)
        for (int j = 0; j <= 3; ++j) {
            result += bicubicParameters(rhoindex, xindex, yindex, j * 4 + i) * xns[i] * yns[j];
        }
    return result;
}

int main() {

    /*   const double rhomax = 3, rhomin = 0.0;                //
	constexpr int xcount = 64, ycount = 64, rhocount = 8; //bump up to 64x64x24 later
	const double xmin = -5, xmax = 5;
	const double ymin = -5, ymax = 5;
	const double A = 1;
	double B = 2.0;*/

    gridDomain g;
    g.rhomax = 2.5;
    g.rhomin = 0;
    g.xmin = g.ymin = -5;
    g.xmax = g.ymax = 5;
    constexpr int xcount = 96, ycount = 96, rhocount = 24; //bump up to 64x64x35 later or 128x128x35
    constexpr double A = 2;
    constexpr double B = 2;
    constexpr double Normalizer = 50.0;
    std::vector<double> rhoset;
    std::vector<double> xset;
    std::vector<double> yset;

    auto verySmoothFunc = [](double row, double ex, double why) -> double {
        double temp = ex * ex + why * why;
        if (temp >= 25)
            return 0;
        else
            return (15 * std::exp(1 / (temp / 25.0 - 1.0)));
    };

    auto verySmoothFunc2 = [](double row, double ex, double why) -> double {
        double temp = ex * ex + why * why;
        if (temp >= 25)
            return 0;
        else
            return std::exp(-row) * (30 * std::exp(1 / (temp / 25.0 - 1.0)));
    };

    auto ZeroFunc = [](double row, double ex, double why) -> double { return 0; };
    auto constantFuncAnalytic = [](double row, double ex, double why) -> double { return 2 * pi; };
    auto linearFunc = [](double row, double ex, double why) -> double { return std::max(0.0, 5 - std::abs(ex) - std::abs(why)); };
    //auto constantFuncAnalytic = [](double row, double ex, double why) -> double {return 2*pi;};
    auto testfunc1 = [A, Normalizer](double row, double ex, double why) -> double { return Normalizer * exp(-A * (ex * ex + why * why)); };
    auto testfunc1_analytic = [A, Normalizer](double row, double ex, double why) -> double {
        return Normalizer * exp(-A * (ex * ex + why * why + row * row)) * boost::math::cyl_bessel_i(0, 2 * A * row * std::sqrt(ex * ex + why * why));
    };
    auto testfunc2 = [Normalizer, A, B](double row, double ex, double why) -> double { return Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic = [Normalizer, A, B](double row, double ex, double why) -> double { return Normalizer * exp(-B * row * row) * exp(-A * (ex * ex + why * why + row * row)) * boost::math::cyl_bessel_i(0, 2 * A * row * std::sqrt(ex * ex + why * why)); };
    auto testfunc2_analytic_dx = [Normalizer, A, B](double row, double ex, double why) -> double { return -2 * A * ex * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic_dy = [Normalizer, A, B](double row, double ex, double why) -> double { return -2 * A * why * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic_dx_dy = [Normalizer, A, B](double row, double ex, double why) -> double { return 4 * A * A * ex * why * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };

    rhoset = LinearSpacedArray(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray(g.xmin, g.xmax, xcount);
    yset = LinearSpacedArray(g.ymin, g.ymax, ycount);

    GyroAveragingGrid<rhocount, xcount, ycount> grid(rhoset, xset, yset);
    grid.GyroAveragingTestSuite(testfunc2, testfunc2_analytic);
    //derivTest(g, testfunc2, testfunc2_analytic_dx, testfunc2_analytic_dy, testfunc2_analytic_dx_dy);
    //interpAnalysis(g, testfunc2, testfunc2_analytic);
    //testInterpImprovement();
    //testArcIntegralBicubic();
}

void testArcIntegralBicubic() {
    constexpr double r = 0.3, s0 = 0.6, s1 = 2.2, xc = -0.2, yc = -1.4;
    auto coeffs = arcIntegralBicubic(r, xc, yc, s0, s1);
    for (int i = 0; i <= 3; ++i)
        for (int j = 0; j <= 3; ++j) {
            auto f = [i, j](double x) -> double { return std::pow(xc + r * std::sin(x), i) * std::pow(yc - r * std::cos(x), j); };
            double res = TanhSinhIntegrate(s0, s1, f);
            std::cout << i << "\t" << j << "\t" << res << "\t" << coeffs[j * 4 + i] << "\t" << res - coeffs[j * 4 + i] << "\n";
        }
}

template <int count, typename TFunc1, typename TFunc2>
void interpAnalysisInnerLoop(const gridDomain &g, TFunc1 f,
                             TFunc2 analytic) {
    constexpr int rhocount = 4; //TODO MAGIC NUMBER SHOULD BE PASSED IN
    std::vector<double> rhoset;
    std::vector<double> xset;
    std::vector<double> yset;
    rhoset = LinearSpacedArray(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray(g.xmin, g.xmax, count);
    yset = LinearSpacedArray(g.ymin, g.ymax, count);
    GyroAveragingGrid<rhocount, count, count> grid(rhoset, xset, yset);
    grid.InterpErrorAnalysis(f, analytic);
}

template <typename TFunc1, typename TFunc2>
void interpAnalysis(const gridDomain &g, TFunc1 f,
                    TFunc2 analytic) {

    constexpr int counts[] = {6, 12, 24, 48, 96, 192};
    interpAnalysisInnerLoop<counts[0]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[1]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[2]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[3]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[4]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[5]>(g, f, analytic);
}

template <typename TFunc1, typename TFunc2, typename TFunc3, typename TFunc4>
void derivTest(const gridDomain &g, TFunc1 f,
               TFunc2 f_x, TFunc3 f_y, TFunc4 f_xy) {

    constexpr int count = 36;
    constexpr int rhocount = 4;
    std::vector<double> rhoset;
    std::vector<double> xset;
    std::vector<double> yset;
    rhoset = LinearSpacedArray(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray(g.xmin, g.xmax, count);
    yset = LinearSpacedArray(g.ymin, g.ymax, count);
    GyroAveragingGrid<rhocount, count, count> grid(rhoset, xset, yset);
    grid.derivsErrorAnalysis(f, f_x, f_y, f_xy);
}

template <int rhocount, int xcount, int ycount>
template <typename TFunc1, typename TFunc2, typename TFunc3, typename TFunc4>
void GyroAveragingGrid<rhocount, xcount, ycount>::derivsErrorAnalysis(TFunc1 f, TFunc2 f_x, TFunc3 f_y, TFunc4 f_xy) {
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

    csvPrinter(analytic[0], 1);
    std::cout << std::endl;
    csvPrinter(numeric[0], 1);
    std::cout << std::endl;
    csvPrinter(analytic[1], 1);
    std::cout << std::endl;
    csvPrinter(numeric[1], 1);
    std::cout << std::endl;
    csvPrinter(analytic[2], 1);
    std::cout << std::endl;
    csvPrinter(numeric[2], 1);
    std::cout << std::endl;
    csvPrinter(analytic[3], 1);
    std::cout << std::endl;
    csvPrinter(numeric[3], 1);
    std::cout << std::endl;

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

template <int rhocount, int xcount, int ycount>
template <typename TFunc1, typename TFunc2>
void GyroAveragingGrid<rhocount, xcount, ycount>::InterpErrorAnalysis(TFunc1 f, TFunc2 analytic) {

    fill(gridValues, f);               //This is the base grid of values we will interpolate.
    fill(analytic_averages, analytic); //analytic formula for gyroaverages
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
    fillBicubicInterp(bicubicInterp, f);
    std::cout << "There were " << trapezoidInterp.data.size() << " entries. \n";
    std::cout << "Diffs:\n";
    std::cout << "rho        Analytic vs Quadrature       Error due to truncation         Error due to interp             Analytic vs interp              interp vs dot-product GA          GPU vs CPU\n";

    for (int i = 0; i < rhocount; i++) {
        std::cout.precision(5);
        std::cout << std::fixed << rhoset[i] << std::scientific << std::setw(15)
                  << RMSNormDiff(analytic_averages, almostExactGA, i) << "\t"
                  << maxNormDiff(analytic_averages, almostExactGA, i) << "\t"
                  << RMSNormDiff(almostExactGA, truncatedAlmostExactGA, i) << "\t"
                  << maxNormDiff(almostExactGA, truncatedAlmostExactGA, i) << "\t"
                  << RMSNormDiff(truncatedAlmostExactGA, trapezoidInterp, i) << "\t"
                  << maxNormDiff(truncatedAlmostExactGA, trapezoidInterp, i) << "\t"
                  << RMSNormDiff(analytic_averages, trapezoidInterp, i) << "\t"
                  << maxNormDiff(analytic_averages, trapezoidInterp, i) << "\t"
                  << RMSNormDiff(truncatedAlmostExactGA, bicubicInterp, i) << "\t"
                  << maxNormDiff(truncatedAlmostExactGA, bicubicInterp, i) << "\n";
    }
}
//below function is for testing and will be refactored later.
void testInterpImprovement() {

    gridDomain g;
    g.rhomax = 0.9;
    g.rhomin = 0;
    g.xmin = g.ymin = -3;
    g.xmax = g.ymax = 3;
    constexpr int xcount = 30, ycount = 30;
    constexpr int xcountb = xcount * 2;
    constexpr int ycountb = xcount * 2;
    constexpr double A = 2;
    constexpr double B = 2;
    constexpr double Normalizer = 50.0;

    constexpr int rhocount = 4; //TODO MAGIC NUMBER SHOULD BE PASSED IN
    std::vector<double> rhoset;
    std::vector<double> xset, xsetb;
    std::vector<double> yset, ysetb;
    rhoset = LinearSpacedArray(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray(g.xmin, g.xmax, xcount);
    yset = LinearSpacedArray(g.ymin, g.ymax, ycount);

    xsetb = LinearSpacedArray(g.xmin, g.xmax, xcountb);
    ysetb = LinearSpacedArray(g.ymin, g.ymax, ycountb);

    GyroAveragingGrid<rhocount, xcount, ycount> smallgrid(rhoset, xset, yset);
    GyroAveragingGrid<rhocount, xcountb, ycountb> biggrid(rhoset, xsetb, ysetb);
    //auto testfunc2 = [Normalizer, A, B](double row, double ex, double why) -> double { return (1) * row; };

    auto testfunc2 = [Normalizer, A, B](double row, double ex, double why) -> double { return Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic = [Normalizer, A, B](double row, double ex, double why) -> double { return Normalizer * exp(-B * row * row) * exp(-A * (ex * ex + why * why + row * row)) * boost::math::cyl_bessel_i(0, 2 * A * row * std::sqrt(ex * ex + why * why)); };
    auto testfunc2_analytic_dx = [Normalizer, A, B](double row, double ex, double why) -> double { return -2 * A * ex * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic_dy = [Normalizer, A, B](double row, double ex, double why) -> double { return -2 * A * why * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic_dx_dy = [Normalizer, A, B](double row, double ex, double why) -> double { return 4 * A * A * ex * why * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };

    smallgrid.fill(smallgrid.gridValues, testfunc2); //This is the base grid of values we will interpolate.
    //smallgrid.fill(smallgrid.analytic_averages, testfunc2_analytic); //analytic formula for gyroaverages
    smallgrid.setupInterpGrid();
    smallgrid.setupDerivsGrid();
    smallgrid.setupBicubicGrid();

    smallgrid.setupBicubicGrid();

    GyroAveragingGrid<rhocount, xcountb, ycountb>::fullgrid exact, lin, bic;
    for (int i = 0; i < rhocount; ++i)
        std::cout << rhoset;
    for (int i = 0; i < rhocount; ++i)
        for (int j = 0; j < xcountb; ++j)
            for (int k = 0; k < ycountb; ++k) {
                exact(i, j, k) = testfunc2(rhoset[i], xsetb[j], ysetb[k]);
                lin(i, j, k) = smallgrid.interp2d(i, xsetb[j], ysetb[k]);
                //bic(i, j, k) = smallgrid.interpNaiveBicubic(i, xsetb[j], ysetb[k]);
                bic(i, j, k) = smallgrid.interpNaiveBicubic(i, xsetb[j], ysetb[k]);
            }

    for (int i = 1; i < rhocount; ++i) {

        //smallgrid.csvPrinter(smallgrid.gridValues, i);
        //std::cout << "\n";

        biggrid.csvPrinter(exact, i);
        std::cout << "\n";
        biggrid.csvPrinter(lin, i);
        std::cout << "\n";
        biggrid.csvPrinter(bic, i);
        std::cout << "\n";
        biggrid.csvPrinter(bic, i);
        std::cout << "\n";

        biggrid.csvPrinterDiff(lin, exact, i);
        std::cout << "\n";
        biggrid.csvPrinterDiff(bic, exact, i);
        std::cout << "\n";
        biggrid.csvPrinterDiff(bic, exact, i);
        std::cout << "\n";
    }
    return;
}

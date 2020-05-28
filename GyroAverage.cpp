
#ifndef NDEBUG
#define NDEBUG
#endif

//#undef NDEBUG

#define VIENNACL_WITH_UBLAS 1
#define VIENNACL_HAVE_EIGEN 1

#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif

#ifdef INCL_MATH_CONSTANTS 
#include<math_constants.h>
#endif

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

#include <algorithm>
#include <array>
//#include <boost/math/differentiaton/finite_difference.hpp>  //this needs a fairly recent version of boost.  boo.
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/math/special_functions/bessel.hpp>
//#include <boost/timer/timer.hpp>
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
#include <eigen3/Eigen/Eigen>

#include "ga.h"





int main() {

    typedef double mainReal;


    gridDomain<mainReal> g;
    g.rhomax = 3;
    g.rhomin = 0;
    g.xmin = g.ymin = -5;
    g.xmax = g.ymax = 5;
    constexpr int xcount = 32, ycount = 32, rhocount = 24; //bump up to 64x64x35 later or 128x128x35
    constexpr mainReal A = 2;
    constexpr mainReal B = 2;
    constexpr mainReal Normalizer = 50.0;
    std::vector<mainReal> rhoset;
    std::vector<mainReal> xset;
    std::vector<mainReal> yset;

    auto verySmoothFunc = [](mainReal row, mainReal ex, mainReal why) -> mainReal {
        mainReal temp = ex * ex + why * why;
        if (temp >= 25)
            return 0;
        else
            return (15 * std::exp(1 / (temp / 25.0 - 1.0)));
    };

    auto verySmoothFunc2 = [](mainReal row, mainReal ex, mainReal why) -> mainReal {
        mainReal temp = ex * ex + why * why;
        if (temp >= 25)
            return 0;
        else
            return std::exp(-row) * (30 * std::exp(1 / (temp / 25.0 - 1.0)));
    };

    auto ZeroFunc = [](mainReal row, mainReal ex, mainReal why) -> mainReal { return 0; };
    auto constantFuncAnalytic = [](mainReal row, mainReal ex, mainReal why) -> mainReal { return 2 * pi; };
    auto linearFunc = [](mainReal row, mainReal ex, mainReal why) -> mainReal { return std::max(0.0, 5 - std::abs(ex) - std::abs(why)); };
    auto testfunc1 = [A, Normalizer](mainReal row, mainReal ex, mainReal why) -> mainReal { return Normalizer * exp(-A * (ex * ex + why * why)); };
    auto testfunc1_analytic = [A, Normalizer](mainReal row, mainReal ex, mainReal why) -> mainReal {
        return Normalizer * exp(-A * (ex * ex + why * why + row * row)) * boost::math::cyl_bessel_i(0, 2 * A * row * std::sqrt(ex * ex + why * why));
    };
    auto testfunc2 = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal { return Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal { return Normalizer * exp(-B * row * row) * exp(-A * (ex * ex + why * why + row * row)) * boost::math::cyl_bessel_i(0, 2 * A * row * std::sqrt(ex * ex + why * why)); };
    auto testfunc2_analytic_dx = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal { return -2 * A * ex * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic_dy = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal { return -2 * A * why * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic_dx_dy = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal { return 4 * A * A * ex * why * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };

    rhoset = LinearSpacedArray<mainReal>(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray<mainReal>(g.xmin, g.xmax, xcount);
    yset = LinearSpacedArray<mainReal>(g.ymin, g.ymax, ycount);

    GyroAveragingGrid<rhocount, xcount, ycount> grid(rhoset, xset, yset);
    errorAnalysis(g, testfunc2, testfunc2_analytic);
    grid.GyroAveragingTestSuite(testfunc2, testfunc2_analytic);
    derivTest(g, testfunc2, testfunc2_analytic_dx, testfunc2_analytic_dy, testfunc2_analytic_dx_dy);
    interpAnalysis(g, testfunc2, testfunc2_analytic);
    testInterpImprovement();
    testArcIntegralBicubic();
}

template <class RealT = double>
void testArcIntegralBicubic() {
    constexpr RealT r = 0.3, s0 = 0.6, s1 = 2.2, xc = -0.2, yc = -1.4;
    std::array<RealT, 16> coeffs;
    arcIntegralBicubic(coeffs, r, xc, yc, s0, s1);
    for (int i = 0; i <= 3; ++i)
        for (int j = 0; j <= 3; ++j) {
            auto f = [i, j](RealT x) -> RealT { return std::pow(xc + r * std::sin(x), i) * std::pow(yc - r * std::cos(x), j); };
            RealT res = TanhSinhIntegrate(s0, s1, f);
            std::cout << i << "\t" << j << "\t" << res << "\t" << coeffs[j * 4 + i] << "\t" << res - coeffs[j * 4 + i] << "\n";
        }
}

template <int count, typename TFunc1, typename TFunc2, class RealT>
void interpAnalysisInnerLoop(const gridDomain<RealT> &g, TFunc1 f,
                             TFunc2 analytic) {
    constexpr int rhocount = 4; //TODO MAGIC NUMBER SHOULD BE PASSED IN
    std::vector<RealT> rhoset;
    std::vector<RealT> xset;
    std::vector<RealT> yset;
    rhoset = LinearSpacedArray<RealT>(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, count);
    yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, count);
    GyroAveragingGrid<rhocount, count, count> grid(rhoset, xset, yset);
    grid.InterpErrorAnalysis(f, analytic);
}

template <int count, typename TFunc1, typename TFunc2, class RealT>
void errorAnalysisInnerLoop(const gridDomain<RealT> &g, TFunc1 f,
                            TFunc2 analytic) {
    constexpr int rhocount = 1; //TODO MAGIC NUMBER SHOULD BE PASSED IN
    std::vector<RealT> rhoset;
    std::vector<RealT> xset;
    std::vector<RealT> yset;
    rhoset.push_back(g.rhomax);
    xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, count);
    yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, count);
    GyroAveragingGrid<rhocount, count, count,RealT> grid(rhoset, xset, yset);
    grid.compactErrorAnalysis(f, analytic);
}

template <typename TFunc1, typename TFunc2, class RealT>
void interpAnalysis(const gridDomain<RealT> &g, TFunc1 f,
                    TFunc2 analytic) {

    constexpr int counts[] = {6, 12, 24, 48, 96, 192};
    interpAnalysisInnerLoop<counts[0]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[1]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[2]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[3]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[4]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[5]>(g, f, analytic);
}

template <typename TFunc1, typename TFunc2, class RealT>
void errorAnalysis(const gridDomain<RealT> &g, TFunc1 f,
                   TFunc2 analytic) {

  constexpr int counts[] = {6, 12, 24, 48, 96, 192, 384, 768}; //we skip the last one, it's too big/slow.  
    std::cout << "        gridN           Input Grid      Analytic Estimate               Linear Interp                   Bicubic Interp                 Lin rel err      Bicubic Rel err\n";
    errorAnalysisInnerLoop<counts[0]>(g, f, analytic);
    errorAnalysisInnerLoop<counts[1]>(g, f, analytic);
    errorAnalysisInnerLoop<counts[2]>(g, f, analytic);
    errorAnalysisInnerLoop<counts[3]>(g, f, analytic);
    errorAnalysisInnerLoop<counts[4]>(g, f, analytic);
    //errorAnalysisInnerLoop<counts[5]>(g, f, analytic);
    //errorAnalysisInnerLoop<counts[6]>(g, f, analytic);
}

template <typename TFunc1, typename TFunc2, typename TFunc3, typename TFunc4, class RealT>
void derivTest(const gridDomain <RealT> &g, TFunc1 f,
               TFunc2 f_x, TFunc3 f_y, TFunc4 f_xy) {

    constexpr int count = 36;
    constexpr int rhocount = 4;
    std::vector<RealT> rhoset;
    std::vector<RealT> xset;
    std::vector<RealT> yset;
    rhoset = LinearSpacedArray(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray(g.xmin, g.xmax, count);
    yset = LinearSpacedArray(g.ymin, g.ymax, count);
    GyroAveragingGrid<rhocount, count, count> grid(rhoset, xset, yset);
    grid.derivsErrorAnalysis(f, f_x, f_y, f_xy);
}


double inline DCTBasisFunction(double p, double q, int i, int j, int N) {
    
    double a=2,b=2;
    if (p==0) a=1;
    if (q==0) b=1;
    return std::sqrt(a / N) * std::sqrt(b / N) * std::cos(pi * (2 * i + 1) * p / (2.0 * N)) * std::cos(pi * (2 * j + 1) * q / (2.0 * N));
}

void inline arcIntegralBicubic(std::array<double, 16> &coeffs, //this function is being left in double, intentionally, for now
                               double rho, double xc, double yc, double s0, double s1) {
    //we are going to write down indefinite integrals of (xc + rho*sin(x))^i (xc-rho*cos(x))^j
    // it seems like in c++, we can't make a 2d-array of lambdas that bind (rho, xc, yc)
    //without paying a performance penalty.
    //The below is not meant to be so human readable.  It was generated partly by mathematica.

    //crazy late-night thought - can this whole thing be constexpr?
    const double c = xc;
    const double d = yc;
    const double r = rho;
    const double r2 = rho * rho;
    const double r3 = r2 * rho;
    const double r4 = r2 * r2;
    const double r5 = r3 * r2;

    const double c2 = c * c;
    const double c3 = c2 * c;

    const double d2 = d * d;
    const double d3 = d2 * d;

    using std::cos;
    using std::sin;

#define f00(x) (x)

#define f10(x) (c * x - r * cos(x))

#define f20(x) (c2 * x - 2 * c * r * cos(x) + r2 * x / 2 - r2 * sin(2 * x) / 4)

#define f30(x) ((1.0 / 12.0) * (3 * c * (4 * c2 * x + 6 * r2 * x - 3 * r2 * sin(2 * x)) - 9 * r * (4 * c2 + r2) * cos(x) + r3 * cos(3 * x)))

#define f01(x) (d * x - r * sin(x))

#define f11(x) (c * d * x - c * r * sin(x) - d * r * cos(x) + r2 * cos(x) * cos(x) / 2.0)

#define f21(x) ((1.0 / 12.0) * (12.0 * c2 * d * x - 12 * c2 * r * sin(x) - 24 * c * d * r * cos(x) + 6 * c * r2 * cos(2 * x) + 6 * d * r2 * x - 3 * d * r2 * sin(2 * x) - 3 * r3 * sin(x) + r3 * sin(3 * x)))

#define f31(x) ((1.0 / 96.0) * (48 * c * d * x * (2 * c2 + 3 * r2) - 72 * d * r * (4 * c2 + r2) * cos(x) - 24 * c * r * (4 * c2 + 3 * r2) * sin(x) + 12 * r2 * (6 * c2 + r2) * cos(2 * x) - 72 * c * d * r2 * sin(2 * x) + 24 * c * r3 * sin(3 * x) + 8 * d * r3 * cos(3 * x) - 3 * r4 * cos(4 * x)))

#define f02(x) ((1.0 / 2.0) * (x * (2 * d2 + r2) + r * sin(x) * (r * cos(x) - 4 * d)))

#define f12(x) ((1.0 / 12.0) * (6 * c * x * (2 * d2 + r2) - 24 * c * d * r * sin(x) + 3 * c * r2 * sin(2 * x) - 3 * r * (4 * d2 + r2) * cos(x) + 6 * d * r2 * cos(2 * x) + r3 * (-1.0 * cos(3 * x))))

#define f22(x) ((1.0 / 96.0) * (24 * r2 * (c2 - d2) * sin(2 * x) + 12 * x * (4 * c2 * (2 * d2 + r2) + 4 * d2 * r2 + r4) - 48 * d * r * (4 * c2 + r2) * sin(x) - 48 * c * r * (4 * d2 + r2) * cos(x) + 96 * c * d * r2 * cos(2 * x) - 16 * c * r3 * cos(3 * x) + 16 * d * r3 * sin(3 * x) - 3 * r4 * sin(4 * x)))

#define f32(x) ((1.0 / 480.0) * (120 * c * r2 * (c2 - 3 * d2) * sin(2 * x) + 60 * c * x * (4 * c2 * (2 * d2 + r2) + 3 * (4 * d2 * r2 + r4)) - 60 * r * cos(x) * (6 * c2 * (4 * d2 + r2) + 6 * d2 * r2 + r4) - 10 * r3 * cos(3 * x) * (12 * c2 - 4 * d2 + r2) - 240 * c * d * r * (4 * c2 + 3 * r2) * sin(x) + 120 * d * r2 * (6 * c2 + r2) * cos(2 * x) + 240 * c * d * r3 * sin(3 * x) - 45 * c * r4 * sin(4 * x) - 30 * d * r4 * cos(4 * x) + 6 * r5 * cos(5 * x)))

#define f03(x) ((1.0 / 12.0) * (12 * d3 * x - 9 * r * (4 * d2 + r2) * sin(x) + 18 * d * r2 * x + 9 * d * r2 * sin(2 * x) - r3 * sin(3 * x)))

#define f13(x) ((1.0 / 96.0) * (48 * c * d * x * (2 * d2 + 3 * r2) - 72 * c * r * (4 * d2 + r2) * sin(x) + 72 * c * d * r2 * sin(2 * x) - 8 * c * r3 * sin(3 * x) + 12 * r2 * (6 * d2 + r2) * cos(2 * x) - 24 * d * r * (4 * d2 + 3 * r2) * cos(x) - 24 * d * r3 * cos(3 * x) + 3 * r4 * cos(4 * x)))

#define f23(x) ((1.0 / 480.0) * (480 * c2 * d3 * x - 1440 * c2 * d2 * r * sin(x) + 720 * c2 * d * r2 * x + 360 * c2 * d * r2 * sin(2 * x) - 360 * c2 * r3 * sin(x) - 40 * c2 * r3 * sin(3 * x) + 120 * c * r2 * (6 * d2 + r2) * cos(2 * x) - 240 * c * d * r * (4 * d2 + 3 * r2) * cos(x) - 240 * c * d * r3 * cos(3 * x) + 30 * c * r4 * cos(4 * x) + 240 * d3 * r2 * x - 120 * d3 * r2 * sin(2 * x) - 360 * d2 * r3 * sin(x) + 120 * d2 * r3 * sin(3 * x) + 180 * d * r4 * x - 45 * d * r4 * sin(4 * x) - 60 * r5 * sin(x) + 10 * r5 * sin(3 * x) + 6 * r5 * sin(5 * x)))

#define f33(x) ((1.0 / 960.0) * (960 * c3 * d3 * x - 2880 * c3 * d2 * r * sin(x) + 1440 * c3 * d * r2 * x + 720 * c3 * d * r2 * sin(2 * x) - 720 * c3 * r3 * sin(x) - 80 * c3 * r3 * sin(3 * x) + 45 * r2 * cos(2 * x) * (8 * c2 * (6 * d2 + r2) + 8 * d2 * r2 + r4) - 360 * d * r * cos(x) * (c2 * (8 * d2 + 6 * r2) + 2 * d2 * r2 + r4) - 720 * c2 * d * r3 * cos(3 * x) + 90 * c2 * r4 * cos(4 * x) + 1440 * c * d3 * r2 * x - 720 * c * d3 * r2 * sin(2 * x) - 2160 * c * d2 * r3 * sin(x) + 720 * c * d2 * r3 * sin(3 * x) + 1080 * c * d * r4 * x - 270 * c * d * r4 * sin(4 * x) - 360 * c * r5 * sin(x) + 60 * c * r5 * sin(3 * x) + 36 * c * r5 * sin(5 * x) + 80 * d3 * r3 * cos(3 * x) - 90 * d2 * r4 * cos(4 * x) - 60 * d * r5 * cos(3 * x) + 36 * d * r5 * cos(5 * x) - 5 * r5 * r * cos(6 * x)))

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
}


void testInterpImprovement() {

    typedef double localReal;

    gridDomain<localReal> g;
    g.rhomax = 0.9;
    g.rhomin = 0;
    g.xmin = g.ymin = -3;
    g.xmax = g.ymax = 3;
    constexpr int xcount = 30, ycount = 30;
    constexpr int xcountb = xcount * 2;
    constexpr int ycountb = xcount * 2;
    constexpr localReal A = 2;
    constexpr localReal B = 2;
    constexpr localReal Normalizer = 50.0;

    constexpr int rhocount = 4; //TODO MAGIC NUMBER SHOULD BE PASSED IN
    std::vector<localReal> rhoset;
    std::vector<localReal> xset, xsetb;
    std::vector<localReal> yset, ysetb;
    rhoset = LinearSpacedArray<localReal>(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray<localReal>(g.xmin, g.xmax, xcount);
    yset = LinearSpacedArray<localReal>(g.ymin, g.ymax, ycount);

    xsetb = LinearSpacedArray<localReal>(g.xmin, g.xmax, xcountb);
    ysetb = LinearSpacedArray<localReal>(g.ymin, g.ymax, ycountb);

    GyroAveragingGrid<rhocount, xcount, ycount> smallgrid(rhoset, xset, yset);
    GyroAveragingGrid<rhocount, xcountb, ycountb> biggrid(rhoset, xsetb, ysetb);
    //auto testfunc2 = [Normalizer, A, B](RealT row, RealT ex, RealT why) -> RealT { return (1) * row; };

    auto testfunc2 = [Normalizer, A, B](localReal row, localReal ex, localReal why) -> localReal { return Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic = [Normalizer, A, B](localReal row, localReal ex, localReal why) -> localReal { return Normalizer * exp(-B * row * row) * exp(-A * (ex * ex + why * why + row * row)) * boost::math::cyl_bessel_i(0, 2 * A * row * std::sqrt(ex * ex + why * why)); };
    auto testfunc2_analytic_dx = [Normalizer, A, B](localReal row, localReal ex, localReal why) -> localReal { return -2 * A * ex * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic_dy = [Normalizer, A, B](localReal row, localReal ex, localReal why) -> localReal { return -2 * A * why * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic_dx_dy = [Normalizer, A, B](localReal row, localReal ex, localReal why) -> localReal { return 4 * A * A * ex * why * Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };

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




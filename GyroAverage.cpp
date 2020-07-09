
//#ifndef NDEBUG
//#define NDEBUG
//#endif

#define NDEBUG

#define VIENNACL_WITH_UBLAS 1
#define VIENNACL_HAVE_EIGEN 1

#if (defined __GNUC__) && (__GNUC__ > 4 || __GNUC_MINOR__ >= 7)
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif

#ifdef INCL_MATH_CONSTANTS
#include <math_constants.h>
#endif

#include <algorithm>
#include <array>
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
//#include <boost/math/differentiaton/finite_difference.hpp>  //this needs a
// fairly recent version of boost.  boo.
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/math/special_functions/bessel.hpp>
//#include <boost/timer/timer.hpp>
#include <fftw3.h>
#include <math.h>
#include <omp.h>

#include <cassert>
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <vector>

#include "ga.h"

constexpr int inline mymax(int a, int b) {
    if (a > b)
        return a;
    else
        return b;
}

struct resultsRecord {
    OOGA::calculatorType type;
    int N;
    std ::vector<double> error;
    std::vector<double> rhoset;
    double initTime;
    double calcTime;
    int bits;
    friend std::ostream& operator<<(std::ostream& output, const resultsRecord& r) {
        auto nameMap = OOGA::calculatorNameMap();
        output << nameMap[r.type] << ";"
               << r.N << ";"
               << r.initTime << ";"
               << r.calcTime << ";"

               << r.bits << ";";
        for (auto e : r.error)
            output << " " << e;
        output << ";";

        return output;
    }
};

template <int N, int rhocount, class RealT, typename TFunc1>
std::vector<resultsRecord> testRun(const std::vector<OOGA::calculatorType>& calclist, TFunc1 testfunc, OOGA::gridDomain<RealT>& g) {
    using namespace OOGA;
    std::vector<resultsRecord> runResults;
    constexpr int xcount = N;
    constexpr int ycount = N;
    std::vector<RealT> rhoset;
    std::vector<RealT> xset;
    std::vector<RealT> yset;

    rhoset = LinearSpacedArray<RealT>(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, xcount);
    yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, ycount);

    functionGrid<rhocount, xcount, ycount> f(rhoset, xset, yset),
        exact(rhoset, xset, yset), result(rhoset, xset, yset);

    std::vector<std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>> calcset;
    exact.fillAlmostExactGA(testfunc);
    for (auto i = 0; i < calclist.size(); ++i) {
        auto func = [&](int j) -> void {
            calcset.emplace_back(GACalculator<rhocount, xcount, ycount, RealT, N>::Factory::newCalculator(calclist[j], g, exact));  //we are using precisely 2x padding for the padcount.
        };
        double initTime = measure<std::chrono::milliseconds>::execution(func, i);  // TODO:  try to use a more accurate padcount
        f.fill(testfunc);
        auto& b = f;
        auto func2 = [&](void) -> void {
            calcset.back()->calculate(b);
        };

        double calcTime = measure<std::chrono::nanoseconds>::execution2(func2);
        result = (calcset.back()->calculate(f));
        resultsRecord r;
        r.type = calclist[i];
        r.N = N;
        r.rhoset = rhoset;
        r.initTime = initTime;
        r.calcTime = calcTime;
        r.bits = sizeof(RealT);
        r.error = rhoset;
        for (int k = 0; k < rhocount; ++k)
            r.error[k] = exact.maxNormDiff(result.gridValues, k) / exact.maxNorm(k);

        runResults.push_back(r);
        calcset.back().reset();
    }
    return runResults;
}

int main() {
    //fft_testing();
    using namespace OOGA;
    typedef double mainReal;
    constexpr mainReal mainRhoMin = 0.25;
    constexpr mainReal mainRhoMax = 3;
    constexpr mainReal mainxyMin = -8;
    constexpr mainReal mainxyMax = 8;
    gridDomain<mainReal> g;
    g.rhomax = mainRhoMax;
    g.rhomin = mainRhoMin;
    g.xmin = g.ymin = mainxyMin;
    g.xmax = g.ymax = mainxyMax;
    constexpr int xcount = 16, ycount = 16,
                  rhocount = 3;  // bump up to 64x64x35 later or 128x128x35
    constexpr mainReal A = 2;
    constexpr mainReal B = 2;
    constexpr mainReal Normalizer = 50.0;
    std::vector<mainReal> rhoset;
    std::vector<mainReal> xset;
    std::vector<mainReal> yset;
    auto nameMap = OOGA::calculatorNameMap();

    rhoset = LinearSpacedArray<mainReal>(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray<mainReal>(g.xmin, g.xmax, xcount);
    yset = LinearSpacedArray<mainReal>(g.ymin, g.ymax, ycount);

    functionGrid<rhocount, xcount, ycount> f(rhoset, xset, yset),
        exact(rhoset, xset, yset);

    std::vector<std::unique_ptr<GACalculator<rhocount, xcount, ycount, mainReal>>> calcset;
    std::vector<OOGA::calculatorType> calclist;
    calclist.push_back(OOGA::calculatorType::linearCPU);
    calclist.push_back(OOGA::calculatorType::linearDotProductCPU);
    calclist.push_back(OOGA::calculatorType::linearDotProductGPU);
    calclist.push_back(OOGA::calculatorType::bicubicCPU);
    calclist.push_back(OOGA::calculatorType::bicubicDotProductCPU);
    calclist.push_back(OOGA::calculatorType::bicubicDotProductGPU);
    calclist.push_back(OOGA::calculatorType::DCTCPUCalculator2);
    calclist.push_back(OOGA::calculatorType::DCTCPUPaddedCalculator2);
    //constexpr mainReal padtest = xcount * mainRhoMax / std::abs(mainxyMax - mainxyMin);
    //constexpr int padcount = mymax(8, 4 + static_cast<int>(std::ceil(padtest)));

    /*for (auto i = 0; i < calclist.size(); ++i) {
        std::cout << "Method ";
        auto func = [&](int j) -> void {
            calcset.emplace_back(GACalculator<rhocount, xcount, ycount, mainReal, 12>::Factory::newCalculator(calclist[j], g, exact));
        };
        std::cout << i << " took:";
        std::cout << measure<std::chrono::milliseconds>::execution(func, i);
        std::cout << " milliseconds to initialize" << std::endl;
      }*/

    //std::cout << "Done initializing." << std::endl;
    auto testfunc2 = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal {
        return Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row);
    };
    auto testfunc2_analytic = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal {
        return Normalizer * exp(-B * row * row) * exp(-A * (ex * ex + why * why + row * row)) *
               boost::math::cyl_bessel_i(0, 2 * A * row * std::sqrt(ex * ex + why * why));
    };

    auto res = testRun<16, rhocount, double>(calclist, testfunc2, g);
    //const int p = 3, q = 6;
    /*auto basistest = [p, q, g, xcount, ycount](mainReal row, mainReal ex, mainReal why) -> mainReal {
        mainReal xint = (ex - g.xmin) / (g.xmax - g.xmin) * (xcount - 1);
        mainReal yint = (why - g.ymin) / (g.ymax - g.ymin) * (ycount - 1);
        return DCTBasisFunction2(p, q, xint, yint, xcount);
    };*/

    /*for (int p = 0; p < xcount; p++)
        for (int q = 0; q < ycount; q++) {
            auto basistest = [p, q, g, xcount, ycount](mainReal row, mainReal ex, mainReal why) -> mainReal {
                mainReal xint = (ex - g.xmin) / (g.xmax - g.xmin) * xcount;
                mainReal yint = (why - g.ymin) / (g.ymax - g.ymin) * ycount;
                return DCTBasisFunction2(p, q, xint, yint, xcount);
            };*/

    /*std::vector<functionGrid<rhocount, xcount, ycount, mainReal>> results;
    exact.fill(testfunc2_analytic);  //this is not the analytic function.
    results.push_back(exact);
    f.fill(testfunc2);
    int counter = 0;
    for (auto& c : calcset) {
        std::cout << "Method ";
        std::cout << counter << " took:";
        auto& b = f;
        auto func = [&](void) -> void {
            c->calculate(b);
        };

        std::cout << measure<std::chrono::nanoseconds>::execution2(func);
        std::cout << " nanoseconds to calculate." << std::endl;
        results.push_back(c->calculate(f));
        counter++;
    }
    int s = results.size();
    Eigen::MatrixXd maxnorm(s, s), rms(s, s);
    for (int i = 0; i < s; i++)
        for (int j = 0; j < s; j++) {
            maxnorm(i, j) = results[i].maxNormDiff(results[j].gridValues, 29);
            rms(i, j) = results[i].RMSNormDiff(results[j].gridValues, 29);
        }
    std::cout << maxnorm << std::endl
              << std::endl
              << std::endl
              << rms << std::endl
              << std::endl;*/

    auto res1 = testRun<8, rhocount, double>(calclist, testfunc2, g);
    for (auto r : res1)
        std::cout << r << std::endl;

    res1 = testRun<16, rhocount, double>(calclist, testfunc2, g);
    for (auto r : res1)
        std::cout << r << std::endl;

    res1 = testRun<32, rhocount, double>(calclist, testfunc2, g);
    for (auto r : res1)
        std::cout << r << std::endl;

    res1 = testRun<64, rhocount, double>(calclist, testfunc2, g);
    for (auto r : res1)
        std::cout << r << std::endl;

    res1 = testRun<128, rhocount, double>(calclist, testfunc2, g);
    for (auto r : res1)
        std::cout << r << std::endl;

    res1 = testRun<256, rhocount, double>(calclist, testfunc2, g);
    for (auto r : res1)
        std::cout << r << std::endl;
}

void fft_testing() {
    using namespace OOGA;
    typedef double mainReal;

    gridDomain<mainReal> g;
    g.rhomax = 0.25;
    g.rhomin = 1.55;
    g.xmin = g.ymin = -2;
    g.xmax = g.ymax = 2;
    constexpr int xcount = 4, ycount = 4,
                  rhocount = 3;  // bump up to 64x64x35 later or 128x128x35

    std::vector<mainReal> rhoset;  //TODO we should write a function to initialize a functiongrid from a gridDomain.
    std::vector<mainReal> xset;
    std::vector<mainReal> yset;

    rhoset = LinearSpacedArray<mainReal>(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray<mainReal>(g.xmin, g.xmax, xcount);
    yset = LinearSpacedArray<mainReal>(g.ymin, g.ymax, ycount);
    for (int p = 0; p < xcount; ++p)
        for (int q = 0; q < ycount; ++q) {
            if (p > 3 || q > 3)
                continue;
            auto basistest = [p, q, g, xcount, ycount](mainReal row, mainReal ex, mainReal why) -> mainReal {
                mainReal xint = (ex - g.xmin) / (g.xmax - g.xmin) * (xcount - 1);
                mainReal yint = (why - g.ymin) / (g.ymax - g.ymin) * (ycount - 1);
                mainReal dr = DCTBasisFunction2(p, q, xint, yint, xcount);
                /* std::cout << "row: " << row
                          << "ex: " << ex
                          << "why: " << why
                          << "xint: " << xint
                          << "yint: " << yint
                          << "p: " << p
                          << "q: " << q
                          << "N: " << xcount
                          << "dr: " << dr << std::endl;*/
                return dr;
            };

            /*auto basistest2 = [p, q, g, xcount, ycount](int row, int ex, int why) -> mainReal {
                mainReal xint = (ex - g.xmin) / (g.xmax - g.xmin) * xcount;
                mainReal yint = (why - g.ymin) / (g.ymax - g.ymin) * ycount;
                return DCTBasisFunction2(p, q, xint, yint, xcount);
            };*/

            functionGrid<rhocount, xcount, ycount> in(rhoset, xset, yset), in2(rhoset, xset, yset), out(rhoset, xset, yset);
            //in2.fill(basistest);
            in2.clearGrid();
            in2.gridValues(0, p, q) = 1;
            fftw_plan plan;
            //plan = fftw_plan_r2r_2d(xcount, ycount, in2.gridValues.data.data(), out.gridValues.data.data(), FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);  //FORWARD "DCT"
            plan = fftw_plan_r2r_2d(xcount, ycount, in2.gridValues.data.data(), out.gridValues.data.data(), FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);  //INVERSE "DCT"
            fftw_execute(plan);
            in2.csvPrinter(0);
            std::cout << std::endl;
            out.csvPrinter(0);
            std::cout << std::endl;
            in.fill(basistest);
            in.csvPrinter(0);
            std::cout << std::endl;

            /*for (int i = 0; i < xcount; ++i) for (int j = 0; j < ycount; ++j) {
                if (std::abs(out.gridValues(0, i, j) > 0.01))
                    std::cout << p << " " << q << " " << i << " " << j << " " << out.gridValues(0, i, j) << std::endl;
            }
            */
        }
    std::cout << std::endl;
}

namespace OOGA {

template <class RealT = double>
void testArcIntegralBicubic() {
    constexpr RealT r = 0.3, s0 = 0.6, s1 = 2.2, xc = -0.2, yc = -1.4;
    std::array<RealT, 16> coeffs;
    arcIntegralBicubic(coeffs, r, xc, yc, s0, s1);
    for (int i = 0; i <= 3; ++i)
        for (int j = 0; j <= 3; ++j) {
            auto f = [i, j](RealT x) -> RealT {
                return std::pow(xc + r * std::sin(x), i) *
                       std::pow(yc - r * std::cos(x), j);
            };
            RealT res = TanhSinhIntegrate(s0, s1, f);
            std::cout << i << "\t" << j << "\t" << res << "\t"
                      << coeffs[j * 4 + i] << "\t" << res - coeffs[j * 4 + i]
                      << "\n";
        }
}

void inline arcIntegralBicubic(
    std::array<double, 16>& coeffs,  // this function is being left in double,
                                     // intentionally, for now
    double rho, double xc, double yc, double s0, double s1) {
    // we are going to write down indefinite integrals of (xc + rho*sin(x))^i
    // (xc-rho*cos(x))^j
    // it seems like in c++, we can't make a 2d-array of lambdas that bind (rho,
    // xc, yc)
    // without paying a performance penalty.
    // The below is not meant to be so human readable.  It was generated partly
    // by mathematica.

    // crazy late-night thought - can this whole thing be constexpr?
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

#define f30(x)                                                                 \
    ((1.0 / 12.0) * (3 * c * (4 * c2 * x + 6 * r2 * x - 3 * r2 * sin(2 * x)) - \
                     9 * r * (4 * c2 + r2) * cos(x) + r3 * cos(3 * x)))

#define f01(x) (d * x - r * sin(x))

#define f11(x) \
    (c * d * x - c * r * sin(x) - d * r * cos(x) + r2 * cos(x) * cos(x) / 2.0)

#define f21(x)                                                             \
    ((1.0 / 12.0) *                                                        \
     (12.0 * c2 * d * x - 12 * c2 * r * sin(x) - 24 * c * d * r * cos(x) + \
      6 * c * r2 * cos(2 * x) + 6 * d * r2 * x - 3 * d * r2 * sin(2 * x) - \
      3 * r3 * sin(x) + r3 * sin(3 * x)))

#define f31(x)                                                                 \
    ((1.0 / 96.0) * (48 * c * d * x * (2 * c2 + 3 * r2) -                      \
                     72 * d * r * (4 * c2 + r2) * cos(x) -                     \
                     24 * c * r * (4 * c2 + 3 * r2) * sin(x) +                 \
                     12 * r2 * (6 * c2 + r2) * cos(2 * x) -                    \
                     72 * c * d * r2 * sin(2 * x) + 24 * c * r3 * sin(3 * x) + \
                     8 * d * r3 * cos(3 * x) - 3 * r4 * cos(4 * x)))

#define f02(x) \
    ((1.0 / 2.0) * (x * (2 * d2 + r2) + r * sin(x) * (r * cos(x) - 4 * d)))

#define f12(x)                                                   \
    ((1.0 / 12.0) *                                              \
     (6 * c * x * (2 * d2 + r2) - 24 * c * d * r * sin(x) +      \
      3 * c * r2 * sin(2 * x) - 3 * r * (4 * d2 + r2) * cos(x) + \
      6 * d * r2 * cos(2 * x) + r3 * (-1.0 * cos(3 * x))))

#define f22(x)                                                                 \
    ((1.0 / 96.0) * (24 * r2 * (c2 - d2) * sin(2 * x) +                        \
                     12 * x * (4 * c2 * (2 * d2 + r2) + 4 * d2 * r2 + r4) -    \
                     48 * d * r * (4 * c2 + r2) * sin(x) -                     \
                     48 * c * r * (4 * d2 + r2) * cos(x) +                     \
                     96 * c * d * r2 * cos(2 * x) - 16 * c * r3 * cos(3 * x) + \
                     16 * d * r3 * sin(3 * x) - 3 * r4 * sin(4 * x)))

#define f32(x)                                                         \
    ((1.0 / 480.0) *                                                   \
     (120 * c * r2 * (c2 - 3 * d2) * sin(2 * x) +                      \
      60 * c * x * (4 * c2 * (2 * d2 + r2) + 3 * (4 * d2 * r2 + r4)) - \
      60 * r * cos(x) * (6 * c2 * (4 * d2 + r2) + 6 * d2 * r2 + r4) -  \
      10 * r3 * cos(3 * x) * (12 * c2 - 4 * d2 + r2) -                 \
      240 * c * d * r * (4 * c2 + 3 * r2) * sin(x) +                   \
      120 * d * r2 * (6 * c2 + r2) * cos(2 * x) +                      \
      240 * c * d * r3 * sin(3 * x) - 45 * c * r4 * sin(4 * x) -       \
      30 * d * r4 * cos(4 * x) + 6 * r5 * cos(5 * x)))

#define f03(x)                                                         \
    ((1.0 / 12.0) *                                                    \
     (12 * d3 * x - 9 * r * (4 * d2 + r2) * sin(x) + 18 * d * r2 * x + \
      9 * d * r2 * sin(2 * x) - r3 * sin(3 * x)))

#define f13(x)                                                             \
    ((1.0 / 96.0) *                                                        \
     (48 * c * d * x * (2 * d2 + 3 * r2) -                                 \
      72 * c * r * (4 * d2 + r2) * sin(x) + 72 * c * d * r2 * sin(2 * x) - \
      8 * c * r3 * sin(3 * x) + 12 * r2 * (6 * d2 + r2) * cos(2 * x) -     \
      24 * d * r * (4 * d2 + 3 * r2) * cos(x) - 24 * d * r3 * cos(3 * x) + \
      3 * r4 * cos(4 * x)))

#define f23(x)                                                                 \
    ((1.0 / 480.0) *                                                           \
     (480 * c2 * d3 * x - 1440 * c2 * d2 * r * sin(x) +                        \
      720 * c2 * d * r2 * x + 360 * c2 * d * r2 * sin(2 * x) -                 \
      360 * c2 * r3 * sin(x) - 40 * c2 * r3 * sin(3 * x) +                     \
      120 * c * r2 * (6 * d2 + r2) * cos(2 * x) -                              \
      240 * c * d * r * (4 * d2 + 3 * r2) * cos(x) -                           \
      240 * c * d * r3 * cos(3 * x) + 30 * c * r4 * cos(4 * x) +               \
      240 * d3 * r2 * x - 120 * d3 * r2 * sin(2 * x) -                         \
      360 * d2 * r3 * sin(x) + 120 * d2 * r3 * sin(3 * x) + 180 * d * r4 * x - \
      45 * d * r4 * sin(4 * x) - 60 * r5 * sin(x) + 10 * r5 * sin(3 * x) +     \
      6 * r5 * sin(5 * x)))

#define f33(x)                                                             \
    ((1.0 / 960.0) *                                                       \
     (960 * c3 * d3 * x - 2880 * c3 * d2 * r * sin(x) +                    \
      1440 * c3 * d * r2 * x + 720 * c3 * d * r2 * sin(2 * x) -            \
      720 * c3 * r3 * sin(x) - 80 * c3 * r3 * sin(3 * x) +                 \
      45 * r2 * cos(2 * x) * (8 * c2 * (6 * d2 + r2) + 8 * d2 * r2 + r4) - \
      360 * d * r * cos(x) * (c2 * (8 * d2 + 6 * r2) + 2 * d2 * r2 + r4) - \
      720 * c2 * d * r3 * cos(3 * x) + 90 * c2 * r4 * cos(4 * x) +         \
      1440 * c * d3 * r2 * x - 720 * c * d3 * r2 * sin(2 * x) -            \
      2160 * c * d2 * r3 * sin(x) + 720 * c * d2 * r3 * sin(3 * x) +       \
      1080 * c * d * r4 * x - 270 * c * d * r4 * sin(4 * x) -              \
      360 * c * r5 * sin(x) + 60 * c * r5 * sin(3 * x) +                   \
      36 * c * r5 * sin(5 * x) + 80 * d3 * r3 * cos(3 * x) -               \
      90 * d2 * r4 * cos(4 * x) - 60 * d * r5 * cos(3 * x) +               \
      36 * d * r5 * cos(5 * x) - 5 * r5 * r * cos(6 * x)))

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

}  // namespace OOGA

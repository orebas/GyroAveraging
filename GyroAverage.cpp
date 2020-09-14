
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

/*#include <algorithm>
#include <array>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

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

#include <omp.h>

//#include <cassert>
//#include <cmath>
//#include <iomanip>
//#include <iostream>
//#include <iterator>
//#include <limits>
#include <map>
#include <vector>
*/
#include "ga.h"

template <class RealT = double>
struct resultsRecord {
    OOGA::calculatorType type = OOGA::calculatorType::linearCPU;
    int N = 0;
    std::vector<RealT> rhoset;
    double initTime = 0;
    double calcTime = 0;
    int bits = 0;
    std ::vector<RealT> error;
    friend std::ostream& operator<<(std::ostream& output, const resultsRecord<RealT>& r) {
        auto nameMap = OOGA::calculatorNameMap();
        output << nameMap[r.type] << ","
               << r.N << ","
               << r.initTime / 1000 << ","
               << r.calcTime << ","
               << 1e9 / r.calcTime << ","
               << r.bits << ", "
               << *std::max_element(r.error.begin(), r.error.end()) << ",";
        for (auto e : r.error) {
            output << "," << e;
        }
        output << ",";
        return output;
    }
    //std::string header() {
    //}
    resultsRecord(OOGA::calculatorType t_i, int N_i, std ::vector<RealT> rhoset_i, double initTime_i, double calcTime_i, int bits_i)
        : type(t_i), N(N_i), rhoset(rhoset_i), initTime(initTime_i), calcTime(calcTime_i), bits(bits_i), error(rhoset_i) {
    }
};

template <class RealT = double>
std::ostream& operator<<(std::ostream& output, const std::vector<resultsRecord<RealT>>& r) {
    for (const auto& e : r) {
        std::cout << e << std::endl;
    }
    return output;
}

template <int N, int rhocount, class RealT, bool cheb = false, typename TFunc1>
std::vector<resultsRecord<RealT>> testRun(const std::vector<OOGA::calculatorType>& calclist, TFunc1 testfunc, OOGA::gridDomain& g) {
    using OOGA::functionGrid, OOGA::GACalculator, OOGA::gridDomain, OOGA::LinearSpacedArray, OOGA::measure;

    std::vector<resultsRecord<RealT>> runResults;
    constexpr int xcount = N;
    constexpr int ycount = N;
    std::vector<RealT> rhoset;
    std::vector<RealT> xset, lin_xset, lin_yset;
    std::vector<RealT> yset;

    rhoset = LinearSpacedArray<RealT>(g.rhomin, g.rhomax, rhocount);
    if (!cheb) {
        xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, xcount);
        yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, ycount);
    } else {
        xset = chebPoints<RealT>(xcount);
        yset = chebPoints<RealT>(ycount);
    }

    lin_xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, xcount);
    lin_yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, ycount);

    functionGrid<rhocount, xcount, ycount, RealT>
        f(rhoset, xset, yset),
        exact(rhoset, lin_xset, lin_yset), result(rhoset, lin_xset, lin_yset);

    std::vector<std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>> calcset;
    exact.fillTruncatedAlmostExactGA(testfunc);

    for (size_t i = 0; i < calclist.size(); ++i) {
        auto func = [&](int j) -> void { calcset.emplace_back(GACalculator<rhocount, xcount, ycount, RealT, N>::Factory::newCalculator(calclist[j], g, exact)); };
        double initTime = measure<std::chrono::milliseconds>::execution(func, i);
        f.fill(testfunc);
        auto& b = f;
        auto func2 = [&]() -> void {
            calcset.back()->calculate(b);
        };
        double calcTime = measure<std::chrono::nanoseconds>::execution2(func2);
        result = (calcset.back()->calculate(f));
        resultsRecord<RealT> r(calclist[i], N, std::vector<RealT>(rhoset.begin(), rhoset.end()), initTime, calcTime, sizeof(RealT));
        for (int k = 0; k < rhocount; ++k) {
            r.error[k] = exact.maxNormDiff(result.gridValues, k) / exact.maxNorm(k);
        }
        runResults.push_back(r);
        calcset.back().reset(nullptr);
    }
    return runResults;
}

template <int N, int MaxN, int rhocount, class RealT, bool cheb = false, typename TFunc1>
std::vector<resultsRecord<RealT>> testRunRecursive(const std::vector<OOGA::calculatorType>& calclist, TFunc1 testfunc, OOGA::gridDomain& g) {
    bool fail = false;
    std::vector<resultsRecord<RealT>> result, resultNext;  //10 is a magic number.  compile-time though.
    std::vector<resultsRecord<RealT>> fullResult;
    using OOGA::functionGrid, OOGA::GACalculator, OOGA::gridDomain, OOGA::LinearSpacedArray, OOGA::measure;
    if constexpr (N > MaxN) {
        return result;
    } else {
        try {
            std::cout << "running " << N << " " << rhocount << std::endl;
            result = testRun<N, rhocount, RealT, cheb, TFunc1>(calclist, testfunc, g);
            fullResult.insert(fullResult.end(), result.begin(), result.end());
            result = testRun<N + 4, rhocount, RealT, cheb, TFunc1>(calclist, testfunc, g);
            fullResult.insert(fullResult.end(), result.begin(), result.end());
            result = testRun<N + 8, rhocount, RealT, cheb, TFunc1>(calclist, testfunc, g);
            fullResult.insert(fullResult.end(), result.begin(), result.end());
            result = testRun<N + 12, rhocount, RealT, cheb, TFunc1>(calclist, testfunc, g);
            fullResult.insert(fullResult.end(), result.begin(), result.end());
            result = testRun<N + 16, rhocount, RealT, cheb, TFunc1>(calclist, testfunc, g);
            fullResult.insert(fullResult.end(), result.begin(), result.end());
            result = testRun<N + 20, rhocount, RealT, cheb, TFunc1>(calclist, testfunc, g);
            fullResult.insert(fullResult.end(), result.begin(), result.end());
            result = testRun<N + 24, rhocount, RealT, cheb, TFunc1>(calclist, testfunc, g);
            fullResult.insert(fullResult.end(), result.begin(), result.end());
            result = testRun<N + 28, rhocount, RealT, cheb, TFunc1>(calclist, testfunc, g);
            fullResult.insert(fullResult.end(), result.begin(), result.end());
            result = testRun<N + 32, rhocount, RealT, cheb, TFunc1>(calclist, testfunc, g);
            fullResult.insert(fullResult.end(), result.begin(), result.end());
            result = testRun<N + 36, rhocount, RealT, cheb, TFunc1>(calclist, testfunc, g);
            fullResult.insert(fullResult.end(), result.begin(), result.end());

        } catch (std::exception& e) {
            fail = true;
            std::cout << "Exception1" << std::endl;
        }
        if constexpr (N < MaxN) {
            if (!fail)
                try {
                    resultNext = testRunRecursive<N + 40, MaxN, rhocount, RealT, cheb, TFunc1>(calclist, testfunc, g);
                } catch (std::exception& e) {
                    std::cout << "Exception2" << std::endl;
                }
        }
        fullResult.insert(result.end(), resultNext.begin(), resultNext.end());
        return fullResult;
    }
}

int main() {
    //fft_testing();
    //chebDevel();
    //fftw_cleanup();
    //cd return 0;
    //using namespace OOGA;
    using OOGA::chebBasisFunction, OOGA::functionGrid, OOGA::GACalculator, OOGA::gridDomain, OOGA::LinearSpacedArray;
    using mainReal = double;
    constexpr mainReal mainRhoMin = 0.0 / 4.0;   //used to be 0.25/4
    constexpr mainReal mainRhoMax = 3.45 / 4.0;  //used to be 3/4
    constexpr mainReal mainxyMin = -1;
    constexpr mainReal mainxyMax = 1;
    gridDomain g;
    g.rhomax = mainRhoMax;
    g.rhomin = mainRhoMin;
    g.xmin = g.ymin = mainxyMin;
    g.xmax = g.ymax = mainxyMax;
    constexpr int xcount = 16, ycount = 16,
                  rhocount = 20;  // bump up to 64x64x35 later or 128x128x35
    constexpr mainReal A = 24;
    constexpr mainReal B = 1.1;
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

    std::vector<OOGA::calculatorType> chebCalclist;
    chebCalclist.push_back(OOGA::calculatorType::chebCPUDense);
    chebCalclist.push_back(OOGA::calculatorType::chebGPUDense);
    //constexpr mainReal padtest = xcount * mainRhoMax / std::abs(mainxyMax - mainxyMin);
    //constexpr int padcount = mymax(8, 4 + static_cast<int>(std::ceil(padtest)));

    auto testfunc2 = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal {
        return Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row);
    };
    /*auto testfunc2_analytic = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal {
        return Normalizer * exp(-B * row * row) * exp(-A * (ex * ex + why * why + row * row)) *
               boost::math::cyl_bessel_i(0, 2 * A * row * std::sqrt(ex * ex + why * why));
    };*/

    /*auto ezfunc_old = [xcount](mainReal row, mainReal ex, mainReal why) -> mainReal {
        return row * 0 + (1 - ex * ex) * (1 - why * why) * chebBasisFunction(2, 2, ex, why, xcount);
    };*/
    /*auto ezfunc = [](mainReal row, mainReal ex, mainReal why) -> mainReal {
        mainReal temp = row * 0 + 25.0 * ex * ex + 25.0 * why * why;
        if (temp >= 25) {
            return 0;
        }
        return (30 * std::exp(1 / (temp / 25.0 - 1.0)));
    };*/

    for (auto& cal_i : calclist) {
        std::vector<OOGA::calculatorType> one_item_list;
        one_item_list.push_back(cal_i);
        auto res = testRunRecursive<8, 192, rhocount, double>(one_item_list, testfunc2, g);
        std::cout << res << std::endl;
        auto res2 = testRunRecursive<8, 192, rhocount, float>(one_item_list, testfunc2, g);
        std::cout << res2 << std::endl;
    }

    for (auto& cal_i : chebCalclist) {
        std::vector<OOGA::calculatorType> one_item_list;
        one_item_list.push_back(cal_i);
        auto res = testRunRecursive<8, 192, rhocount, double, true>(one_item_list, testfunc2, g);
        std::cout << res << std::endl;
        auto res_2 = testRunRecursive<8, 192, rhocount, float, true>(one_item_list, testfunc2, g);
        std::cout << res_2 << std::endl;
    }

    /*auto res1 = testRun<8, rhocount, double, true>(chebCalclist, testfunc2, g);
    std::cout << res1;
    res1 = testRun<16, rhocount, double, true>(chebCalclist, testfunc2, g);
    std::cout << res1;
    res1 = testRun<32, rhocount, double, true>(chebCalclist, testfunc2, g);
    std::cout << res1;
    res1 = testRun<48, rhocount, double, true>(chebCalclist, testfunc2, g);
    std::cout << res1;
    res1 = testRun<64, rhocount, double, true>(chebCalclist, testfunc2, g);
    std::cout << res1;
    //    res1 = testRun<96, rhocount, double, true>(chebCalclist, testfunc2, g);
    //std::cout << res1;

    res1 = testRun<8, rhocount, double>(calclist, testfunc2, g);
    std::cout << res1;
    res1 = testRun<16, rhocount, double>(calclist, testfunc2, g);
    std::cout << res1;
    res1 = testRun<32, rhocount, double>(calclist, testfunc2, g);
    std::cout << res1;
    res1 = testRun<48, rhocount, double>(calclist, testfunc2, g);
    std::cout << res1;
    res1 = testRun<64, rhocount, double>(calclist, testfunc2, g);
    std::cout << res1;
    res1 = testRun<96, rhocount, double>(calclist, testfunc2, g);
    std::cout << res1;

    auto res2 = testRun<8, rhocount, float, true>(chebCalclist, testfunc2, g);
    std::cout << res2;
    res2 = testRun<16, rhocount, float, true>(chebCalclist, testfunc2, g);
    std::cout << res2;
    res2 = testRun<32, rhocount, float, true>(chebCalclist, testfunc2, g);
    std::cout << res2;
    res2 = testRun<48, rhocount, float, true>(chebCalclist, testfunc2, g);
    std::cout << res2;
    res2 = testRun<64, rhocount, float, true>(chebCalclist, testfunc2, g);
    std::cout << res2;
    res2 = testRun<96, rhocount, float, true>(chebCalclist, testfunc2, g);
    std::cout << res2;

    res2 = testRun<8, rhocount, float>(calclist, testfunc2, g);
    std::cout << res2;
    res2 = testRun<16, rhocount, float>(calclist, testfunc2, g);
    std::cout << res2;
    res2 = testRun<32, rhocount, float>(calclist, testfunc2, g);
    std::cout << res2;
    res2 = testRun<48, rhocount, float>(calclist, testfunc2, g);
    std::cout << res2;
    res2 = testRun<64, rhocount, float>(calclist, testfunc2, g);
    std::cout << res2;
    res2 = testRun<96, rhocount, float>(calclist, testfunc2, g);
    std::cout << res2;*/

    fftw_cleanup();
}

void fft_testing() {
    //using namespace OOGA;
    using OOGA::DCTBasisFunction2;
    using OOGA::functionGrid;
    using OOGA::gridDomain;
    using OOGA::LinearSpacedArray;
    typedef double mainReal;

    gridDomain g;
    g.rhomax = 0.25;
    g.rhomin = 1.55;
    g.xmin = g.ymin = -2;
    g.xmax = g.ymax = 2;
    constexpr int xcount = 4, ycount = 4,
                  rhocount = 3;  // bump up to 64x64x35 later or 128x128x35

    std::vector<mainReal> rhoset;  //TODO(orebas) we should write a function to initialize a functiongrid from a gridDomain.
    std::vector<mainReal> xset;
    std::vector<mainReal> yset;

    rhoset = LinearSpacedArray<mainReal>(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray<mainReal>(g.xmin, g.xmax, xcount);
    yset = LinearSpacedArray<mainReal>(g.ymin, g.ymax, ycount);
    for (int p = 0; p < xcount; ++p) {
        for (int q = 0; q < ycount; ++q) {
            if (p > 3 || q > 3) {
                continue;
            }
            auto basistest = [p, q, g, xcount, ycount](mainReal row, mainReal ex, mainReal why) -> mainReal {
                mainReal xint = (ex - g.xmin) / (g.xmax - g.xmin) * (xcount - 1);
                mainReal yint = (why - g.ymin) / (g.ymax - g.ymin) * (ycount - 1);
                mainReal dr = row * 0 + DCTBasisFunction2(p, q, xint, yint, xcount);
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
            fftw_plan plan = fftw_plan_r2r_2d(xcount, ycount, in2.gridValues.data.data(), out.gridValues.data.data(), FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
            fftw_execute(plan);
            in2.csvPrinter(0);
            std::cout << std::endl;
            out.csvPrinter(0);
            std::cout << std::endl;
            in.fill(basistest);
            in.csvPrinter(0);
            std::cout << std::endl;
        }
        /*for (int i = 0; i < xcount; ++i) for (int j = 0; j < ycount; ++j) {
                if (std::abs(out.gridValues(0, i, j) > 0.01))
                    std::cout << p << " " << q << " " << i << " " << j << " " << out.gridValues(0, i, j) << std::endl;
            }
            */
    }
    std::cout << std::endl;
}
void chebDevel() {
    //using namespace OOGA;
    using OOGA::chebBasisFunction;
    using OOGA::functionGrid;
    using OOGA::gridDomain;
    using OOGA::LinearSpacedArray;
    using mainReal = double;
    constexpr mainReal mainRhoMin = 0.25 / 4.0;
    constexpr mainReal mainRhoMax = 3.0 / 4.0;
    constexpr mainReal mainxyMin = -1;
    constexpr mainReal mainxyMax = 1;
    gridDomain g;
    g.rhomax = mainRhoMax;
    g.rhomin = mainRhoMin;
    g.xmin = g.ymin = mainxyMin;
    g.xmax = g.ymax = mainxyMax;
    constexpr int xcount = 13, ycount = 13,
                  rhocount = 3;  // bump up to 64x64x35 later or 128x128x35
    constexpr mainReal A = 1.5;
    constexpr mainReal B = 1.1;
    constexpr mainReal Normalizer = 50.0;
    std::vector<mainReal> rhoset;
    std::vector<mainReal> xset;
    std::vector<mainReal> yset;

    rhoset = LinearSpacedArray<mainReal>(g.rhomin, g.rhomax, rhocount);
    xset = chebPoints(xcount);
    yset = chebPoints(ycount);

    functionGrid<rhocount, xcount, ycount> f(rhoset, xset, yset);
    functionGrid<rhocount, xcount, ycount> exact(rhoset, xset, yset);
    functionGrid<rhocount, xcount, ycount> m(rhoset, xset, yset);

    auto testfunc2 = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal {
        ex = ex * 4;
        why = why * 4;
        return Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row);
    };

    f.fill(testfunc2);
    OOGA::fftw_wrapper_2d<rhocount, xcount, ycount, mainReal> plan(FFTW_REDFT00);

    for (int p = 0; p < xcount; p++) {
        for (int q = 0; q < ycount; q++) {
            f.clearGrid();
            m = f;
            exact = f;
            f.gridValues(0, p, q) = 1;

            auto cheb_basis_func = [p, q](mainReal row, mainReal ex, mainReal why) -> mainReal {
                return chebBasisFunction(p, q, ex, why, xcount);
            };

            exact.fill(cheb_basis_func);

            std::copy(f.gridValues.data.begin(), f.gridValues.data.begin() + xcount * ycount * rhocount, plan.fftin);
            plan.execute();
            std::copy(plan.fftout, plan.fftout + rhocount * xcount * ycount, m.gridValues.data.begin());  //add division by 4
            for (auto& x : m.gridValues.data) {
                x /= 4.0;
            }

            std::cout << p << " " << q << " " << m.maxNormDiff(exact.gridValues, 0) << std::endl;
        }
    }
}

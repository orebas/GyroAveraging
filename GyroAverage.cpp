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

#undef BOOST_HAS_FLOAT128

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
#include <boost/program_options.hpp>

#undef BOOST_HAS_FLOAT128

#include "ga.h"

template <class RealT = double>
struct resultsRecord {
    std::string function_name;
    OOGA::calculatorType type = OOGA::calculatorType::linearCPU;
    int N = 0;
    std::vector<RealT> rhoset;
    double initTime = 0;
    double calcTime = 0;
    int bits = 0;
    std ::vector<RealT> error;
    friend std::ostream& operator<<(std::ostream& output, const resultsRecord<RealT>& r) {
        auto nameMap = OOGA::calculatorNameMap();
        output << r.function_name << ","
               << nameMap[r.type] << ","
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
    resultsRecord(const std::string& fn, OOGA::calculatorType t_i, int N_i, std ::vector<RealT> rhoset_i, double initTime_i, double calcTime_i, int bits_i)
        : function_name(fn), type(t_i), N(N_i), rhoset(rhoset_i), initTime(initTime_i), calcTime(calcTime_i), bits(bits_i), error(rhoset_i) {
    }
};

template <class RealT = double>
std::ostream& operator<<(std::ostream& output, const std::vector<resultsRecord<RealT>>& r) {
    for (const auto& e : r) {
        std::cout << e << std::endl;
    }
    return output;
}

template <class RealT, typename TFunc1>
resultsRecord<RealT> testConvergence(TFunc1 testfunc, const std::string& fn, OOGA::gridDomain& g, int rhocount) {
    using OOGA::functionGrid;
    using OOGA::GACalculator;
    using OOGA::gridDomain;
    using OOGA::LinearSpacedArray;
    using OOGA::measure;

    for (int N = 8; N < 1024; N++) {
        int xcount = N;
        int ycount = N;
        std::vector<RealT> rhoset;
        std::vector<RealT> cheb_xset, lin_xset, lin_yset;
        std::vector<RealT> cheb_yset;

        rhoset = LinearSpacedArray<RealT>(g.rhomin, g.rhomax, rhocount);
        cheb_xset = chebPoints<RealT>(xcount);
        cheb_yset = chebPoints<RealT>(ycount);
        lin_xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, xcount);
        lin_yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, ycount);

        functionGrid<RealT>
            f(rhoset, lin_xset, lin_yset), f_cheb(rhoset, cheb_xset, cheb_yset), exact(rhoset, lin_xset, lin_yset), result(rhoset, lin_xset, lin_yset);

        exact.fillTruncatedAlmostExactGA(testfunc);
        // NOT DONE
    }
    std::vector<RealT> rhoset;
    std::vector<RealT> xset, lin_xset, lin_yset;
    std::vector<RealT> yset;
}

template <class RealT, typename TFunc1>
resultsRecord<RealT> testRun(const std::string& function_name, OOGA::calculatorType calcType, TFunc1 testfunc, const OOGA::gridDomain& g, int N, int rhocount, OOGA::fileCache* cache = nullptr, bool cheb = false) {
    using OOGA::functionGrid;
    using OOGA::GACalculator;
    using OOGA::gridDomain;
    using OOGA::LinearSpacedArray;
    using OOGA::measure;

    int xcount = N;
    int ycount = N;
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

    functionGrid<RealT>
        f(rhoset, xset, yset),
        exact(rhoset, lin_xset, lin_yset), result(rhoset, lin_xset, lin_yset);

    std::unique_ptr<GACalculator<RealT>> calculator;
    std::ostringstream cacheFileName;
    cacheFileName << function_name << "."
                  << "Trunc-exact"
                  << "." << sizeof(RealT) << "."
                  << f.xcount << "."
                  << f.ycount << "."
                  << f.rhocount << "."
                  << f.rhoset.front() << "."
                  << f.rhoset.back();
    if (cache != nullptr) {
        ////////////////
        std::vector<RealT> check;
        check = cache->read<RealT>(cacheFileName.str());
        if (static_cast<long int>(check.size()) == exact.gridValues.data.size()) {
            //std::cout << "Succesful read" << std::endl;
            exact.gridValues.data = check;
        } else {
            exact.fillTruncatedAlmostExactGA(testfunc);
            cache->save(cacheFileName.str(), exact.gridValues.data.data(), exact.gridValues.data.size() * sizeof(RealT));
        }
    }

    auto func = [&]() -> void { calculator = (GACalculator<RealT>::Factory::newCalculator(calcType, g, exact, cache, xcount / 2)); };
    double initTime = measure<std::chrono::milliseconds>::execution(func);
    f.fill(testfunc);
    auto& b = f;
    auto func2 = [&]() -> void {
        calculator->calculate(b);
    };
    double calcTime = measure<std::chrono::nanoseconds>::execution(func2);
    result = (calculator->calculate(f));
    resultsRecord<RealT> runResults(function_name, calcType, N, std::vector<RealT>(rhoset.begin(), rhoset.end()), initTime, calcTime, sizeof(RealT));
    for (int k = 0; k < rhocount; ++k) {
        runResults.error[k] = exact.maxNormDiff(result.gridValues, k) / exact.maxNorm(k);
    }
    calculator.reset(nullptr);
    std::cout << runResults << std::endl;
    return runResults;
}

template <int rhocount, class RealT, typename TFunc1>
void testRunDiag(const std::string function_name, OOGA::calculatorType calcType, TFunc1 testfunc, const OOGA::gridDomain& g, OOGA::fileCache* cache = nullptr, bool cheb = false, int N = 8) {
    using OOGA::functionGrid;
    using OOGA::GACalculator;
    using OOGA::gridDomain;
    using OOGA::LinearSpacedArray;
    using OOGA::measure;

    int xcount = N;
    int ycount = N;
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

    functionGrid<RealT>
        f(rhoset, xset, yset),
        exact(rhoset, lin_xset, lin_yset), result(rhoset, lin_xset, lin_yset);

    std::unique_ptr<GACalculator<RealT>> calculator;
    exact.fillTruncatedAlmostExactGA(testfunc);

    auto func = [&]() -> void { calculator = (GACalculator<RealT>::Factory::newCalculator(calcType, g, exact, cache, xcount / 2)); };
    double initTime = measure<std::chrono::milliseconds>::execution(func);
    f.fill(testfunc);
    auto& b = f;
    auto func2 = [&]() -> void {
        calculator->calculate(b);
    };
    double calcTime = measure<std::chrono::nanoseconds>::execution(func2);
    result = (calculator->calculate(f));
    resultsRecord<RealT> runResults(function_name, calcType, N, std::vector<RealT>(rhoset.begin(), rhoset.end()), initTime, calcTime, sizeof(RealT));

    for (int k = 0; k < rhocount; ++k) {
        std::cout << "rho is " << rhoset[k] << std::endl;
        std::cout << "f: \n";
        f.csvPrinter(k);
        std::cout << "exact(trunc):\n";
        exact.csvPrinter(k);
        std::cout << "result\n";
        result.csvPrinter(k);
        std::cout << "diff\n";
        exact.csvPrinterDiff(result, k);
        std::cout << std::endl;

        runResults.error[k] = exact.maxNormDiff(result.gridValues, k) / exact.maxNorm(k);
    }
    calculator.reset(nullptr);
    std::cout << runResults << std::endl;
}

/*template <int N, int rhocount, class RealT, bool cheb = false, typename TFunc1>
  std::vector<resultsRecord<RealT>> testRunMultiple(const std::vector<OOGA::calculatorType>& calclist, TFunc1 testfunc, OOGA::gridDomain& g) {
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
}*/

template <int rhocount, class RealT, typename TFunc1>
void testRunList(const std::string function_name, OOGA::calculatorType calcType, TFunc1 testfunc, const OOGA::gridDomain& g, OOGA::fileCache* cache = nullptr, bool cheb = false) {
    try {
        for (int i = 8; i < 518; i += 4) {  //go to 385 or farther?
            auto r = testRun<RealT>(function_name, calcType, testfunc, g, i, rhocount, cache, cheb);
            if (r.initTime > 5000 * 1000 || r.calcTime > 9e10) {
                break;
            }
        }

    } catch (std::exception& e) {
        std::cout << "Finished a list. " << e.what() << std::endl;
    }
}

/*template <int N, int MaxN, int rhocount, class RealT, bool cheb = false, typename TFunc1>
  std::vector<resultsRecord<RealT>> testRunRecursive(const std::vector<OOGA::calculatorType>& calclist, TFunc1 testfunc, OOGA::gridDomain& g) {
    bool fail = false;
    std::vector<resultsRecord<RealT>> result;
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
}*/

void cache_testing(const std::string& directory) {
    using OOGA::fileCache;
    fileCache fc(directory);
    std::vector<double> a(5, 10);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = i + 100;
    }
    std::cout << a << std::endl;

    fc.save("A2.123", a.data(), sizeof(double) * a.size());
    std::vector<double> results = fc.read<double>("A2.123");
    std::cout << results << std::endl;

    //calculator = (GACalculator<RealT>::Factory::newCalculator(OOGA::calculatorType::chebCPUDense, g, exact, xcount / 2));
}

template <class RealT, typename TFunc1>
class functionTemplateHack {
    RealT operator()(RealT a, RealT b, RealT c) {}
};

int main(int argc, char* argv[]) {
    //fft_testing();
    //chebDevel();
    //fftw_cleanup();
    // return 0;
    //using namespace OOGA;
    using OOGA::chebBasisFunction;
    using OOGA::functionGrid;
    using OOGA::GACalculator;
    using OOGA::gridDomain;
    using OOGA::LinearSpacedArray;
    namespace po = boost::program_options;
    using mainReal = double;
    gsl_set_error_handler_off();

    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")("calc", po::value<int>(), "choose which calculator to test run")
        //whitespace because (see below)
        ("func", po::value<int>(), "choose which function to test run")("cache", po::value<std::string>(), "choose the cache directory")
        //whitespace because my editor wraps this poorly
        ("bits", po::value<int>(), "choose 32 or 64 (for float or double)")("diag", po::value<int>(), "set to 1 for diagnostic")("N", po::value<int>(), "N to use for diagnostics");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    int diag_N = 8;
    int calc_option = -1;
    int func_option = -1;
    int bits_option = -1;
    int diag_option = -1;
    std::string cache_dir;

    if (vm.count("help")) {
        std::cout << "Example: --calc=1 --func=2\n";  //TODO(orebas):rewrite this text
        return 1;
    }

    if (vm.count("calc")) {
        // std::cout << "calculator chosen: " << vm["calc"].as<int>() << std::endl;
        calc_option = vm["calc"].as<int>();

    } else {
        std::cout << "Calculator not chosen.  Please pick one." << std::endl;
        return 1;
    }

    if (vm.count("func")) {
        // std::cout << "func chosen: " << vm["func"].as<int>() << std::endl;
        func_option = vm["func"].as<int>();
    } else {
        std::cout << "func not chosen.  Please pick one." << std::endl;
        return 1;
    }

    if (vm.count("bits")) {
        bits_option = vm["bits"].as<int>();
    } else {
        std::cout << "bits not chosen.  Please pick 32 or 64." << std::endl;
        return 1;
    }

    if (vm.count("diag")) {
        diag_option = vm["diag"].as<int>();
    }

    if (vm.count("N")) {
        diag_N = vm["N"].as<int>();
    }

    if (vm.count("cache")) {
        //std::cout << "cache dir chosen: " << vm["cache"].as<std::string>() << std::endl;
        cache_dir = vm["cache"].as<std::string>();
    } else {
        //std::cout << "cache dir not specified.  We will continue without caching." << std::endl;
    }

    //cache_testing(cache_dir);
    //return 0;

    using OOGA::fileCache;
    fileCache cache(cache_dir);

    //the below can in theory be command line args as well:

    constexpr mainReal mainRhoMin = 0.25 / 4.0;  //used to be 0.25/4
    constexpr mainReal mainRhoMax = 3.5 / 4.0;   //used to be 3/4
    constexpr mainReal mainxyMin = -1;
    constexpr mainReal mainxyMax = 1;

    gridDomain g;
    g.rhomax = mainRhoMax;
    g.rhomin = mainRhoMin;
    g.xmin = g.ymin = mainxyMin;
    g.xmax = g.ymax = mainxyMax;
    constexpr int rhocount = 3;

    std::vector<mainReal> rhoset;
    auto nameMap = OOGA::calculatorNameMap();

    std::vector<OOGA::calculatorType> calclist;
    calclist.push_back(OOGA::calculatorType::linearCPU);
    calclist.push_back(OOGA::calculatorType::linearDotProductCPU);
    calclist.push_back(OOGA::calculatorType::bicubicCPU);
    calclist.push_back(OOGA::calculatorType::bicubicDotProductCPU);
    calclist.push_back(OOGA::calculatorType::DCTCPUCalculator2);
    calclist.push_back(OOGA::calculatorType::DCTCPUPaddedCalculator2);
    calclist.push_back(OOGA::calculatorType::chebCPUDense);
    calclist.push_back(OOGA::calculatorType::linearDotProductGPU);
    calclist.push_back(OOGA::calculatorType::bicubicDotProductGPU);
    calclist.push_back(OOGA::calculatorType::chebGPUDense);

    //the below is a relic of when padding was a parameter, and that's a reasonable thing to investigate.
    //constexpr mainReal padtest = xcount * mainRhoMax / std::abs(mainxyMax - mainxyMin);
    //constexpr int padcount = mymax(8, 4 + static_cast<int>(std::ceil(padtest)));

    using fp = mainReal (*)(mainReal row, mainReal ex, mainReal why);  // pointer to function which takes 3 mainReals, and returns one.
    //FYI capture-less lambdas can cast themselves to function pointers, but you can't if you capture.
    std::vector<fp> functionVec;

    auto easyFunc = [](mainReal row, mainReal ex, mainReal why) -> mainReal {
        constexpr mainReal A = 22;
        constexpr mainReal B = 1.1;
        constexpr mainReal Normalizer = 50.0;
        return Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row);
    };

    using std::abs;
    using std::max;

    auto rungeFunc = [](double row, double ex, double why) -> double {
        return ((1.0 - ex * ex) * (1.0 - why * why)) / (1 + 25 * ((ex - 0.2) * (ex - 0.2) + (why + 0.5) * (why + 0.5)));
    };

    auto polyFunc = [](double row, double ex, double why) -> double {
        return ((1.0 - ex * ex) * (1.0 - why * why) * (1 - 0.0 * 3.0 * ex * why));
    };

    auto sqrtFunc = [](double row, double ex, double why) -> double {
        double r = (ex - 0.2) * (ex - 0.2) + (why + 0.5) * (why + 0.5);

        return std::sqrt(std::sqrt(r));
    };

    auto stripFunc = [](double row, double ex, double why) -> double {
        double r = abs(ex - why);
        double l = max(0.0, 0.75 - r);
        return (l * l * l * l * (4 * r + 1)) * (1.0 - ex * ex) * (1.0 - why * why);
    };

    auto hardFunc = [](double row, double ex, double why) -> double {
        double r = abs(ex - why);
        double l = max(0.0, 0.75 - r);
        return (l * l * l * l * (4 * r + 1) + 1.0 / (1 + 25 * ((ex - 0.2) * (ex - 0.2) + (why + 0.5) * (why + 0.5)))) *
               std::cbrt(1.0 - ex * ex) * std::cbrt(1.0 - why * why);
    };

    auto crazyhardFunc = [](double row, double ex, double why) -> double {
        auto dist = ex * ex + why * why;
        auto hard = exp(dist) * pow(
                                    (1.0 / cosh(4.0 * sin(40.0 * dist))),
                                    exp(dist));
        hard += exp(ex) *
                pow((1.0 / cosh(4.0 * sin(40.0 * ex))), exp(ex));
        if (why + ex / 2.0 < 00.0) {
            hard += 1.5;
        }
        hard *= (1.0 - ex * ex);
        hard *= (1.0 - why * why);
        return hard;
    };
    std::vector<std ::string> functionNameVec;
    functionVec.push_back(easyFunc);
    functionNameVec.push_back("SmoothExp");

    functionVec.push_back(polyFunc);
    functionNameVec.push_back("SmoothPoly");

    functionVec.push_back(rungeFunc);
    functionNameVec.push_back("SmoothRunge");

    functionVec.push_back(sqrtFunc);
    functionNameVec.push_back("NonsmoothSqrt");

    functionVec.push_back(stripFunc);
    functionNameVec.push_back("NonsmoothAbs");

    functionVec.push_back(hardFunc);
    functionNameVec.push_back("NonsmoothRungeAbs");

    std::string function_name = "ConstantZero";
    auto func_lambda = [](double row, double ex, double why) -> double {
        return 0;
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

    if (calc_option < 0 || calc_option >= static_cast<int>(calclist.size())) {
        std::cout << "That's not a valid calculator." << std::endl;
        return 1;
    }
    if (calclist[calc_option] == OOGA::calculatorType::DCTCPUCalculator2) {
        exit(0);
    }
    bool cheb_grid_needed = false;
    if (calclist[calc_option] == OOGA::calculatorType::chebCPUDense || calclist[calc_option] == OOGA::calculatorType::chebGPUDense) {
        cheb_grid_needed = true;
    }
    std::cout
        << "functionName, calculator,N,initTime, calcTime, calcHz, bytes, maxError, blankColumn, err1, err2, err3, Blank" << std::endl;

    if (diag_option == 1) {
        testRunDiag<rhocount, double>(functionNameVec[func_option], calclist[calc_option], functionVec[func_option], g, &cache, cheb_grid_needed, diag_N);
        return 0;
    }
    if (bits_option == 64) {
        testRunList<rhocount, double>(functionNameVec[func_option], calclist[calc_option], functionVec[func_option], g, &cache, cheb_grid_needed);
    }
    if (bits_option == 32) {
        testRunList<rhocount, float>(functionNameVec[func_option], calclist[calc_option], functionVec[func_option], g, &cache, cheb_grid_needed);
    }
    /*switch (func_option) {
        case 0:

            function_name = "smooth_exp";
            testRunList<rhocount, double>(function_name, calclist[calc_option], easyFunc, g, &cache, cheb_grid_needed);
            //testRunList<rhocount, float>(function_name, calclist[calc_option], easyfunc, g, cheb_grid_needed);

            break;

        case 1:
            function_name = "nonsmooth_sqrt_r";
            testRunList<rhocount, double>(function_name, calclist[calc_option], sqrtFunc, g, &cache, cheb_grid_needed);
            //testRunList<rhocount, float>(function_name, calclist[calc_option], mediumfunc, g, cheb_grid_needed);
            break;
        case 2:
            function_name = "smooth_runge";
            testRunList<rhocount, double>(function_name, calclist[calc_option], rungeFunc, g, &cache, cheb_grid_needed);
            //testRunList<rhocount, float>(function_name, calclist[calc_option], mediumfunc, g, cheb_grid_needed);
            break;
        case 3:
            function_name = "nonsmooth_abs";

            testRunList<rhocount, double>(function_name, calclist[calc_option], stripFunc, g, &cache, cheb_grid_needed);
            //testRunList<rhocount, float>(function_name, calclist[calc_option], crazyhardfunc, g, cheb_grid_needed);
            break;
        case 4:
            function_name = "nonsmooth_abs_plus_runge";

            testRunList<rhocount, double>(function_name, calclist[calc_option], hardFunc, g, &cache, cheb_grid_needed);
            //testRunList<rhocount, float>(function_name, calclist[calc_option], crazyhardfunc, g, cheb_grid_needed);
            break;
        default:
            break;
    };*/

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

            functionGrid<mainReal> in(rhoset, xset, yset), in2(rhoset, xset, yset), out(rhoset, xset, yset);
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

    functionGrid<mainReal> f(rhoset, xset, yset);
    functionGrid<mainReal> exact(rhoset, xset, yset);
    functionGrid<mainReal> m(rhoset, xset, yset);

    auto testfunc2 = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal {
        ex = ex * 4;
        why = why * 4;
        return Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row);
    };

    f.fill(testfunc2);
    OOGA::fftw_wrapper_2d<mainReal> plan(rhocount, xcount, ycount, FFTW_REDFT00);

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
/*
double temp(double f) {
    double r = 0;
    double x = 0;
    double y = 0;
    double r2 = r * r;
    double r3 = r * r * r;
    double r4 = r * r * r * r;
    double r5 = r * r * r * r * r;
    double r6 = r * r * r * r * r * r;
    double r7 = r * r * r * r * r * r * r;

    double x2 = x * x;
    double x3 = x * x * x;
    double x4 = x * x * x * x;
    double x5 = x * x * x * x * x;
    double x6 = x * x * x * x * x * x;
    double x7 = x * x * x * x * x * x * x;

    double y2 = y * y;
    double y3 = y * y * y;
    double y4 = y * y * y * y;
    double y5 = y * y * y * y * y;
    double y6 = y * y * y * y * y * y;
    double y7 = y * y * y * y * y * y * y;

    
}*/

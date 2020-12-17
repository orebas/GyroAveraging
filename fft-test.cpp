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
#include <boost/program_options.hpp>

#include "ga.h"
#include "gautils.h"
#include "pocketfft_hdronly.h"

int main(int argc, char **argv) {
    std::cout << "Test\n";
    for (double t = -0.9999; t < 1.0; t += 0.05) {
        double a = boost::math::chebyshev_t(5, t);
        double b = OOGA::fast_cheb_t(5, t);
        std::cout
            << t << " \t" << a - b << "\t" << a << "\t" << b << std::endl;
    }
    exit(0);
    using RealT = double;
    using mainReal = double;

    constexpr mainReal mainRhoMin = 0.25 / 4.0;  //used to be 0.25/4
    constexpr mainReal mainRhoMax = 3.5 / 4.0;   //used to be 3/4
    constexpr mainReal mainxyMin = -1;
    constexpr mainReal mainxyMax = 1;

    OOGA::gridDomain g;
    g.rhomax = mainRhoMax;
    g.rhomin = mainRhoMin;
    g.xmin = g.ymin = mainxyMin;
    g.xmax = g.ymax = mainxyMax;
    constexpr int rhocount = 1;

    const int r = 1;
    const int N = 8;
    int xcount = N;
    int ycount = N;
    std::vector<RealT> rhoset;
    std::vector<RealT> cheb_xset, lin_xset, lin_yset;
    std::vector<RealT> cheb_yset;

    rhoset = OOGA::LinearSpacedArray<RealT>(g.rhomin, g.rhomax, rhocount);
    cheb_xset = chebPoints<RealT>(xcount);
    cheb_yset = chebPoints<RealT>(ycount);
    lin_xset = OOGA::LinearSpacedArray<RealT>(g.xmin, g.xmax, xcount);
    lin_yset = OOGA::LinearSpacedArray<RealT>(g.ymin, g.ymax, ycount);

    OOGA::functionGrid<mainReal> f(rhoset, lin_xset, lin_yset);
    auto t1 = f;
    auto t2 = f;

    /*auto easyFunc = [](mainReal row, mainReal ex, mainReal why) -> mainReal {
        constexpr mainReal A = 22;
        constexpr mainReal B = 1.1;
        constexpr mainReal Normalizer = 50.0;
        return Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row);
    };*/

    auto easyFunc = [](mainReal row, mainReal ex, mainReal why) -> mainReal {
        constexpr mainReal A = 22;
        constexpr mainReal B = 1.1;
        constexpr mainReal Normalizer = 50.0;
        return cos(2 * OOGA::pi * ex) * sin(4 * OOGA::pi * why) + ex * ex * (1.0 - why) + 0.2 + why / 4.0;
    };

    f.fill(easyFunc);
    f.csvPrinter(0);

    OOGA::fftw_wrapper_2d<RealT>
        plan(r, N, N, FFTW_REDFT10);

    std::copy(f.gridValues.data.begin(), f.gridValues.data.begin() + N * N, plan.fftin);
    plan.execute();

    using namespace pocketfft;
    shape_t shape;
    shape.push_back(N);
    shape.push_back(N);
    shape_t axes;
    axes.push_back(0);
    axes.push_back(1);
    std::vector<double> datain = f.gridValues.data;

    stride_t strided(shape.size());
    size_t tmpd = sizeof(double);

    /*for (int i = shape.size() - 1; i >= 0; --i) {
        strided[i] = tmpd;
        tmpd *= shape[i];
    }*/
    strided[0] = tmpd;
    strided[1] = tmpd * N;
    std::vector<double> res(N * N, 0);
    std::copy(f.gridValues.data.begin(), f.gridValues.data.begin() + N * N, datain.data());

    dct<double>(shape, strided, strided, axes, 3, datain.data(), res.data(), 1, false, 1.);

    std::copy(plan.fftout, plan.fftout + N * N, t1.gridValues.data.data());
    std::copy(res.data(), res.data() + N * N, t2.gridValues.data.data());
    std::cout << "\nFFTW:\n";
    t1.csvPrinter(0);
    std::cout << "\npocketFFT\n";
    t2.csvPrinter(0);

    std::cout << "\n Diff\n";
    t1.csvPrinterDiff(t2, 0);
}
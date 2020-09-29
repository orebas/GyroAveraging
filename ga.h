//TODO(orebas): triangular cheb matrices
//TODO(orebas): harder functions (franke, runge, etc)
//TODO(orebas):  clean up header dependencies
//TODO(orebas): zero case seems to fail with some calculators.  hardcode it?  exclude it?
#ifndef GYROAVERAGING_GA_H
#define GYROAVERAGING_GA_H




#include<iostream>
#include <fftw3.h>
#include <omp.h>

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/chebyshev_transform.hpp>
#include <boost/math/special_functions/next.hpp>
#include <eigen3/Eigen/Eigen>
#include <exception>
#include <iostream>
#include <new>
#include <boost/optional.hpp>

#include "viennacl/compressed_matrix.hpp"
/*#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <string>
#include <vector>*/

#include "gautils.h"

namespace OOGA {

//Below defines our domain of integration.  It used to be templated, but this didn't add much.
//template <class RealT = double>
struct gridDomain {
    using RealT = double;
    RealT xmin = 0, xmax = 0, ymin = 0, ymax = 0, rhomin = 0, rhomax = 0;
};

//The below class holds function values without reference to the function itself.  It contains the code for calculating finite differences,
// bicubic coefficients, for filling in a grid from lambas, and for taking norms (and norm diffs).  This object is the main input into the various GA calculators.

//TODO(orebas): I think we should have two seperate classes that compose with functionGrid, one for equispaced and one for cheb
//also they should just take griddomain as constructor (and xcount,ycount).  no need to take arbitrary grid points.
template <class RealT = double>
class functionGrid {
   public:
    int xcount;
    int ycount;
    int rhocount;

   public:
    using fullgrid = Array3d<RealT>;              // rhocount, xcount, ycount,
    using fullgridInterp = Array4d<RealT>;        //rhocount, xcount, ycount, 4,
    using derivsGrid = Array4d<RealT>;            // at each rho,x,y calculate [f,f_x,f_y,f_xy] // rhocount, xcount, ycount, 4,
    using bicubicParameterGrid = Array4d<RealT>;  // rhocount, xcount, ycount, 16,
    using biquadParameterGrid = Array4d<RealT>;   // rhocount, xcount, ycount, 9,
    using SpM = Eigen::SparseMatrix<RealT, Eigen::RowMajor>;
    using SpT = Eigen::Triplet<RealT>;

    std::vector<RealT> rhoset;
    std::vector<RealT> xset;
    std::vector<RealT> yset;
    fullgrid gridValues;  // input values of f

    void csvPrinter(int rho) {
        for (int j = 0; j < xcount; j++) {
            for (int k = 0; k < ycount; k++) {
                std::cout << gridValues(rho, j, k) << ",";
            }
            std::cout << std::endl;
        }
    }

    void csvPrinterDiff(const functionGrid &n, int rho) {
        for (int j = 0; j < xcount; j++) {
            for (int k = 0; k < ycount; k++) {
                std::cout << gridValues(rho, j, k) - n.gridValues(rho, j, k) << ",";
            }
            std::cout << std::endl;
        }
    }

    void clearGrid() {
#pragma omp parallel for collapse(3)
        for (auto i = 0; i < rhocount; i++) {
            for (auto j = 0; j < xcount; j++) {
                for (auto k = 0; k < ycount; k++) {
                    gridValues(i, j, k) = 0;
                }
            }
        }
    }
    RealT RMSNorm(int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++) {
            for (int k = 0; k < ycount; k++) {
                result += gridValues(rho, j, k) * gridValues(rho, j, k);
            }
            return std::sqrt(result / (xcount * ycount));
        }
    }
    RealT maxNorm(int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++) {
            for (int k = 0; k < ycount; k++) {
                result = std::max(result, std::abs(gridValues(rho, j, k)));
            }
        }
        return result;
    }

    RealT RMSNormDiff(const fullgrid &m2, int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++) {
            for (int k = 0; k < ycount; k++) {
                RealT t = gridValues(rho, j, k) - m2(rho, j, k);
                result += t * t;
            }
        }
        return std::sqrt(result / (xcount * ycount));
    }
    RealT maxNormDiff(const fullgrid &m2, int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++) {
            for (int k = 0; k < ycount; k++) {
                RealT t = gridValues(rho, j, k) - m2(rho, j, k);
                result = std::max(result, std::abs(t));
            }
        }
        return result;
    }

    // below fills a grid, given a function of rho, x, and y
    template <typename TFunc>
    void fill(TFunc f) {
#pragma omp parallel for collapse(3)
        for (auto i = 0; i < rhocount; i++) {
            for (auto j = 0; j < xcount; j++) {
                for (auto k = 0; k < ycount; k++) {
                    gridValues(i, j, k) = f(rhoset[i], xset[j], yset[k]);
                }
            }
        }
    }
    // below fills a grid, given a function of i,j,k
    template <typename TFunc>
    void fillbyindex(TFunc f) {
#pragma omp parallel for collapse(3)
        for (int i = 0; i < rhocount; i++) {
            for (int j = 0; j < xcount; j++) {
                for (int k = 0; k < ycount; k++) {
                    gridValues(i, j, k) = f(i, j, k);
                }
            }
        }
    }
    template <typename TFunc1>
    void fillAlmostExactGA(TFunc1 f) {                   // f is only used to fill the rho=0 case
        fillbyindex([&](int i, int j, int k) -> RealT {  // calculates GA by adaptive trapezoid rule on actual input function.
            RealT xc = xset[j];
            RealT yc = yset[k];
            if (rhoset[i] == 0) {
                return f(i, xc, yc);
            }
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
    void fillTruncatedAlmostExactGA(TFunc1 f) {          //calculates GA by adaptive trapezoid rule on actual intput function
        fillbyindex([&](int i, int j, int k) -> RealT {  //but hard truncates to 0 outside of defined domain
            RealT xc = xset[j];
            RealT yc = yset[k];
            if (rhoset[i] == 0) {
                return f(i, xc, yc);
            }
            auto new_f = [&](RealT x) -> RealT {
                RealT ex = xc + rhoset[i] * std::sin(x);
                RealT why = yc - rhoset[i] * std::cos(x);
                if ((ex < xset[0]) || (ex > xset.back())) {
                    return 0;
                }
                if ((why < yset[0]) || (why > yset.back())) {
                    return 0;
                }
                return f(rhoset[i], ex, why);
            };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);  //we can use trapezoid here too.
            return result;
        });
    }

    template <typename TFunc1>
    void fillTrapezoidInterp(TFunc1 f) {  // this calls interp2d. gridValues must be
                                          // filled, but we don't need setupInterpGrid.
        fillbyindex([&](int i, int j, int k) -> RealT {
            RealT xc = xset[j];
            RealT yc = yset[k];
            if (rhoset[i] == 0) {
                return f(rhoset[i], xc, yc);
            }
            auto new_f = [&](RealT x) -> RealT {
                return interp2d(i, xc + rhoset[i] * std::sin(x),
                                yc - rhoset[i] * std::cos(x));
            };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
    }
    // below requires the bicubic parameter grid to be populated.
    void fillBicubicInterp() {
        fillbyindex([&](int i, int j, int k) -> RealT {
            RealT xc = xset[j];
            RealT yc = yset[k];
            if (rhoset[i] == 0) {
                return gridValues(i, j, k);
            }
            auto new_f = [&](RealT x) -> RealT {
                return interpNaiveBicubic(i, xc + rhoset[i] * std::sin(x),
                                          yc - rhoset[i] * std::cos(x));
            };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
    }
    functionGrid(const std::vector<RealT> &rhos, const std::vector<RealT> &xes,
                 const std::vector<RealT> &yies)
        : xcount(xes.size()), ycount(yies.size()), rhocount(rhos.size()), rhoset(rhos), xset(xes), yset(yies), gridValues(rhocount, xcount, ycount) {
        std::sort(xset.begin(), xset.end());
        std::sort(yset.begin(), yset.end());
        std::sort(rhoset.begin(), rhoset.end());
    }

    void interpIndexSearch(const RealT x, const RealT y, int &xindex,
                           int &yindex) const {
        if ((x < xset[0]) || (y < yset[0]) || (x > xset.back()) ||
            (y > yset.back())) {
            xindex = xcount - 1;  // the top right corner should have zeros.
            yindex = ycount - 1;
            return;
        }
        xindex =
            std::upper_bound(xset.begin(), xset.end(), x) - xset.begin() - 1;
        yindex =
            std::upper_bound(yset.begin(), yset.end(), y) - yset.begin() - 1;
        xindex = std::min(std::max(xindex, 0), xcount - 2);
        yindex = std::min(ycount - 2, std::max(yindex, 0));
        assert((xset[xindex]) <= x && (xset[xindex + 1] >= x));
        assert((yset[yindex]) <= y && (yset[yindex + 1] >= y));
    }

    RealT interp2d(int rhoindex, const RealT x, const RealT y) const {
        assert((rhoindex >= 0) && (rhoindex < rhocount));
        if ((x <= xset[0]) || (y <= yset[0]) || (x >= xset.back()) || (y >= yset.back())) {
            return 0;
        }
        int xindex = 0, yindex = 0;
        interpIndexSearch(x, y, xindex, yindex);
        auto result = BilinearInterpolation<RealT>(
            gridValues(rhoindex, xindex, yindex),
            gridValues(rhoindex, xindex + 1, yindex),
            gridValues(rhoindex, xindex, yindex + 1),
            gridValues(rhoindex, xindex + 1, yindex + 1), xset[xindex],
            xset[xindex + 1], yset[yindex], yset[yindex + 1], x, y);

        return result;
    }

    inline void integrand(const RealT rho, const RealT xc, const RealT yc,
                          const RealT gamma, RealT &xn, RealT &yn) {
        xn = xc + rho * std::sin(gamma);
        yn = yc - rho * std::cos(gamma);
    }

    static bicubicParameterGrid setupBicubicGrid(  //this calculates bicubic interpolation parameters
        const functionGrid<RealT> &f) {
        using Eigen::Matrix;

        auto d = f.calcDerivsGrid();
        bicubicParameterGrid b(f.rhocount, f.xcount, f.ycount, 16);

#pragma omp parallel for collapse(3)
        for (int i = 0; i < f.rhocount; i++) {
            // we explicitly rely on parameters being initialized to 0,
            // including the top and right sides.
            for (int j = 0; j < f.xcount - 1; j++) {
                for (int k = 0; k < f.ycount - 1; k++) {
                    RealT x0 = f.xset[j], x1 = f.xset[j + 1];
                    RealT y0 = f.yset[k], y1 = f.yset[k + 1];
                    Matrix<RealT, 4, 4> X, Y, RHS, A, temp1, temp2;

                    RHS << d(i, j, k, 0), d(i, j, k + 1, 0), d(i, j, k, 2),
                        d(i, j, k + 1, 2), d(i, j + 1, k, 0),
                        d(i, j + 1, k + 1, 0), d(i, j + 1, k, 2),
                        d(i, j + 1, k + 1, 2), d(i, j, k, 1), d(i, j, k + 1, 1),
                        d(i, j, k, 3), d(i, j, k + 1, 3), d(i, j + 1, k, 1),
                        d(i, j + 1, k + 1, 1), d(i, j + 1, k, 3),
                        d(i, j + 1, k + 1, 3);
                    X << 1, x0, x0 * x0, x0 * x0 * x0, 1, x1, x1 * x1,
                        x1 * x1 * x1, 0, 1, 2 * x0, 3 * x0 * x0, 0, 1, 2 * x1,
                        3 * x1 * x1;
                    Y << 1, 1, 0, 0, y0, y1, 1, 1, y0 * y0, y1 * y1, 2 * y0,
                        2 * y1, y0 * y0 * y0, y1 * y1 * y1, 3 * y0 * y0,
                        3 * y1 * y1;

                    // temp1 = X.fullPivLu().inverse(); //this line crashes on
                    // my home machine without optimization turned on. temp2 =
                    // Y.fullPivLu().inverse();

                    A = X.inverse() * RHS * Y.inverse();
                    for (int t = 0; t < 16; ++t) {
                        b(i, j, k, t) = A(t % 4, t / 4);
                    }
                }
            }
        }
        return b;
    }

    derivsGrid calcDerivsGrid() const {    //this calculates finite difference derivative estimates.
        RealT ydenom = yset[1] - yset[0];  //see e.g. http://www.holoborodko.com/pavel/2014/11/04/computing-mixed-derivatives-by-finite-differences/
        RealT xdenom = xset[1] - xset[0];
        const fullgrid &g = gridValues;
        derivsGrid derivs(rhocount, xcount, ycount, 4);

#pragma omp parallel for
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
                for (int j = 2; j <= xcount - 3; j++) {
                    derivs(i, j, k, 1) =
                        (1.0 * g(i, j - 2, k) + -8.0 * g(i, j - 1, k) +
                         0.0 * g(i, j, k) + 8.0 * g(i, j + 1, k) +
                         -1.0 * g(i, j + 2, k)) /
                        (12.0 * xdenom);
                }
            }
            for (int j = 0; j < xcount; j++) {
                derivs(i, j, 0, 2) = 0;
                derivs(i, j, ycount - 1, 2) = 0;
                derivs(i, j, 1, 2) =
                    (-3.0 * g(i, j, 0) + -10.0 * g(i, j, 1) +
                     18.0 * g(i, j, 2) + -6.0 * g(i, j, 3) + 1.0 * g(i, j, 4)) /
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
                     g(i, j + 1, ycount - 2 - 1) -
                     g(i, j - 1, ycount - 2 + 1)) /
                    (4 * xdenom * ydenom);
            }
            for (int k = 1; k < ycount - 1; k++) {
                derivs(i, 1, k, 3) = (g(i, 1 - 1, k - 1) + g(i, 1 + 1, k + 1) -
                                      g(i, 1 + 1, k - 1) - g(i, 1 - 1, k + 1)) /
                                     (4 * xdenom * ydenom);

                derivs(i, xcount - 2, k, 3) =
                    (g(i, xcount - 2 - 1, k - 1) + g(i, xcount - 2 + 1, k + 1) -
                     g(i, xcount - 2 + 1, k - 1) -
                     g(i, xcount - 2 - 1, k + 1)) /
                    (4 * xdenom * ydenom);
            }
        }
        return derivs;
    }
    std::array<RealT, 4> arcIntegral(RealT rho, RealT xc, RealT yc, RealT s0,
                                     RealT s1);
};

// below code creates an RAII framework for FFTW.  Right now we have only specialized for float and double
// and only handle the cases we need.
template <class RealT>
class fftw_wrapper_2d {
   public:
    BOOST_STATIC_ASSERT(sizeof(RealT) == 0);
};
//specialization for double

template <>
class fftw_wrapper_2d<double> {
   private:
    fftw_plan plan;

   public:
    using RealT = double;
    RealT *fftin;
    RealT *fftout;

    int rhocount;
    int xcount;
    int ycount;
    fftw_wrapper_2d() = delete;
    explicit fftw_wrapper_2d(int rc, int xc, int yc, fftw_r2r_kind t) : rhocount(rc), xcount(xc), ycount(yc) {
        int rank = 2;
        int n[] = {xcount, ycount};
        int howmany = rhocount;
        int idist = xcount * ycount;
        int odist = xcount * ycount;
        int istride = 1;
        int ostride = 1;
        fftw_r2r_kind type[] = {t, t};

        fftin = static_cast<RealT *>(fftw_malloc(rhocount * xcount * ycount * sizeof(RealT)));
        if (fftin == nullptr) {
            throw std::bad_alloc();
        }
        fftout = static_cast<RealT *>(fftw_malloc(rhocount * xcount * ycount * sizeof(RealT)));
        if (fftout == nullptr) {
            fftw_free(fftin);
            throw std::bad_alloc();
        }

        plan = fftw_plan_many_r2r(rank, static_cast<int *>(n), howmany, fftin, static_cast<int *>(n), istride, idist, fftout, static_cast<int *>(n), ostride, odist, static_cast<fftw_r2r_kind *>(type), FFTW_MEASURE);
    }

    fftw_wrapper_2d(const fftw_wrapper_2d<RealT> &other) = delete;
    fftw_wrapper_2d(fftw_wrapper_2d<RealT> &&other) noexcept
        : plan{other.plan}, fftin{other.fftin}, fftout{other.fftout}, rhocount(other.rhocount), xcount(other.xcount), ycount(other.ycount) {
        other.fftin = nullptr;
        other.fftout = nullptr;
    }

    fftw_wrapper_2d<RealT> &operator=(const fftw_wrapper_2d<RealT> &other) = delete;

    fftw_wrapper_2d<RealT> &operator=(fftw_wrapper_2d<RealT> &&other) = delete;
    void execute(const functionGrid<RealT> &in, functionGrid<RealT> *out) {
        std::copy(in.gridValues.data.begin(), in.gridValues.data.begin() + in.xset.size() * in.yset.size() * in.rhoset.size(), fftin);
        fftw_execute(plan);
        std::copy(fftout, fftout + in.rhoset.size() * in.xset.size() * in.yset.size(), out->gridValues.data.begin());
    }

    void execute() {
        fftw_execute(plan);
    }
    ~fftw_wrapper_2d() {
        fftw_free(fftin);
        fftw_free(fftout);
        fftw_destroy_plan(plan);
    };
};

//specialization for float (fp32)
template <>
class fftw_wrapper_2d<float> {
   private:
    fftwf_plan plan;

   public:
    using RealT = float;
    RealT *fftin;
    RealT *fftout;
    int rhocount;
    int xcount;
    int ycount;

    fftw_wrapper_2d() = delete;
    explicit fftw_wrapper_2d(int rc, int xc, int yc, fftwf_r2r_kind t) : rhocount(rc), xcount(xc), ycount(yc) {
        int rank = 2;
        int n[] = {xcount, ycount};
        int howmany = rhocount;
        int idist = xcount * ycount;
        int odist = xcount * ycount;
        int istride = 1;
        int ostride = 1;
        fftw_r2r_kind type[] = {t, t};

        fftin = static_cast<RealT *>(fftwf_malloc(rhocount * xcount * ycount * sizeof(RealT)));
        fftout = static_cast<RealT *>(fftwf_malloc(rhocount * xcount * ycount * sizeof(RealT)));
        plan = fftwf_plan_many_r2r(rank, static_cast<int *>(n), howmany, fftin, static_cast<int *>(n), istride, idist, fftout, static_cast<int *>(n), ostride, odist, static_cast<fftw_r2r_kind *>(type), FFTW_MEASURE);
    }

    fftw_wrapper_2d(const fftw_wrapper_2d<RealT> &other) = delete;
    fftw_wrapper_2d(fftw_wrapper_2d<RealT> &&other) noexcept
        : plan{other.plan}, fftin{other.fftin}, fftout{other.fftout}, rhocount(other.rhocount), xcount(other.xcount), ycount(other.ycount) {
        other.fftin = nullptr;
        other.fftout = nullptr;
    }

    fftw_wrapper_2d<RealT> &operator=(const fftw_wrapper_2d<RealT> &other) = delete;

    fftw_wrapper_2d<RealT> &operator=(fftw_wrapper_2d<RealT> &&other) = delete;
    void execute(const functionGrid<RealT> &in, functionGrid<RealT> *out) {
        std::copy(in.gridValues.data.begin(), in.gridValues.data.begin() + in.xset.size() * in.yset.size() * in.rhoset.size(), fftin);
        fftwf_execute(plan);
        std::copy(fftout, fftout + in.xset.size() * in.yset.size() * in.rhoset.size(), out->gridValues.data.begin());
    }

    void execute() {
        fftwf_execute(plan);
    }
    ~fftw_wrapper_2d() {
        fftwf_free(fftin);
        fftwf_free(fftout);
        fftwf_destroy_plan(plan);
    };
};

/* GACalculator API
Create
Calculate
Name
Short Description
Long Description
Destroy/Free
*/

//Below is some prelim code before we define the main gyroaverage calculators.
//First, the below enums represent the calculators we have implemeneted so far.
//We also have a map from this enum to short descriptions of each calculator, formatted
//for printing tables.

enum class calculatorType { linearCPU,
                            linearDotProductCPU,
                            bicubicCPU,
                            bicubicDotProductCPU,
                            DCTCPUCalculator,
                            DCTCPUCalculator2,
                            DCTCPUPaddedCalculator2,
                            bicubicDotProductGPU,
                            linearDotProductGPU,
                            chebCPUDense,
                            chebGPUDense };

inline std::map<OOGA::calculatorType, std::string> calculatorNameMap() {
    std::map<OOGA::calculatorType, std::string> nameMap;
    nameMap[OOGA::calculatorType::linearCPU] =
        "linear interp; trapezoid rule; CPU ";
    nameMap[OOGA::calculatorType::linearDotProductCPU] =
        "linear interp; CPU Sparse Matrix   ";
    nameMap[OOGA::calculatorType::bicubicCPU] =
        "bicubic interp; trapezoid rule; CPU";
    nameMap[OOGA::calculatorType::bicubicDotProductCPU] =
        "bicubic interp; CPU Sparse Matrix  ";
    nameMap[OOGA::calculatorType::DCTCPUCalculator] =
        "Very slow fourier Method ; deprecated";
    nameMap[OOGA::calculatorType::DCTCPUCalculator2] =
        "DCT+Bessel+IDCT                    ";
    nameMap[OOGA::calculatorType::DCTCPUPaddedCalculator2] =
        "DCT+Bessel+IDCT; on padded grid    ";
    nameMap[OOGA::calculatorType::bicubicDotProductGPU] =
        "bicubic interp; GPU Sparse Matrix  ";
    nameMap[OOGA::calculatorType::linearDotProductGPU] =
        "linear interp; GPU Sparse Matrix   ";
    nameMap[OOGA::calculatorType::chebCPUDense] =
        "chebyshev interp; CPU Dense Matrix ";
    nameMap[OOGA::calculatorType::chebGPUDense] =
        "chebyshev interp; GPU Dense Matrix ";

    return nameMap;
}

//The below abstract base class defines the interfaces for our calculators.
//TODO(orebas): The interface design is broken, as right now we can pass chebyshev grids
//to equispaced calculators and vice versa; this should be a compile time error (i.e. a type problem).

//to use this, you call
//auto calculator = OOGA::GACalculator<rhocount, xcount, ycount, RealT, N>::Factory::newCalculator(calculatorType, gridDomain, functionGrid))
//then you build a functionGrid of the appropriate size, and you call
// auto result = calculator->calculate(f);

//each of the calculators, when constructed, generally siezes whatever resources they need, which includes
// (often) a lot of memory, including GPU memory where appropriate.
//Right now we don't check or catch memory allocation failures.  However destruction is
// RAII-style and we have tested for memory leaks.

template <class RealT = double>
class GACalculator {
   public:
    struct Factory {
        static std::unique_ptr<GACalculator<RealT>>
        newCalculator(calculatorType c, const gridDomain &g, functionGrid<RealT> &f, int padcount = 0);
    };
    virtual ~GACalculator() = default;
    GACalculator() = default;
    virtual functionGrid<RealT> calculate(
        functionGrid<RealT> &f) = 0;
    GACalculator(const GACalculator &) = delete;
    GACalculator &operator=(const GACalculator &) = delete;
    GACalculator(GACalculator &&other) = delete;
    GACalculator &operator=(GACalculator &&) = delete;
};

template <class RealT = double>
class linearCPUCalculator
    : public GACalculator<RealT> {
   public:
    friend class GACalculator<RealT>;
    static std::unique_ptr<GACalculator<RealT>>
    create() {
        return std::make_unique<linearCPUCalculator>();
    }
    functionGrid<RealT> calculate(
        functionGrid<RealT> &f) override {
        functionGrid<RealT> m =
            f;  // do we need to write a copy constructor?

        m.fillbyindex([&](int i, int j, int k) -> RealT {
            RealT xc = f.xset[j];
            RealT yc = f.yset[k];
            if (f.rhoset[i] == 0) {
                return f.gridValues(i, j, k);
            }
            auto new_f = [&](RealT x) -> RealT {
                return f.interp2d(i, xc + f.rhoset[i] * std::sin(x),
                                  yc - f.rhoset[i] * std::cos(x));
            };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
        return m;
    }
};

template <class RealT = double>
class linearDotProductCPU
    : public GACalculator<RealT> {
   private:
    Eigen::SparseMatrix<RealT, Eigen::RowMajor> linearSparseTensor;

   public:
    friend class GACalculator<RealT>;

    static std::unique_ptr<GACalculator<RealT>>
    create(const functionGrid<RealT> &f) {
        return std::make_unique<linearDotProductCPU>(f);
    }
    static Eigen::SparseMatrix<RealT, Eigen::RowMajor> assembleFastGACalc(const functionGrid<RealT> &ftocopy) {
        auto f = ftocopy;
        Eigen::SparseMatrix<RealT, Eigen::RowMajor> LTOffsetTensor;
        LTOffsetTensor.setZero();

        int max_threads = omp_get_max_threads() + 1;

        std::vector<std::vector<Eigen::Triplet<RealT>>>
            TripletVecVec(max_threads);
#pragma omp parallel for collapse(2) num_threads(8)  //todo:  can we collapse(3)
        for (auto i = 0; i < f.rhocount; i++) {
            for (auto j = 0; j < f.xcount; j++) {
                for (auto k = 0; k < f.ycount; k++) {
                    std::vector<indexedPoint<RealT>> intersections;
                    RealT rho = f.rhoset[i];
                    RealT xc = f.xset[j];
                    RealT yc = f.yset[k];
                    RealT xmin = f.xset[0], xmax = f.xset.back();
                    RealT ymin = f.yset[0], ymax = f.yset.back();
                    std::vector<RealT> xIntersections, yIntersections;
                    // these two loops calculate all potential intersection points
                    // between GA circle and the grid.
                    for (auto v : f.xset) {
                        if (std::abs(v - xc) <= rho) {
                            RealT deltax = v - xc;
                            RealT deltay = std::sqrt(rho * rho - deltax * deltax);
                            if ((yc + deltay >= ymin) && (yc + deltay <= ymax)) {
                                intersections.push_back(
                                    indexedPoint<RealT>(v, yc + deltay, 0));
                            }
                            if ((yc - deltay >= ymin) && (yc - deltay <= ymax)) {
                                intersections.push_back(
                                    indexedPoint<RealT>(v, yc - deltay, 0));
                            }
                        }
                    }
                    for (auto v : f.yset) {
                        if (std::abs(v - yc) <= rho) {
                            RealT deltay = v - yc;
                            RealT deltax = std::sqrt(rho * rho - deltay * deltay);
                            if ((xc + deltax >= xmin) && (xc + deltax <= xmax)) {
                                intersections.push_back(
                                    indexedPoint<RealT>(xc + deltax, v, 0));
                            }
                            if ((xc - deltax >= xmin) && (xc - deltax <= xmax)) {
                                intersections.push_back(
                                    indexedPoint<RealT>(xc - deltax, v, 0));
                            }
                        }
                    }
                    for (auto &v : intersections) {
                        v.s = std::atan2(v.xvalue - xc, yc - v.yvalue);
                        if (v.s < 0) {
                            v.s += 2 * pi;
                        }
                        assert((0 <= v.s) && (v.s < 2 * pi));
                        assert((xc + std::sin(v.s) * rho) - v.xvalue < 1e-10);
                        assert((yc - std::cos(v.s) * rho) - v.yvalue < 1e-10);
                    }
                    indexedPoint<RealT> temp(0, 0, 0);
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
                        int xInterpIndex = 0, yInterpIndex = 0;
                        if (s1 - s0 < 1e-12) {
                            continue;  // if two of our points are equal or very
                                       // close to one another, we make the arc
                                       // larger. this will probably happen for s=0
                                       // and s=pi/2, but doesn't cost us anything.
                        }
                        integrand(rho, xc, yc, (s0 + s1) / 2, xmid,
                                  ymid);  // this just calculates into (xmid,ymid)
                                          // the point half through the arc.
                        std::array<RealT, 4> coeffs = arcIntegral(rho, xc, yc, s0, s1);
                        f.interpIndexSearch(xmid, ymid, xInterpIndex, yInterpIndex);

                        if (!((xInterpIndex == (f.xset.size() - 1)) &&
                              (yInterpIndex == (f.yset.size() - 1)))) {
                            RealT x = f.xset[xInterpIndex],
                                  a = f.xset[xInterpIndex + 1];
                            RealT y = f.yset[yInterpIndex],
                                  b = f.yset[yInterpIndex + 1];
                            RealT c1 = coeffs[0] / (2 * pi);
                            RealT c2 = coeffs[1] / (2 * pi);
                            RealT c3 = coeffs[2] / (2 * pi);
                            RealT c4 = coeffs[3] / (2 * pi);
                            RealT denom = (a - x) * (b - y);

                            std::array<int, 4> LTSources({0, 0, 0, 0}),
                                LTTargets({0, 0, 0, 0});
                            std::array<RealT, 4> LTCoeffs({0, 0, 0, 0});
                            LTSources[0] =
                                &(f.gridValues(i, xInterpIndex, yInterpIndex)) -
                                &(f.gridValues(0, 0, 0));
                            LTSources[1] =
                                &(f.gridValues(i, xInterpIndex + 1, yInterpIndex)) -
                                &(f.gridValues(0, 0, 0));
                            LTSources[2] =
                                &(f.gridValues(i, xInterpIndex, yInterpIndex + 1)) -
                                &(f.gridValues(0, 0, 0));
                            LTSources[3] = &(f.gridValues(i, xInterpIndex + 1,
                                                          yInterpIndex + 1)) -
                                           &(f.gridValues(0, 0, 0));
                            auto &fastGALTResult = f.gridValues;
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
                                TripletVecVec[omp_get_thread_num()].emplace_back(
                                    Eigen::Triplet<RealT>(LTTargets[l], LTSources[l], LTCoeffs[l]));
                            }
                        }
                    }
                }
            }  // namespace OOGA
        }      // namespace OOGA
        std::vector<Eigen::Triplet<RealT>>
            Triplets;

        for (int i = 0; i < TripletVecVec.size(); i++) {
            for (auto iter = TripletVecVec[i].begin();
                 iter != TripletVecVec[i].end(); ++iter) {
                Triplets.emplace_back(*iter);
            }
        }

        LTOffsetTensor.resize(f.rhocount * f.xcount * f.ycount,
                              f.rhocount * f.xcount * f.ycount);
        LTOffsetTensor.setFromTriplets(Triplets.begin(), Triplets.end());
        return LTOffsetTensor;

        // std::cout << "Number of RealT  products needed for LT calc: " <<
        // LTOffsetTensor.nonZeros() << " and rough memory usage is " <<
        // LTOffsetTensor.nonZeros() * (sizeof(RealT) + sizeof(long)) << std::endl;
    }  // namespace OOGA
    explicit linearDotProductCPU(const functionGrid<RealT> &f) {
        linearSparseTensor = assembleFastGACalc(f);
    };

    functionGrid<RealT> calculate(
        functionGrid<RealT> &f) override {
        functionGrid<RealT> m =
            f;

        m.clearGrid();
        Eigen::Map<Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic>> source(
            f.gridValues.data.data(), f.rhocount * f.xcount * f.ycount, 1);
        Eigen::Map<Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic>> target(
            m.gridValues.data.data(), f.rhocount * f.xcount * f.ycount, 1);
        target = linearSparseTensor * source;
        return m;
    }
};  // namespace OOGA

template <class RealT = double>
class bicubicCPUCalculator
    : public GACalculator<RealT> {
    //typename functionGrid<RealT>::bicubicParameterGrid bicubicParameters;

   public:
    friend class GACalculator<RealT>;
    static std::unique_ptr<GACalculator<RealT>>
    create() {
        return std::make_unique<bicubicCPUCalculator>();
    }

    RealT interpNaiveBicubic(const functionGrid<RealT> &f,
                             const typename functionGrid<RealT>::bicubicParameterGrid &b,
                             int rhoindex, const RealT x, const RealT y) const {
        assert((rhoindex >= 0) && (rhoindex < f.rhocount));
        if ((x <= f.xset[0]) || (y <= f.yset[0]) || (x >= f.xset.back()) ||
            (y >= f.yset.back())) {
            return 0;
        }
        int xindex = 0, yindex = 0;
        RealT result = 0;
        f.interpIndexSearch(x, y, xindex, yindex);
        std::array<RealT, 4> xns = {1, x, x * x, x * x * x};
        std::array<RealT, 4> yns = {1, y, y * y, y * y * y};
        for (int i = 0; i <= 3; ++i) {
            for (int j = 0; j <= 3; ++j) {
                result += b(rhoindex, xindex, yindex, j * 4 + i) *
                          xns[i] * yns[j];  //TODO(orebas) horner's method (bivariate)
            }
        }
        return result;
    }

    functionGrid<RealT> calculate(
        functionGrid<RealT> &f) override {
        functionGrid<RealT> m = f;
        typename functionGrid<RealT>::bicubicParameterGrid b = functionGrid<RealT>::setupBicubicGrid(f);
        m.fillbyindex([&](int i, int j, int k) -> RealT {
            RealT xc = f.xset[j];
            RealT yc = f.yset[k];
            if (f.rhoset[i] == 0) {
                return f.gridValues(i, j, k);
            }
            auto new_f = [&](RealT x) -> RealT {
                return this->interpNaiveBicubic(f, b, i, xc + f.rhoset[i] * std::sin(x),
                                                yc - f.rhoset[i] * std::cos(x));
            };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
        return m;
    }
};

/* BELOW IS NOT FUNCTIONAL, dont call it. so far it's mostly copied from bicubic.  I may abandon this. */
//Here we had coded the beginnings of a biquadratic calculator.  the idea is to match
// on each patch the values at the corners (the nodes) as well as finite difference estimates for
// "outward" derivatives at the midpoint of each edge, and  a single value as the mixed derivative
// at the midpoint of the rectangle.

/*
// FFT - The below is abandoned code.; it only computes the first rhoset[0].
//We are keeping it because the function DCTTest should be part of a testing framework.
//also the initialization is VERY slow.
template <int rhocount, int xcount, int ycount, class RealT = RealT>
class DCTCPUCalculator
    : public GACalculator< RealT> {
   private:
    RealT *fftin, *fftout;
    fftw_plan plan;
    Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic> denseGAMatrix;

   public:
    friend class GACalculator< RealT>;

    void slowTruncFill(const gridDomain &g) {
        denseGAMatrix.resize(xcount * ycount, xcount * ycount);
        denseGAMatrix.setZero();
        std::vector<RealT> rhoset;
        std::vector<RealT> xset;
        std::vector<RealT> yset;

        rhoset = LinearSpacedArray<RealT>(g.rhomin, g.rhomax, rhocount);
        xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, xcount);
        yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, ycount);
        {
#pragma omp parallel for
            for (int p = 0; p < xcount; ++p) {
                functionGrid<rhocount, xcount, ycount> f(rhoset, xset, yset);
                for (int q = 0; q < ycount; ++q) {
                    auto basistest = [p, q, g](RealT row, RealT ex, RealT why) -> RealT {
                        RealT xint = (ex - g.xmin) / (g.xmax - g.xmin) * (xcount - 1);
                        RealT yint = (why - g.ymin) / (g.ymax - g.ymin) * (ycount - 1);
                        return row * 0.0 + DCTBasisFunction2(p, q, xint, yint, xcount);
                    };

                    f.fillTruncatedAlmostExactGA(basistest);
                    Eigen::Map<Eigen::Matrix<RealT, xcount * ycount, 1>> m(f.gridValues.data.data());
                    denseGAMatrix.row(ycount * p + q) = m;  //TODO(orebas) NOT FINISHED REDO THIS LINE
                }
            }
        }
        denseGAMatrix.transposeInPlace();
        //std::cout << "dense matrix: \n " << denseGAMatrix << std::endl;
    }

    void fastFill(const gridDomain &g) {
        denseGAMatrix.resize(xcount * ycount, xcount * ycount);
        denseGAMatrix.setZero();
        std::vector<RealT> rhoset;
        std::vector<RealT> xset;
        std::vector<RealT> yset;

        rhoset = LinearSpacedArray<RealT>(g.rhomin, g.rhomax, rhocount);
        xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, xcount);
        yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, ycount);
        {
#pragma omp parallel for
            for (int p = 0; p < xcount; ++p) {
                functionGrid<rhocount, xcount, ycount> f(rhoset, xset, yset);
                for (int q = 0; q < ycount; ++q) {
                    auto besseltest = [p, q, g](RealT row, RealT ex, RealT why) {
                        constexpr RealT N = xcount;
                        RealT xint = (ex - g.xmin) / (g.xmax - g.xmin) * (xcount - 1);
                        RealT yint = (why - g.ymin) / (g.ymax - g.ymin) * (ycount - 1);
                        RealT ap = 2, bq = 2;
                        if (p == 0) {
                            ap = 1;
                        }
                        if (q == 0) {
                            bq = 1;
                        }

                        RealT a = pi * p * (2.0 * xint * (2 - 1) + 1) / (2.0 * N);  //change "2" to "N"
                        RealT c = pi * q * (2.0 * yint * (2 - 1) + 1) / (2.0 * N);
                        RealT d = pi * p * row * (N - 1) / (N * (g.xmax - g.xmin));
                        RealT b = -pi * q * row * (N - 1) / (N * (g.ymax - g.ymin));
                        RealT j = boost::math::cyl_bessel_j(0, std::sqrt(b * b + d * d));
                        return ap * bq * std::cos(a) * std::cos(c) * j;
                    };

                    f.fill(besseltest);
                    Eigen::Map<Eigen::Matrix<RealT, xcount * ycount, 1>> m(f.gridValues.data.data());
                    denseGAMatrix.row(ycount * p + q) = m;  //TODO(orebas) NOT FINISHED REDO THIS LINE
                }
            }
        }
        denseGAMatrix.transposeInPlace();
        //std::cout << "dense matrix: \n " << denseGAMatrix << std::endl;
    }

    explicit DCTCPUCalculator(const gridDomain &g) {
        fftin = (RealT *)fftw_malloc(xcount * ycount * sizeof(RealT));
        fftout = (RealT *)fftw_malloc(xcount * ycount * sizeof(RealT));
        plan = fftw_plan_r2r_2d(xcount, ycount, fftin, fftout, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);  //Forward DCT

        fastFill(g);
    }

    void DCTTest(const gridDomain &g) {
        Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic> tm, testm;  //"true matrix" vs "test matrix"

        tm.resize(xcount * ycount, xcount * ycount);
        tm.setZero();
        testm.resize(xcount * ycount, xcount * ycount);
        testm.setZero();
        std::vector<RealT> rhoset;
        std::vector<RealT> xset;
        std::vector<RealT> yset;

        rhoset = LinearSpacedArray<RealT>(g.rhomin, g.rhomax, rhocount);
        xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, xcount);
        yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, ycount);

#pragma omp parallel for
        for (int p = 0; p < xcount; ++p) {
            functionGrid<rhocount, xcount, ycount> f(rhoset, xset, yset);
            functionGrid<rhocount, xcount, ycount> ftest(rhoset, xset, yset);

            for (int q = 0; q < ycount; ++q) {
                auto basistest = [p, q, g](RealT row, RealT ex, RealT why) -> RealT {
                    RealT xint = (ex - g.xmin) / (g.xmax - g.xmin) * (xcount - 1);
                    RealT yint = (why - g.ymin) / (g.ymax - g.ymin) * (ycount - 1);
                    return row * 0 + DCTBasisFunction2(p, q, xint, yint, xcount);
                };

                auto besseltest = [p, q, g](RealT row, RealT ex, RealT why) {
                    constexpr RealT N = xcount;
                    RealT xint = (ex - g.xmin) / (g.xmax - g.xmin) * (xcount - 1);
                    RealT yint = (why - g.ymin) / (g.ymax - g.ymin) * (ycount - 1);
                    RealT ap = 2, bq = 2;
                    if (p == 0) {
                        ap = 1;
                    }
                    if (q == 0) {
                        bq = 1;
                    }

                    RealT a = pi * p * (2.0 * xint * (2 - 1) + 1) / (2.0 * N);  //change "2" to "N"
                    RealT c = pi * q * (2.0 * yint * (2 - 1) + 1) / (2.0 * N);

                    RealT d = pi * p * row * (N - 1) / (N * (g.xmax - g.xmin));
                    RealT b = -pi * q * row * (N - 1) / (N * (g.ymax - g.ymin));
                    RealT j = boost::math::cyl_bessel_j(0, std::sqrt(b * b + d * d));
                    return ap * bq * std::cos(a) * std::cos(c) * j;
                };

                //f.fillTruncatedAlmostExactGA(basistest);  //test impact of truncation later.
                f.fillAlmostExactGA(basistest);
                ftest.fill(besseltest);
                Eigen::Map<Eigen::Matrix<RealT, xcount * ycount, 1>> m(f.gridValues.data.data());
                Eigen::Map<Eigen::Matrix<RealT, xcount * ycount, 1>> m2(ftest.gridValues.data.data());

                tm.row(ycount * p + q) = m;
                testm.row(ycount * p + q) = m2;
            }
        }
        tm.transposeInPlace();
        testm.transposeInPlace();
        //std::cout << "True Matrix:" << std::endl
        //          << tm << std::endl
        //          << std::endl
        //          << std::endl
        //          << "Test Matrix:" << std::endl
        //          << testm << std::endl
        //          << std::endl
        //          << std::endl
        //          << "Truncated Matrix:" << std::endl
        //          << denseGAMatrix << std::endl
        //          << std::endl
        //          << std::endl;
}  // namespace OOGA

~DCTCPUCalculator() override {
    fftw_free(fftin);
    fftw_free(fftout);
    fftw_destroy_plan(plan);
}
static std::unique_ptr<GACalculator< RealT>>
create(const gridDomain &g) {
    return std::make_unique<DCTCPUCalculator>(g);
}

functionGrid<RealT>  calculate(
    functionGrid<RealT>  &f) override {
    functionGrid<RealT>  m = f;  // do we need to write a copy constructor?

     

    std::copy(f.gridValues.data.begin(), f.gridValues.data.begin() + xcount * ycount, fftin);
    fftw_execute(plan);
    Eigen::Map<Eigen::Matrix<RealT, xcount * ycount, 1>> X(fftout);
    //std::cout << "FFT result: \n " << X << std::endl;
    X *= (1.0 / (xcount * ycount * 4));

    Eigen::Map<Eigen::Matrix<RealT, xcount * ycount, 1>>
        b(m.gridValues.data.data());
    b = denseGAMatrix * X;
    //std::cout << std::endl
    //         << b << std::endl;
    return m;
}
};
*/

//FFT

//FFT2
template <class RealT = double>
class DCTCPUCalculator2
    : public GACalculator<RealT> {
   private:
    gridDomain g;
    Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic> denseGAMatrix;

    fftw_wrapper_2d<RealT> plan;
    fftw_wrapper_2d<RealT> plan_inv;

   public:
    friend class GACalculator<RealT>;
    std::shared_ptr<std::vector<RealT>> besselVals = nullptr;

    explicit DCTCPUCalculator2(const gridDomain &gd, const functionGrid<RealT> &f, std::shared_ptr<std::vector<RealT>> besselbuff = nullptr)
        : plan(f.rhoset.size(), f.xset.size(), f.yset.size(), FFTW_REDFT10),
          plan_inv(f.rhoset.size(), f.xset.size(), f.yset.size(), FFTW_REDFT01) {
        g = gd;
        if (besselbuff != nullptr) {
            besselVals = besselbuff;
        } else {
            besselVals = std::make_shared<std::vector<RealT>>(f.rhoset.size() * f.xset.size() * f.yset.size(), 0);
            auto rhoset = LinearSpacedArray(g.rhomin, g.rhomax, f.rhoset.size());

#pragma omp parallel for collapse(3)
            for (int r = 0; r < f.rhocount; r++) {
                for (int p = 0; p < f.xcount; ++p) {
                    for (int q = 0; q < f.ycount; ++q) {
                        RealT Nx = f.xcount;
                        RealT Ny = f.ycount;
                        double d = pi * p * rhoset[r] * (Nx - 1) / (Nx * (g.xmax - g.xmin));
                        double b = -pi * q * rhoset[r] * (Ny - 1) / (Ny * (g.ymax - g.ymin));
                        double j = boost::math::cyl_bessel_j(0, std::sqrt(b * b + d * d));
                        (*besselVals)[r * f.xset.size() * f.yset.size() + f.yset.size() * p + q] = j / (f.xset.size() * f.yset.size() * 4.0);
                    }
                }
            }
        }
    }

    static std::unique_ptr<GACalculator<RealT>>
    create(const gridDomain &g, const functionGrid<RealT> &f) {
        return std::make_unique<DCTCPUCalculator2>(g, f);
    }

    functionGrid<RealT> calculate(
        functionGrid<RealT> &f) override {
        functionGrid<RealT> m = f;

        std::copy(f.gridValues.data.begin(), f.gridValues.data.begin() + f.xcount * f.ycount * f.rhocount, plan.fftin);
        plan.execute();
        std::copy(plan.fftout, plan.fftout + f.rhocount * f.xcount * f.ycount, plan_inv.fftin);
        for (int i = 0; i < f.rhocount * f.xcount * f.ycount; ++i) {
            plan_inv.fftin[i] *= (*besselVals)[i];
        }
        plan_inv.execute();
        std::copy(plan_inv.fftout, plan_inv.fftout + f.rhocount * f.xcount * f.ycount, m.gridValues.data.begin());

        return m;
    }
};

//FFT2
//FFTPADDED
template <class RealT = double>
class DCTCPUPaddedCalculator
    : public GACalculator<RealT> {
   private:
    int padcount;
    gridDomain paddedg;

    std::vector<RealT> xnew;
    std::vector<RealT> ynew;
    boost::optional<functionGrid<RealT>> paddedf;  //rhocount, xcount + padcount*2, ycount+padcount*2,RealT

   public:
    friend class GACalculator<RealT>;
    std::unique_ptr<DCTCPUCalculator2<RealT>> dctCalc;  //  rhocount, xcount + padcount * 2, ycount + padcount * 2,

    explicit DCTCPUPaddedCalculator(const gridDomain &g, const functionGrid<RealT> &f, int pcount, std::shared_ptr<std::vector<RealT>> besselbuff = nullptr)
        : padcount(pcount), paddedg(g) {
        RealT deltax = (g.xmax - g.xmin) / (f.xcount - 1.0);
        RealT deltay = (g.ymax - g.ymin) / (f.ycount - 1.0);
        paddedg.xmin = g.xmin - deltax * padcount;
        paddedg.xmax = g.xmax + deltax * padcount;
        paddedg.ymin = g.xmin - deltay * padcount;
        paddedg.ymax = g.xmax + deltay * padcount;

        RealT xdelta = f.xset[1] - f.xset[0];
        RealT ydelta = f.yset[1] - f.yset[0];
        RealT xmax = f.xset.back();
        RealT ymax = f.yset.back();
        RealT xmin = f.xset[0];
        RealT ymin = f.yset[0];
        xnew = LinearSpacedArray(xmin - xdelta * padcount, xmin - xdelta, padcount);
        ynew = LinearSpacedArray(ymin - ydelta * padcount, ymin - ydelta, padcount);

        auto xPost = LinearSpacedArray(xmin + xdelta, xmax + xdelta * padcount, padcount);
        auto yPost = LinearSpacedArray(ymin + ydelta, ymax + ydelta * padcount, padcount);
        xnew.insert(xnew.end(), f.xset.begin(), f.xset.end());
        xnew.insert(xnew.end(), xPost.begin(), xPost.end());
        ynew.insert(ynew.end(), f.yset.begin(), f.yset.end());
        ynew.insert(ynew.end(), yPost.begin(), yPost.end());
        paddedf = functionGrid<RealT>{f.rhoset, xnew, ynew};
        dctCalc = std::make_unique<DCTCPUCalculator2<RealT>>(paddedg, paddedf.value(), besselbuff);  //rhocount, xcount + padcount * 2, ycount + padcount * 2,
    }

    static std::unique_ptr<DCTCPUPaddedCalculator<RealT>>
    create(const gridDomain &g, const functionGrid<RealT> &f, int pcount, std::shared_ptr<std::vector<RealT>> besselbuff = nullptr) {
        return std::make_unique<DCTCPUPaddedCalculator>(g, f, pcount, besselbuff);
    }

    functionGrid<RealT> calculate(
        functionGrid<RealT> &f) override {
        functionGrid<RealT> m = f;

        paddedf.value().clearGrid();

#pragma omp parallel for collapse(3)
        for (int i = 0; i < f.rhocount; ++i) {
            for (int j = 0; j < f.xcount; ++j) {
                for (int k = 0; k < f.ycount; ++k) {
                    paddedf.value().gridValues(i, j + padcount, k + padcount) = f.gridValues(i, j, k);
                }
            }
        }
        auto paddedf2 = dctCalc->calculate(paddedf.value());

#pragma omp parallel for collapse(3)
        for (int i = 0; i < f.rhocount; ++i) {
            for (int j = 0; j < f.xcount; ++j) {
                for (int k = 0; k < f.ycount; ++k) {
                    m.gridValues(i, j, k) = paddedf2.gridValues(i, j + padcount, k + padcount);
                }
            }
        }

        return m;
    }
};
//FFT2

//cheb
template <class RealT = double>
class chebCPUDense
    : public GACalculator<RealT> {
   private:
    fftw_wrapper_2d<RealT> plan;                                                      // = fftw_wrapper_2d<rhocount, xcount, ycount, RealT>(FFTW_REDFT00);
    std::vector<Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic>> denseGAMatrix;  //we want rhocount of these.

   public:
    friend class GACalculator<RealT>;

    /* void slowTruncFill(const gridDomain &g) {
        for (int rho_iter = 0; rho_iter < rhocount; ++rho_iter) {
            denseGAMatrix[rho_iter].resize(xcount * ycount, xcount * ycount);
            denseGAMatrix[rho_iter].setZero();
            std::vector<RealT> rhoset;
            std::vector<RealT> xset;
            std::vector<RealT> yset;

            rhoset = LinearSpacedArray<RealT>(g.rhomin, g.rhomax, rhocount);
            xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, xcount);
            yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, ycount);
            {
#pragma omp parallel for
                for (int p = 0; p < xcount; ++p) {
                    functionGrid<rhocount, xcount, ycount> f(rhoset, xset, yset);
                    for (int q = 0; q < ycount; ++q) {
                        auto basistest = [p, q, g](RealT row, RealT ex, RealT why) -> RealT {
                            return row * 0 + chebBasisFunction(p, q, ex, why, xcount);
                        };

                        f.fillTruncatedAlmostExactGA(basistest);
                        Eigen::Map<Eigen::Matrix<RealT, xcount * ycount, 1>> m(f.gridValues.data.data());
                        denseGAMatrix[rho_iter].row(ycount * p + q) = m;  //TODO(orebas) NOT FINISHED REDO THIS LINE
                    }
                }
            }
            denseGAMatrix[rho_iter].transposeInPlace();
            //std::cout << "dense matrix: \n " << denseGAMatrix[rho_iter] << std::endl;
        }
    }*/

    void static fastFill(const gridDomain &g, const functionGrid<RealT> &paramf, std::vector<Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic>> &dgma) {
        for (int rho_iter = 0; rho_iter < paramf.rhocount; ++rho_iter) {
            dgma[rho_iter].resize(paramf.xcount * paramf.ycount, paramf.xcount * paramf.ycount);
            dgma[rho_iter].setZero();
        }
        std::vector<RealT> rhoset;
        std::vector<RealT> xset;
        std::vector<RealT> yset;

        rhoset = LinearSpacedArray<RealT>(g.rhomin, g.rhomax, paramf.rhocount);
        xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, paramf.xcount);
        yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, paramf.ycount);

        //functionGrid<RealT> f(rhoset, xset, yset);
        std::vector<std::unique_ptr<DCTCPUPaddedCalculator<RealT>>> calcset;  //TODO(orebas) replace 59 with xcount/2, or something better

        calcset.emplace_back(DCTCPUPaddedCalculator<RealT>::create(g, paramf, paramf.xcount / 2));
        int max_threads = omp_get_max_threads() + 1;
        for (int p = 1; p < max_threads; ++p) {
            calcset.emplace_back(DCTCPUPaddedCalculator<RealT>::create(g, paramf, paramf.xcount / 2, (calcset[0]->dctCalc)->besselVals));
            if (calcset.back() == nullptr) {
                std::cout << "ERROR" << std::endl;
                exit(0);
            }
        }
#pragma omp parallel for  //same as above this breaks the code
        for (int p = 0; p < paramf.xcount; ++p) {
            functionGrid<RealT> threadf(rhoset, xset, yset);
            for (int q = 0; q < threadf.ycount; ++q) {
                const int N = threadf.xcount;
                auto basistest = [p, q, g, N](RealT row, RealT ex, RealT why) -> RealT {
                    return row * 0 + chebBasisFunction(p, q, ex, why, N);
                };

                threadf.fill(basistest);
                auto res = calcset[omp_get_thread_num()]->calculate(threadf);  //this is  FFT + IFFT.
                for (int rho_iter = 0; rho_iter < threadf.rhocount; ++rho_iter) {
                    Eigen::Map<Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic>> m(res.gridValues.data.data() + rho_iter * threadf.xcount * threadf.ycount, threadf.xcount * threadf.ycount, 1);
                    dgma[rho_iter].col(threadf.ycount * p + q) = m;  //TODO(orebas) NOT FINISHED REDO THIS LINE
                }
            }
        }
    }

    explicit chebCPUDense(const gridDomain &g, const functionGrid<RealT> &f)
        : plan(f.rhocount, f.xcount, f.ycount, FFTW_REDFT00), denseGAMatrix(f.rhocount) {
        //slowTruncFill(g);
        fastFill(g, f, denseGAMatrix);
    }

    static std::unique_ptr<GACalculator<RealT>>
    create(const gridDomain &g, const functionGrid<RealT> &f) {
        return std::make_unique<chebCPUDense>(g, f);
    }

    // TODO(orebas):there are some unecessary copies and allocating in the below.
    functionGrid<RealT> calculate(
        functionGrid<RealT> &f) override {
        functionGrid<RealT> m = f;
        std::copy(f.gridValues.data.begin(), f.gridValues.data.begin() + f.rhocount * f.xcount * f.ycount, plan.fftin);
        plan.execute();
        for (int rho_iter = 0; rho_iter < f.rhocount; ++rho_iter) {
            Eigen::Map<Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic>> X(plan.fftout + rho_iter * f.xcount * f.ycount, f.xcount * f.ycount, 1);
            X *= (1.0 / ((f.xcount - 1) * (f.ycount - 1)));

            Eigen::Map<Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic>> b(m.gridValues.data.data() + rho_iter * f.xcount * f.ycount, f.xcount * f.ycount, 1);
            b = denseGAMatrix[rho_iter] * X;
        }
        return m;
    }
};

//cheb

//chebGPU

template <class RealT = double>
class chebGPUDense
    : public GACalculator<RealT> {
   private:
    fftw_wrapper_2d<RealT> plan;  // = fftw_wrapper_2d<rhocount, xcount, ycount, RealT>(FFTW_REDFT00);
    std::vector<viennacl::matrix<RealT>> GPUMatrices;
    viennacl::vector<RealT> GPUSource;
    viennacl::vector<RealT> GPUTarget;

   public:
    friend class GACalculator<RealT>;

    explicit chebGPUDense(const gridDomain &g, const functionGrid<RealT> &f)
        : plan(f.rhocount, f.xcount, f.ycount, FFTW_REDFT00),
          GPUMatrices(f.rhocount, viennacl::matrix<RealT>(f.xcount * f.ycount, f.xcount * f.ycount)),
          GPUSource(f.xcount * f.ycount),
          GPUTarget(f.xcount * f.ycount) {
        std::vector<Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic>> denseGAMatrix(f.rhocount);
        OOGA::chebCPUDense<RealT>::fastFill(g, f, denseGAMatrix);

        for (size_t r = 0; r < GPUMatrices.size(); ++r) {
            viennacl::copy(denseGAMatrix[r], GPUMatrices[r]);
        }

        viennacl::backend::finish();
    }

    static std::unique_ptr<GACalculator<RealT>>
    create(const gridDomain &g, const functionGrid<RealT> &f) {
        return std::make_unique<chebGPUDense>(g, f);
    }

    functionGrid<RealT> calculate(
        functionGrid<RealT> &f) override {
        functionGrid<RealT> m = f;

        std::copy(f.gridValues.data.begin(), f.gridValues.data.begin() + f.rhocount * f.xcount * f.ycount, plan.fftin);
        plan.execute();
        for (int rho_iter = 0; rho_iter < f.rhocount; ++rho_iter) {
            Eigen::Map<Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic>> X(plan.fftout + rho_iter * f.xcount * f.ycount, f.xcount * f.ycount, 1);
            X *= (1.0 / ((f.xcount - 1) * (f.ycount - 1)));
            viennacl::copy(plan.fftout + rho_iter * f.xcount * f.ycount, plan.fftout + (rho_iter + 1) * f.xcount * f.ycount, GPUSource.begin());
            viennacl::backend::finish();
            GPUTarget = viennacl::linalg::prod(GPUMatrices[rho_iter], GPUSource);

            viennacl::backend::finish();

            copy(GPUTarget.begin(), GPUTarget.end(), m.gridValues.data.data() + rho_iter * f.xcount * f.ycount);
        }
        return m;
    }
};

//chebGPU

template <class RealT = double>
void inline arcIntegralBicubic(
    std::array<RealT, 16> &coeffs,  // this function is being left in double,
                                    // intentionally, for now
    RealT rho, RealT xc, RealT yc, RealT s0, RealT s1) {
    // we are going to write down indefinite integrals of (xc + rho*sin(x))^i
    // (xc-rho*cos(x))^j
    // it seems like in c++, we can't make a 2d-array of lambdas that bind (rho,
    // xc, yc)
    // without paying a performance penalty.
    // The below is not meant to be so human readable.  It was generated partly
    // by mathematica.

    // crazy late-night thought - can this whole thing be constexpr?
    const RealT c = xc;
    const RealT d = yc;
    const RealT r = rho;
    const RealT r2 = rho * rho;
    const RealT r3 = r2 * rho;
    const RealT r4 = r2 * r2;
    const RealT r5 = r3 * r2;

    const RealT c2 = c * c;
    const RealT c3 = c2 * c;

    const RealT d2 = d * d;
    const RealT d3 = d2 * d;

    using std::cos;
    using std::sin;

#define f00(x) ((x))

#define f10(x) (c * (x)-r * cos((x)))

#define f20(x) (c2 * (x)-2 * c * r * cos((x)) + r2 * (x) / 2 - r2 * sin(2 * (x)) / 4)

#define f30(x)                                                                     \
    ((1.0 / 12.0) * (3 * c * (4 * c2 * (x) + 6 * r2 * (x)-3 * r2 * sin(2 * (x))) - \
                     9 * r * (4 * c2 + r2) * cos((x)) + r3 * cos(3 * (x))))

#define f01(x) (d * (x)-r * sin((x)))

#define f11(x) \
    (c * d * (x)-c * r * sin((x)) - d * r * cos((x)) + r2 * cos((x)) * cos((x)) / 2.0)

#define f21(x)                                                                 \
    ((1.0 / 12.0) *                                                            \
     (12.0 * c2 * d * (x)-12 * c2 * r * sin((x)) - 24 * c * d * r * cos((x)) + \
      6 * c * r2 * cos(2 * (x)) + 6 * d * r2 * (x)-3 * d * r2 * sin(2 * (x)) - \
      3 * r3 * sin((x)) + r3 * sin(3 * (x))))

#define f31(x)                                                                     \
    ((1.0 / 96.0) * (48 * c * d * (x) * (2 * c2 + 3 * r2) -                        \
                     72 * d * r * (4 * c2 + r2) * cos((x)) -                       \
                     24 * c * r * (4 * c2 + 3 * r2) * sin((x)) +                   \
                     12 * r2 * (6 * c2 + r2) * cos(2 * (x)) -                      \
                     72 * c * d * r2 * sin(2 * (x)) + 24 * c * r3 * sin(3 * (x)) + \
                     8 * d * r3 * cos(3 * (x)) - 3 * r4 * cos(4 * (x))))

#define f02(x) \
    ((1.0 / 2.0) * ((x) * (2 * d2 + r2) + r * sin((x)) * (r * cos((x)) - 4 * d)))

#define f12(x)                                                       \
    ((1.0 / 12.0) *                                                  \
     (6 * c * (x) * (2 * d2 + r2) - 24 * c * d * r * sin((x)) +      \
      3 * c * r2 * sin(2 * (x)) - 3 * r * (4 * d2 + r2) * cos((x)) + \
      6 * d * r2 * cos(2 * (x)) + r3 * (-1.0 * cos(3 * (x)))))

#define f22(x)                                                                     \
    ((1.0 / 96.0) * (24 * r2 * (c2 - d2) * sin(2 * (x)) +                          \
                     12 * (x) * (4 * c2 * (2 * d2 + r2) + 4 * d2 * r2 + r4) -      \
                     48 * d * r * (4 * c2 + r2) * sin((x)) -                       \
                     48 * c * r * (4 * d2 + r2) * cos((x)) +                       \
                     96 * c * d * r2 * cos(2 * (x)) - 16 * c * r3 * cos(3 * (x)) + \
                     16 * d * r3 * sin(3 * (x)) - 3 * r4 * sin(4 * (x))))

#define f32(x)                                                           \
    ((1.0 / 480.0) *                                                     \
     (120 * c * r2 * (c2 - 3 * d2) * sin(2 * (x)) +                      \
      60 * c * (x) * (4 * c2 * (2 * d2 + r2) + 3 * (4 * d2 * r2 + r4)) - \
      60 * r * cos((x)) * (6 * c2 * (4 * d2 + r2) + 6 * d2 * r2 + r4) -  \
      10 * r3 * cos(3 * (x)) * (12 * c2 - 4 * d2 + r2) -                 \
      240 * c * d * r * (4 * c2 + 3 * r2) * sin((x)) +                   \
      120 * d * r2 * (6 * c2 + r2) * cos(2 * (x)) +                      \
      240 * c * d * r3 * sin(3 * (x)) - 45 * c * r4 * sin(4 * (x)) -     \
      30 * d * r4 * cos(4 * (x)) + 6 * r5 * cos(5 * (x))))

#define f03(x)                                                             \
    ((1.0 / 12.0) *                                                        \
     (12 * d3 * (x)-9 * r * (4 * d2 + r2) * sin((x)) + 18 * d * r2 * (x) + \
      9 * d * r2 * sin(2 * (x)) - r3 * sin(3 * (x))))

#define f13(x)                                                                 \
    ((1.0 / 96.0) *                                                            \
     (48 * c * d * (x) * (2 * d2 + 3 * r2) -                                   \
      72 * c * r * (4 * d2 + r2) * sin((x)) + 72 * c * d * r2 * sin(2 * (x)) - \
      8 * c * r3 * sin(3 * (x)) + 12 * r2 * (6 * d2 + r2) * cos(2 * (x)) -     \
      24 * d * r * (4 * d2 + 3 * r2) * cos((x)) - 24 * d * r3 * cos(3 * (x)) + \
      3 * r4 * cos(4 * (x))))

#define f23(x)                                                                                                                                                \
    ((1.0 / 480.0) *                                                                                                                                          \
     (480 * c2 * d3 * (x)-1440 * c2 * d2 * r * sin((x)) +                                                                                                     \
      720 * c2 * d * r2 * (x) + 360 * c2 * d * r2 * sin(2 * (x)) -                                                                                            \
      360 * c2 * r3 * sin((x)) - 40 * c2 * r3 * sin(3 * (x)) +                                                                                                \
      120 * c * r2 * (6 * d2 + r2) * cos(2 * (x)) -                                                                                                           \
      240 * c * d * r * (4 * d2 + 3 * r2) * cos((x)) -                                                                                                        \
      240 * c * d * r3 * cos(3 * (x)) + 30 * c * r4 * cos(4 * (x)) +                                                                                          \
      240 * d3 * r2 * (x)-120 * d3 * r2 * sin(2 * (x)) -                                                                                                      \
      360 * d2 * r3 * sin((x)) + 120 * d2 * r3 * sin(3 * (x)) + 180 * d * r4 * (x)-45 * d * r4 * sin(4 * (x)) - 60 * r5 * sin((x)) + 10 * r5 * sin(3 * (x)) + \
      6 * r5 * sin(5 * (x))))

#define f33(x)                                                               \
    ((1.0 / 960.0) *                                                         \
     (960 * c3 * d3 * (x)-2880 * c3 * d2 * r * sin((x)) +                    \
      1440 * c3 * d * r2 * (x) + 720 * c3 * d * r2 * sin(2 * (x)) -          \
      720 * c3 * r3 * sin((x)) - 80 * c3 * r3 * sin(3 * (x)) +               \
      45 * r2 * cos(2 * (x)) * (8 * c2 * (6 * d2 + r2) + 8 * d2 * r2 + r4) - \
      360 * d * r * cos((x)) * (c2 * (8 * d2 + 6 * r2) + 2 * d2 * r2 + r4) - \
      720 * c2 * d * r3 * cos(3 * (x)) + 90 * c2 * r4 * cos(4 * (x)) +       \
      1440 * c * d3 * r2 * (x)-720 * c * d3 * r2 * sin(2 * (x)) -            \
      2160 * c * d2 * r3 * sin((x)) + 720 * c * d2 * r3 * sin(3 * (x)) +     \
      1080 * c * d * r4 * (x)-270 * c * d * r4 * sin(4 * (x)) -              \
      360 * c * r5 * sin((x)) + 60 * c * r5 * sin(3 * (x)) +                 \
      36 * c * r5 * sin(5 * (x)) + 80 * d3 * r3 * cos(3 * (x)) -             \
      90 * d2 * r4 * cos(4 * (x)) - 60 * d * r5 * cos(3 * (x)) +             \
      36 * d * r5 * cos(5 * (x)) - 5 * r5 * r * cos(6 * (x))))

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

//FFTPADDED

template <class RealT = double>
class bicubicDotProductCPU
    : public GACalculator<RealT> {
   private:
    Eigen::SparseMatrix<RealT, Eigen::RowMajor> BCSparseOperator;

   public:
    friend class GACalculator<RealT>;
    explicit bicubicDotProductCPU(const functionGrid<RealT> &f) {
        BCSparseOperator = assembleFastBCCalc(f);
    };

    static std::unique_ptr<GACalculator<RealT>>
    create(const functionGrid<RealT> &f) {
        return std::make_unique<bicubicDotProductCPU>(f);
    }

    static Eigen::SparseMatrix<RealT, Eigen::RowMajor> assembleFastBCCalc(const functionGrid<RealT> &ftocopy) {
        auto f = ftocopy;
        Eigen::SparseMatrix<RealT, Eigen::RowMajor> BCOffsetTensor;
        BCOffsetTensor.setZero();
        const int myxcount = f.xcount;
        const int myycount = f.ycount;
        //auto ref = functionGrid<RealT>::bicubicParameterGrid::internalRef;
        auto ref2 = [myxcount, myycount](int x, int y, int z, int t) -> int { return x * myxcount * myycount * 16 + y * myycount * 16 + z * 16 + t; };
        //inline int internalRef(int x, int y, int z, int t) {
        //    return x * h * d * l + y * d * l + z * l + t;
        //}
        std::vector<std::vector<Eigen::Triplet<RealT>>>
            TripletVecVec(f.rhocount * f.xcount);
        for (auto i = 0; i < f.rhocount; i++) {
#pragma omp parallel for
            for (auto j = 0; j < f.xcount; j++) {
                for (auto k = 0; k < f.ycount; k++) {
                    std::vector<indexedPoint<RealT>> intersections;
                    RealT rho = f.rhoset[i];
                    RealT xc = f.xset[j];
                    RealT yc = f.yset[k];
                    RealT xmin = f.xset[0], xmax = f.xset.back();
                    RealT ymin = f.yset[0], ymax = f.yset.back();
                    std::vector<RealT> xIntersections, yIntersections;
                    // these two loops calculate all potential intersection points
                    // between GA circle and the grid.
                    for (auto v : f.xset) {
                        if (std::abs(v - xc) <= rho) {
                            RealT deltax = v - xc;
                            RealT deltay = std::sqrt(rho * rho - deltax * deltax);
                            if ((yc + deltay >= ymin) && (yc + deltay <= ymax)) {
                                intersections.push_back(
                                    indexedPoint<RealT>(v, yc + deltay, 0));
                            }
                            if ((yc - deltay >= ymin) && (yc - deltay <= ymax)) {
                                intersections.push_back(
                                    indexedPoint<RealT>(v, yc - deltay, 0));
                            }
                        }
                    }
                    for (auto v : f.yset) {
                        if (std::abs(v - yc) <= rho) {
                            RealT deltay = v - yc;
                            RealT deltax = std::sqrt(rho * rho - deltay * deltay);
                            if ((xc + deltax >= xmin) && (xc + deltax <= xmax)) {
                                intersections.push_back(
                                    indexedPoint<RealT>(xc + deltax, v, 0));
                            }
                            if ((xc - deltax >= xmin) && (xc - deltax <= xmax)) {
                                intersections.push_back(
                                    indexedPoint<RealT>(xc - deltax, v, 0));
                            }
                        }
                    }
                    for (auto &v : intersections) {
                        v.s = std::atan2(v.xvalue - xc, yc - v.yvalue);
                        if (v.s < 0) {
                            v.s += 2 * pi;
                        }
                        assert((0 <= v.s) && (v.s < 2 * pi));
                        assert((xc + std::sin(v.s) * rho) - v.xvalue < 1e-10);
                        assert((yc - std::cos(v.s) * rho) - v.yvalue < 1e-10);
                    }
                    indexedPoint<RealT> temp(0, 0, 0);
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
                        std::array<RealT, 16> coeffs = {};
                        int xInterpIndex = 0, yInterpIndex = 0;
                        if (s1 - s0 <
                            1e-12) {
                            // TODO(orebas) MAGIC NUMBER (here and another place )
                            continue;  // if two of our points are equal or very
                                       // close to one another, we make the arc
                                       // larger. this will probably happen for s=0
                                       // and s=pi/2, but doesn't cost us anything.
                        }
                        integrand(rho, xc, yc, (s0 + s1) / 2, xmid,
                                  ymid);  // this just calculates into (xmid,ymid)
                                          // the point half through the arc.
                        arcIntegralBicubic(coeffs, rho, xc, yc, s0, s1);
                        f.interpIndexSearch(xmid, ymid, xInterpIndex, yInterpIndex);
                        // begin look-thru code
                        if (!((xInterpIndex == (f.xcount - 1)) &&
                              (yInterpIndex == (f.ycount - 1)))) {
                            std::array<int, 16> LTSources = {}, LTTargets = {};
                            std::array<RealT, 16> LTCoeffs = {};
                            for (int l = 0; l < 16; l++) {
                                LTSources[l] =
                                    ref2(i, xInterpIndex, yInterpIndex, l) - ref2(0, 0, 0, 0);
                                LTTargets[l] =
                                    &(f.gridValues(i, j, k)) - &(f.gridValues(0, 0, 0));
                                LTCoeffs[l] = coeffs[l] / (2.0 * pi);

                                TripletVecVec[i * f.xcount + j].emplace_back(
                                    Eigen::Triplet<RealT>(LTTargets[l], LTSources[l], LTCoeffs[l]));
                            }
                        }
                    }
                }
            }
        }
        std::vector<Eigen::Triplet<RealT>> Triplets;
        for (int i = 0; i < f.rhocount * f.xcount; i++) {
            for (auto iter = TripletVecVec[i].begin();
                 iter != TripletVecVec[i].end(); ++iter) {
                //Eigen::Triplet<RealT> lto(*iter);
                Triplets.emplace_back(*iter);
            }
        }
        BCOffsetTensor.resize(f.rhocount * f.xcount * f.ycount,
                              f.rhocount * f.xcount * f.ycount * 16);
        BCOffsetTensor.setFromTriplets(Triplets.begin(), Triplets.end());

        //std::cout << "Number of RealT  products needed for BC calc: " << BCOffsetTensor.nonZeros() << " and rough memory usage is " << BCOffsetTensor.nonZeros() * (sizeof(RealT) + sizeof(long)) << std::endl;
        return BCOffsetTensor;
    }

    functionGrid<RealT> calculate(
        functionGrid<RealT> &f) override {
        functionGrid<RealT> m =
            f;
        m.clearGrid();
        typename functionGrid<RealT>::bicubicParameterGrid b = functionGrid<RealT>::setupBicubicGrid(f);

        Eigen::Map<Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic>> source(b.data.data(), f.rhocount * f.xcount * f.ycount * 16, 1);
        Eigen::Map<Eigen::Matrix<RealT, Eigen::Dynamic, Eigen::Dynamic>> target(m.gridValues.data.data(), f.rhocount * f.xcount * f.ycount, 1);

        target = BCSparseOperator * source;
        return m;
    }
};

// GPU Bicubic DP
template <class RealT = double>
class bicubicDotProductGPU
    : public GACalculator<RealT> {
   private:
    viennacl::compressed_matrix<RealT> vcl_sparse_matrix;

   public:
    friend class GACalculator<RealT>;
    explicit bicubicDotProductGPU(const functionGrid<RealT> &f)
        : vcl_sparse_matrix(f.xcount * f.ycount * f.rhocount, f.xcount * f.ycount * f.rhocount * 16) {
        Eigen::SparseMatrix<RealT, Eigen::RowMajor> BCSparseOperator = bicubicDotProductCPU<RealT>::assembleFastBCCalc(f);
        viennacl::copy(BCSparseOperator, vcl_sparse_matrix);
        typename functionGrid<RealT>::bicubicParameterGrid b = functionGrid<RealT>::setupBicubicGrid(f);
        //gpu_source.resize(b.data.size());
        //gpu_target.resize(f.gridValues.data.size());
        //gpu_source.clear();
        //gpu_target.clear();
        //        std::cout << "no fail 1" << std::endl;
        //auto garbage = this->calculate(f);  //call to initialize compute kernel maybe?
        //TODO(orebas): the above may or not be a virtual function called in a constructor.  those are bad. google it.
        viennacl::backend::finish();
    };

    static std::unique_ptr<GACalculator<RealT>>
    create(const functionGrid<RealT> &f) {
        return std::make_unique<bicubicDotProductGPU>(f);
    }

    functionGrid<RealT> calculate(
        functionGrid<RealT> &f) override {
        functionGrid<RealT> m = f;

        typename functionGrid<RealT>::bicubicParameterGrid b = functionGrid<RealT>::setupBicubicGrid(f);
        viennacl::vector<RealT> gpu_source(b.data.size());
        viennacl::copy(b.data.begin(), b.data.end(), gpu_source.begin());

        viennacl::backend::finish();
        viennacl::vector<RealT> gpu_target = viennacl::linalg::prod(vcl_sparse_matrix, gpu_source);

        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       m.gridValues.data.begin());
        viennacl::backend::finish();
        return m;
    }
};
// GPU Bicubic DP

// GPU Linear DP
template <class RealT = double>
class linearDotProductGPU
    : public GACalculator<RealT> {
   private:
    viennacl::compressed_matrix<RealT> vcl_sparse_matrix;

   public:
    friend class GACalculator<RealT>;
    explicit linearDotProductGPU(const functionGrid<RealT> &f)
        : vcl_sparse_matrix(f.xcount * f.ycount * f.rhocount, f.xcount * f.ycount * f.rhocount) {
        Eigen::SparseMatrix<RealT, Eigen::RowMajor> GASparseOperator = linearDotProductCPU<RealT>::assembleFastGACalc(f);
        viennacl::copy(GASparseOperator, vcl_sparse_matrix);
        //typename functionGrid<RealT> ::linearParameterGrid b = functionGrid<RealT> ::setuplinearGrid(f);
        //std::cout << "no fail 1" << std::endl;
        //auto garbage = this->calculate(f);  //call to initialize compute kernel maybe?
        //TODO(orebas): virtual function call in a constructor is bad.

        viennacl::backend::finish();
    };

    static std::unique_ptr<GACalculator<RealT>>
    create(const functionGrid<RealT> &f) {
        return std::make_unique<linearDotProductGPU>(f);
    }

    functionGrid<RealT> calculate(
        functionGrid<RealT> &f) override {
        functionGrid<RealT> m =
            f;

        viennacl::vector<RealT> gpu_source(f.gridValues.data.size());
        viennacl::copy(f.gridValues.data.begin(), f.gridValues.data.end(), gpu_source.begin());

        viennacl::backend::finish();
        viennacl::vector<RealT> gpu_target = viennacl::linalg::prod(vcl_sparse_matrix, gpu_source);

        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       m.gridValues.data.begin());
        viennacl::backend::finish();
        return m;
    }
};
// GPU Linear DP

template <class RealT>
std::unique_ptr<GACalculator<RealT>>
GACalculator<RealT>::Factory::newCalculator(
    calculatorType c, const gridDomain &g, functionGrid<RealT> &f, int padcount) {
    switch (c) {
        case (calculatorType::linearCPU):
            return linearCPUCalculator<RealT>::create();
        case (calculatorType::bicubicCPU):
            return bicubicCPUCalculator<RealT>::create();
        case (calculatorType::DCTCPUCalculator):
            assert(0);
            //return DCTCPUCalculator< RealT>::create(g);
        case (calculatorType::DCTCPUCalculator2):
            return DCTCPUCalculator2<RealT>::create(g, f);
        case (calculatorType::linearDotProductCPU):
            return linearDotProductCPU<RealT>::create(f);
        case (calculatorType::bicubicDotProductCPU):
            return bicubicDotProductCPU<RealT>::create(f);
        case (calculatorType::DCTCPUPaddedCalculator2):
            return DCTCPUPaddedCalculator<RealT>::create(g, f, padcount);
        case (calculatorType::bicubicDotProductGPU):
            return bicubicDotProductGPU<RealT>::create(f);
        case (calculatorType::linearDotProductGPU):
            return linearDotProductGPU<RealT>::create(f);
        case (calculatorType::chebCPUDense):
            return chebCPUDense<RealT>::create(g, f);
        case (calculatorType::chebGPUDense):
            return chebGPUDense<RealT>::create(g, f);

        default:
            return linearCPUCalculator<RealT>::create();
    }
}

}  // namespace OOGA

namespace OOGA {

template <class RealT = double>
void testArcIntegralBicubic() {
    constexpr RealT r = 0.3, s0 = 0.6, s1 = 2.2, xc = -0.2, yc = -1.4;
    std::array<RealT, 16> coeffs;
    arcIntegralBicubic(coeffs, r, xc, yc, s0, s1);
    for (int i = 0; i <= 3; ++i) {
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
}

}  // namespace OOGA

#endif  // GYROAVERAGING_GA_H

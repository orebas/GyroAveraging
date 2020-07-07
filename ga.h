

// file foobar.h:
#ifndef GYROAVERAGING_GA_H
#define GYROAVERAGING_GA_H
// ... declarations ...

#include <algorithm>
#include <array>
#include <boost/math/special_functions/bessel.hpp>  //remove when you remove testing code.
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <vector>

#include "gautils.h"

namespace OOGA {

template <class RealT = double>
struct gridDomain {
    RealT xmin = 0, xmax = 0, ymin = 0, ymax = 0, rhomin = 0, rhomax = 0;
};

template <int rhocount, int xcount, int ycount, class RealT = double>
class functionGrid {
   public:
    typedef Array3d<rhocount, xcount, ycount, RealT> fullgrid;
    typedef Array4d<rhocount, xcount, ycount, 4, RealT> fullgridInterp;
    typedef Array4d<rhocount, xcount, ycount, 4, RealT>
        derivsGrid;  // at each rho,x,y calculate [f,f_x,f_y,f_xy]
    typedef Array4d<rhocount, xcount, ycount, 16, RealT> bicubicParameterGrid;
    typedef Array4d<rhocount, xcount, ycount, 9, RealT> biquadParameterGrid;
    typedef Eigen::SparseMatrix<RealT, Eigen::RowMajor> SpM;
    typedef Eigen::Triplet<RealT> SpT;

   public:  // change to private later.
    std::vector<RealT> rhoset;
    std::vector<RealT> xset;
    std::vector<RealT> yset;
    fullgrid gridValues;  // input values of f

   public:
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
                std::cout << gridValues(rho, j, k) - n.gridValues(rho, j, k)
                          << ",";
            }
            std::cout << std::endl;
        }
    }

    void clearGrid() {
        for (auto i = 0; i < rhocount; i++) {
            for (auto j = 0; j < xcount; j++)
                for (auto k = 0; k < ycount; k++) {
                    gridValues(i, j, k) = 0;
                }
        }
    }
    RealT RMSNorm(int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++)
                result += gridValues(rho, j, k) * gridValues(rho, j, k);
        return std::sqrt(result / (xcount * ycount));
    }
    RealT maxNorm(int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++)
                result = std::max(result, std::abs(gridValues(rho, j, k)));
        return result;
    }

    RealT RMSNormDiff(const fullgrid &m2, int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++) {
                RealT t = gridValues(rho, j, k) - m2(rho, j, k);
                result += t * t;
            }
        return std::sqrt(result / (xcount * ycount));
    }
    RealT maxNormDiff(const fullgrid &m2, int rho) {
        RealT result = 0;
        for (int j = 0; j < xcount; j++)
            for (int k = 0; k < ycount; k++) {
                RealT t = gridValues(rho, j, k) - m2(rho, j, k);
                result = std::max(result, std::abs(t));
            }
        return result;
    }

    // below fills a grid, given a function of rho, x, and y
    template <typename TFunc>
    void fill(TFunc f) {
#pragma omp parallel for
        for (auto i = 0; i < rhocount; i++) {
            for (auto j = 0; j < xcount; j++)
                for (auto k = 0; k < ycount; k++) {
                    gridValues(i, j, k) = f(rhoset[i], xset[j], yset[k]);
                }
        }
    }
    // below fills a grid, given a function of i,j,k
    template <typename TFunc>
    void fillbyindex(TFunc f) {
#pragma omp parallel for
        for (int i = 0; i < rhocount; i++)
            for (int j = 0; j < xcount; j++)
                for (int k = 0; k < ycount; k++) {
                    gridValues(i, j, k) = f(i, j, k);
                }
    }
    template <typename TFunc1>
    void fillAlmostExactGA(TFunc1 f) {                   // f is only used to fill the rho=0 case
        fillbyindex([&](int i, int j, int k) -> RealT {  // adaptive trapezoid rule on actual input function.
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
    void fillTruncatedAlmostExactGA(TFunc1 f) {
        fillbyindex([&](int i, int j, int k) -> RealT {
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
        TFunc1 f) {  // this calls interp2d. gridValues must be
                     // filled, but we don't need setupInterpGrid.
        fillbyindex([&](int i, int j, int k) -> RealT {
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
    void fillBicubicInterp(void) {
        fillbyindex([&](int i, int j, int k) -> RealT {
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
    functionGrid(const std::vector<RealT> &rhos, const std::vector<RealT> &xes,
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

    inline void integrand(const RealT rho, const RealT xc, const RealT yc,
                          const RealT gamma, RealT &xn, RealT &yn) {
        xn = xc + rho * std::sin(gamma);
        yn = yc - rho * std::cos(gamma);
    }

    static bicubicParameterGrid setupBicubicGrid(
        const functionGrid<rhocount, xcount, ycount, RealT> &f) {
        using namespace Eigen;
        auto d = f.calcDerivsGrid();
        bicubicParameterGrid b;
        for (int i = 0; i < rhocount; i++) {
            // we explicitly rely on parameters being initialized to 0,
            // including the top and right sides.
            for (int j = 0; j < xcount - 1; j++)
                for (int k = 0; k < ycount - 1; k++) {
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
                    // Y.fullPivLu().inverse(); // we should take out the Eigen
                    // dependency methinks.  TODO

                    A = X.inverse() * RHS * Y.inverse();
                    for (int t = 0; t < 16; ++t)
                        b(i, j, k, t) = A(t % 4, t / 4);
                }
        }
        return b;
    }

    derivsGrid calcDerivsGrid() const {
        RealT ydenom = yset[1] - yset[0];
        RealT xdenom = xset[1] - xset[0];
        const fullgrid &g = gridValues;
        derivsGrid derivs;
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
    // void setupInterpGrid();
    // void setupDerivsGrid();  // make accuracy a variable later
    // void setupBicubicGrid();
    // void setupBicubicGrid2();
    // void assembleFastGACalc(void);

    // void assembleFastBCCalc(void);
    // void fastLTCalcOffset();
    // void fastBCCalcOffset();
    std::array<RealT, 4> arcIntegral(RealT rho, RealT xc, RealT yc, RealT s0,
                                     RealT s1);
    // std::array<RealT, 16> arcIntegralBicubic(RealT rho, RealT xc, RealT yc,
    // RealT s0, RealT s1);

    // RealT interp2d(int rhoindex, const RealT x, const RealT y);
    // RealT interpNaiveBicubic(int rhoindex, const RealT x, const RealT y);
    // RealT interpNaiveBicubic2(int rhoindex, const RealT x, const RealT y);

    // friend void testInterpImprovement();
};

/* GACalculator API
Create
Calculate
Name
Short Description
Long Description
Destroy/Free
*/
enum class calculatorType { linearCPU,
                            linearDotProductCPU,
                            bicubicCPU,
                            bicubicDotProductCPU,
                            DCTCPUCalculator,
                            DCTCPUCalculator2,
                            DCTCPUPaddedCalculator2,
                            bicubicDotProductGPU,
                            linearDotProductGPU };

template <int rhocount, int xcount, int ycount, class RealT = double, int padcount = 0>
class GACalculator {
   public:
    struct Factory {
        static std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
        newCalculator(calculatorType c, gridDomain<RealT> &g, functionGrid<rhocount, xcount, ycount, RealT> &f);
    };
    virtual ~GACalculator() = default;
    virtual functionGrid<rhocount, xcount, ycount, RealT> calculate(
        functionGrid<rhocount, xcount, ycount, RealT> &f) = 0;
};

template <int rhocount, int xcount, int ycount, class RealT = double>
class linearCPUCalculator
    : public GACalculator<rhocount, xcount, ycount, RealT> {
   public:
    friend class GACalculator<rhocount, xcount, ycount, RealT>;
    linearCPUCalculator() = default;
    ~linearCPUCalculator() = default;
    static std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
    create() {
        return std::make_unique<linearCPUCalculator>();
    }
    std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>> clone() {
        return std::make_unique<linearCPUCalculator>(*this);
    } /*maybe disable this functionality */
    functionGrid<rhocount, xcount, ycount, RealT> calculate(
        functionGrid<rhocount, xcount, ycount, RealT> &f) {
        functionGrid<rhocount, xcount, ycount, RealT> m =
            f;  // do we need to write a copy constructor?

        m.fillbyindex([&](int i, int j, int k) -> RealT {
            RealT xc = f.xset[j];
            RealT yc = f.yset[k];
            if (f.rhoset[i] == 0) return f.gridValues(i, j, k);
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

template <int rhocount, int xcount, int ycount, class RealT = double>
class linearDotProductCPU
    : public GACalculator<rhocount, xcount, ycount, RealT> {
   private:
    Eigen::SparseMatrix<RealT, Eigen::RowMajor> LTOffsetTensor;

   public:
    friend class GACalculator<rhocount, xcount, ycount, RealT>;
    linearDotProductCPU(const gridDomain<double> &g, functionGrid<rhocount, xcount, ycount, RealT> &f) {
        assembleFastGACalc(g, f);
    };
    ~linearDotProductCPU(){

    };
    static std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
    create(const gridDomain<double> &g, functionGrid<rhocount, xcount, ycount, RealT> &f) {
        return std::make_unique<linearDotProductCPU>(g, f);
    }
    std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>> clone() {
        return std::make_unique<linearDotProductCPU>(*this);
    } /*maybe disable this functionality */

    void assembleFastGACalc(const gridDomain<double> &g, functionGrid<rhocount, xcount, ycount, RealT> &f) {
        LTOffsetTensor.setZero();
        std::vector<std::vector<Eigen::Triplet<RealT>>>
            TripletVecVec(rhocount);
#pragma omp parallel for
        for (auto i = 0; i < rhocount; i++) {
            for (auto j = 0; j < xcount; j++)
                for (auto k = 0; k < ycount; k++) {
                    std::vector<indexedPoint<RealT>> intersections;
                    RealT rho = f.rhoset[i];
                    RealT xc = f.xset[j];
                    RealT yc = f.yset[k];
                    RealT xmin = f.xset[0], xmax = f.xset.back();
                    RealT ymin = f.yset[0], ymax = f.yset.back();
                    std::vector<RealT> xIntersections, yIntersections;
                    // these two loops calculate all potential intersection points
                    // between GA circle and the grid.
                    for (auto v : f.xset)
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
                    for (auto v : f.yset)
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
                        f.interpIndexSearch(xmid, ymid, xInterpIndex, yInterpIndex);

                        if (!((xInterpIndex == (xcount - 1)) &&
                              (yInterpIndex == (ycount - 1)))) {
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
                                TripletVecVec[i].emplace_back(
                                    Eigen::Triplet<RealT>(LTTargets[l], LTSources[l], LTCoeffs[l]));
                            }
                        }
                    }
                }
        }
        std::vector<Eigen::Triplet<RealT>> Triplets;

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

    functionGrid<rhocount, xcount, ycount, RealT> calculate(
        functionGrid<rhocount, xcount, ycount, RealT> &f) {
        functionGrid<rhocount, xcount, ycount, RealT> m =
            f;  // do we need to write a copy constructor?

        m.clearGrid();
        Eigen::Map<Eigen::Matrix<RealT, rhocount * xcount * ycount, 1>> source(
            f.gridValues.data.data());
        Eigen::Map<Eigen::Matrix<RealT, rhocount * xcount * ycount, 1>> target(
            m.gridValues.data.data());
        target = LTOffsetTensor * source;
        return m;
    }
};

template <int rhocount, int xcount, int ycount, class RealT = double>
class bicubicCPUCalculator
    : public GACalculator<rhocount, xcount, ycount, RealT> {
    typename functionGrid<rhocount, xcount, ycount, RealT>::bicubicParameterGrid bicubicParamaters;

   public:
    friend class GACalculator<rhocount, xcount, ycount, RealT>;
    bicubicCPUCalculator() = default;

    ~bicubicCPUCalculator() = default;
    static std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
    create() {
        return std::make_unique<bicubicCPUCalculator>();
    }
    std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>> clone() {
        return std::make_unique<bicubicCPUCalculator>(*this);
    } /*maybe disable this functionality */

    RealT interpNaiveBicubic(const functionGrid<rhocount, xcount, ycount, RealT> &f,
                             const typename functionGrid<rhocount, xcount, ycount, RealT>::bicubicParameterGrid &b,
                             int rhoindex, const RealT x, const RealT y) const {
        assert((rhoindex >= 0) && (rhoindex < rhocount));
        if ((x <= f.xset[0]) || (y <= f.yset[0]) || (x >= f.xset.back()) ||
            (y >= f.yset.back()))
            return 0;
        int xindex = 0, yindex = 0;
        RealT result = 0;
        f.interpIndexSearch(x, y, xindex, yindex);
        RealT xns[4] = {1, x, x * x, x * x * x};
        RealT yns[4] = {1, y, y * y, y * y * y};
        for (int i = 0; i <= 3; ++i)
            for (int j = 0; j <= 3; ++j) {
                result += b(rhoindex, xindex, yindex, j * 4 + i) *
                          xns[i] * yns[j];
            }
        return result;
    }

    functionGrid<rhocount, xcount, ycount, RealT> calculate(
        functionGrid<rhocount, xcount, ycount, RealT> &f) {
        functionGrid<rhocount, xcount, ycount, RealT> m = f;  // do we need to write a copy constructor?
        typename functionGrid<rhocount, xcount, ycount, RealT>::bicubicParameterGrid b = functionGrid<rhocount, xcount, ycount, RealT>::setupBicubicGrid(f);
        m.fillbyindex([&](int i, int j, int k) -> RealT {
            RealT xc = f.xset[j];
            RealT yc = f.yset[k];
            if (f.rhoset[i] == 0) return f.gridValues(i, j, k);
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
template <int rhocount, int xcount, int ycount, class RealT = double>
class biquadCPUCalculator
    : public GACalculator<rhocount, xcount, ycount, RealT> {
    typename functionGrid<rhocount, xcount, ycount, RealT>::biquadParameterGrid biquadParamaters;

   public:
    friend class GACalculator<rhocount, xcount, ycount, RealT>;
    biquadCPUCalculator() = default;

    ~biquadCPUCalculator() = default;
    static std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
    create() {
        return std::make_unique<biquadCPUCalculator>();
    }
    std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>> clone() {
        return std::make_unique<biquadCPUCalculator>(*this);
    } /*maybe disable this functionality */

    typename functionGrid<rhocount, xcount, ycount, RealT>::biquadParameterGrid setupBiquadGrid(
        const functionGrid<rhocount, xcount, ycount, RealT> &f) {
        using namespace Eigen;
        auto d = f.calcDerivsGrid();
        typename functionGrid<rhocount, xcount, ycount, RealT>::biquadParameterGrid b;
        for (int i = 0; i < rhocount; i++) {
            // we explicitly rely on parameters being initialized to 0,
            // including the top and right sides.
            for (int j = 0; j < xcount - 1; j++)
                for (int k = 0; k < ycount - 1; k++) {
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
                    // Y.fullPivLu().inverse(); // we should take out the Eigen
                    // dependency methinks.  TODO

                    A = X.inverse() * RHS * Y.inverse();
                    for (int t = 0; t < 16; ++t)
                        b(i, j, k, t) = A(t % 4, t / 4);
                }
        }
        return b;
    }

    RealT interpNaiveBiquad(const functionGrid<rhocount, xcount, ycount, RealT> &f,
                            const typename functionGrid<rhocount, xcount, ycount, RealT>::biquadParameterGrid &b,
                            int rhoindex, const RealT x, const RealT y) const {
        assert((rhoindex >= 0) && (rhoindex < f.rhocount));
        if ((x <= f.xset[0]) || (y <= f.yset[0]) || (x >= f.xset.back()) ||
            (y >= f.yset.back()))
            return 0;
        int xindex = 0, yindex = 0;
        RealT result = 0;
        f.interpIndexSearch(x, y, xindex, yindex);
        RealT xns[3] = {1, x, x * x};
        RealT yns[3] = {1, y, y * y};
        for (int i = 0; i <= 3; ++i)
            for (int j = 0; j <= 3; ++j) {
                result += b(rhoindex, xindex, yindex, j * 3 + i) *
                          xns[i] * yns[j];
            }
        return result;
    }

    functionGrid<rhocount, xcount, ycount, RealT> calculate(
        functionGrid<rhocount, xcount, ycount, RealT> &f) {
        functionGrid<rhocount, xcount, ycount, RealT> m = f;  // do we need to write a copy constructor?
        typename functionGrid<rhocount, xcount, ycount, RealT>::biquadParameterGrid b = this->setupBiquadGrid(f);
        m.fillbyindex([&](int i, int j, int k) -> RealT {
            RealT xc = f.xset[j];
            RealT yc = f.yset[k];
            if (f.rhoset[i] == 0) return f.gridValues(i, j, k);
            auto new_f = [&](RealT x) -> RealT {
                return this->interpNaiveBiquad(f, b, i, xc + f.rhoset[i] * std::sin(x),
                                               yc - f.rhoset[i] * std::cos(x));
            };
            RealT result = TrapezoidIntegrate(0.0, 2 * pi, new_f) / (2 * pi);
            return result;
        });
        return m;
    }
};

// FFT - we sort of abandoned this; it only computes the first rhoset[0].
//also the initialization is VERY slow.
template <int rhocount, int xcount, int ycount, class RealT = double>
class DCTCPUCalculator
    : public GACalculator<rhocount, xcount, ycount, RealT> {
   private:
    double *fftin, *fftout;
    fftw_plan plan;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> denseGAMatrix;

   public:
    friend class GACalculator<rhocount, xcount, ycount, RealT>;

    void slowTruncFill(const gridDomain<RealT> &g) {
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
                        double xint = (ex - g.xmin) / (g.xmax - g.xmin) * (xcount - 1);
                        double yint = (why - g.ymin) / (g.ymax - g.ymin) * (ycount - 1);
                        return DCTBasisFunction2(p, q, xint, yint, xcount);
                    };

                    f.fillTruncatedAlmostExactGA(basistest);
                    Eigen::Map<Eigen::Matrix<double, xcount * ycount, 1>> m(f.gridValues.data.data());
                    denseGAMatrix.row(ycount * p + q) = m;  //TODO NOT FINISHED REDO THIS LINE
                }
            }
        }
        denseGAMatrix.transposeInPlace();
        //std::cout << "dense matrix: \n " << denseGAMatrix << std::endl;
    }

    void fastFill(const gridDomain<RealT> &g) {
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
                        constexpr double N = xcount;
                        double xint = (ex - g.xmin) / (g.xmax - g.xmin) * (xcount - 1);
                        double yint = (why - g.ymin) / (g.ymax - g.ymin) * (ycount - 1);
                        double ap = 2, bq = 2;
                        if (p == 0) ap = 1;
                        if (q == 0) bq = 1;

                        double a, b, c, d;
                        a = pi * p * (2.0 * xint * (2 - 1) + 1) / (2.0 * N);  //change "2" to "N"
                        c = pi * q * (2.0 * yint * (2 - 1) + 1) / (2.0 * N);

                        d = pi * p * row * (N - 1) / (N * (g.xmax - g.xmin));
                        b = -pi * q * row * (N - 1) / (N * (g.ymax - g.ymin));
                        double j = boost::math::cyl_bessel_j(0, std::sqrt(b * b + d * d));
                        return ap * bq * std::cos(a) * std::cos(c) * j;
                    };

                    f.fill(besseltest);
                    Eigen::Map<Eigen::Matrix<double, xcount * ycount, 1>> m(f.gridValues.data.data());
                    denseGAMatrix.row(ycount * p + q) = m;  //TODO NOT FINISHED REDO THIS LINE
                }
            }
        }
        denseGAMatrix.transposeInPlace();
        //std::cout << "dense matrix: \n " << denseGAMatrix << std::endl;
    }

    DCTCPUCalculator(const gridDomain<RealT> &g) {
        fftin = (double *)fftw_malloc(xcount * ycount * sizeof(double));
        fftout = (double *)fftw_malloc(xcount * ycount * sizeof(double));
        plan = fftw_plan_r2r_2d(xcount, ycount, fftin, fftout, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);  //Forward DCT

        fastFill(g);
    }

    void DCTTest(const gridDomain<RealT> &g) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> tm, testm;  //"true matrix" vs "test matrix"

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
                    double xint = (ex - g.xmin) / (g.xmax - g.xmin) * (xcount - 1);
                    double yint = (why - g.ymin) / (g.ymax - g.ymin) * (ycount - 1);
                    return DCTBasisFunction2(p, q, xint, yint, xcount);
                };

                auto besseltest = [p, q, g](RealT row, RealT ex, RealT why) {
                    constexpr double N = xcount;
                    double xint = (ex - g.xmin) / (g.xmax - g.xmin) * (xcount - 1);
                    double yint = (why - g.ymin) / (g.ymax - g.ymin) * (ycount - 1);
                    double ap = 2, bq = 2;
                    if (p == 0) ap = 1;
                    if (q == 0) bq = 1;

                    double a, b, c, d;
                    a = pi * p * (2.0 * xint * (2 - 1) + 1) / (2.0 * N);  //change "2" to "N"
                    c = pi * q * (2.0 * yint * (2 - 1) + 1) / (2.0 * N);

                    d = pi * p * row * (N - 1) / (N * (g.xmax - g.xmin));
                    b = -pi * q * row * (N - 1) / (N * (g.ymax - g.ymin));
                    double j = boost::math::cyl_bessel_j(0, std::sqrt(b * b + d * d));
                    return ap * bq * std::cos(a) * std::cos(c) * j;
                };

                //f.fillTruncatedAlmostExactGA(basistest);  //test impact of truncation later.
                f.fillAlmostExactGA(basistest);
                ftest.fill(besseltest);
                Eigen::Map<Eigen::Matrix<double, xcount * ycount, 1>> m(f.gridValues.data.data());
                Eigen::Map<Eigen::Matrix<double, xcount * ycount, 1>> m2(ftest.gridValues.data.data());

                tm.row(ycount * p + q) = m;
                testm.row(ycount * p + q) = m2;
            }
        }
        tm.transposeInPlace();
        testm.transposeInPlace();
        /*std::cout << "True Matrix:" << std::endl
                  << tm << std::endl
                  << std::endl
                  << std::endl
                  << "Test Matrix:" << std::endl
                  << testm << std::endl
                  << std::endl
                  << std::endl
                  << "Truncated Matrix:" << std::endl
                  << denseGAMatrix << std::endl
                  << std::endl
                  << std::endl;*/
    }

    ~DCTCPUCalculator() {
        fftw_free(fftin);
        fftw_free(fftout);
    }
    static std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
    create(const gridDomain<RealT> &g) {
        return std::make_unique<DCTCPUCalculator>(g);
    }
    std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>> clone() {
        return std::make_unique<DCTCPUCalculator>(*this);
    } /*maybe disable this functionality */

    functionGrid<rhocount, xcount, ycount, RealT> calculate(
        functionGrid<rhocount, xcount, ycount, RealT> &f) {
        functionGrid<rhocount, xcount, ycount, RealT> m = f;  // do we need to write a copy constructor?

        { /* //testing code;
            const int p = 2, q = 3;
            gridDomain<double> g;
            g.rhomax = 0.25;
            g.rhomin = 1.55;
            g.xmin = g.ymin = -3;
            g.xmax = g.ymax = 3;
            auto basistest = [p, q, g](double row, double ex, double why) -> double {
                double xint = (ex - g.xmin) / (g.xmax - g.xmin) * xcount;
                double yint = (why - g.ymin) / (g.ymax - g.ymin) * ycount;
                return DCTBasisFunction2(p, q, xint, yint, xcount);
            };
            functionGrid<rhocount, xcount, ycount> in(f.rhoset, f.xset, f.yset), in2(f.rhoset, f.xset, f.yset), out(f.rhoset, f.xset, f.yset);
            in2.fill(basistest);
            fftw_plan plan2;
            plan2 = fftw_plan_r2r_2d(xcount, ycount, in2.gridValues.data.data(), out.gridValues.data.data(), FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);  //FORWARD "DCT"
            fftw_execute(plan2);
            for (int i = 0; i < xcount; ++i)
                for (int j = 0; j < ycount; ++j)
                    if (out.gridValues(0, i, j) > 0.01)
                        std::cout << p << " " << q << " " << i << " " << j << " " << out.gridValues(0, i, j) << std::endl;*/
        }

        std::copy(f.gridValues.data.begin(), f.gridValues.data.begin() + xcount * ycount + 1, fftin);
        fftw_execute(plan);
        Eigen::Map<Eigen::Matrix<double, xcount * ycount, 1>> X(fftout);
        //std::cout << "FFT result: \n " << X << std::endl;
        X *= (1.0 / (xcount * ycount * 4));

        Eigen::Map<Eigen::Matrix<double, xcount * ycount, 1>>
            b(m.gridValues.data.data());
        b = denseGAMatrix * X;
        //std::cout << std::endl
        //         << b << std::endl;
        return m;
    }
};  // namespace OOGA
//FFT
//FFT2
template <int rhocount, int xcount, int ycount, class RealT = double>
class DCTCPUCalculator2
    : public GACalculator<rhocount, xcount, ycount, RealT> {
   private:
    double *fftin, *fftout, *besselVals;
    gridDomain<RealT> g;
    fftw_plan plan;
    fftw_plan plan_inv;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> denseGAMatrix;

   public:
    friend class GACalculator<rhocount, xcount, ycount, RealT>;

    DCTCPUCalculator2(const gridDomain<RealT> &gd) {
        g = gd;
        fftin = (double *)fftw_malloc(rhocount * xcount * ycount * sizeof(double));
        fftout = (double *)fftw_malloc(rhocount * xcount * ycount * sizeof(double));
        besselVals = (double *)fftw_malloc(rhocount * xcount * ycount * sizeof(double));
        auto rhoset = LinearSpacedArray(g.rhomin, g.rhomax, rhocount);
        for (int r = 0; r < rhocount; r++)
            for (int p = 0; p < xcount; ++p) {
                for (int q = 0; q < ycount; ++q) {
                    constexpr double Nx = xcount;
                    constexpr double Ny = ycount;
                    double b, d;
                    d = pi * p * rhoset[r] * (Nx - 1) / (Nx * (g.xmax - g.xmin));
                    b = -pi * q * rhoset[r] * (Ny - 1) / (Ny * (g.ymax - g.ymin));
                    double j = boost::math::cyl_bessel_j(0, std::sqrt(b * b + d * d));
                    besselVals[r * xcount * ycount + ycount * p + q] = j / (xcount * ycount * 4.0);
                }
            }

        int rank = 2;
        int n[] = {xcount, ycount};
        int howmany = rhocount;
        int idist, odist, istride, ostride;
        idist = xcount * ycount;
        odist = xcount * ycount;
        istride = ostride = 1;
        fftw_r2r_kind fwd[] = {FFTW_REDFT10, FFTW_REDFT10};
        fftw_r2r_kind inv[] = {FFTW_REDFT01, FFTW_REDFT01};

        plan = fftw_plan_many_r2r(rank, n, howmany, fftin, n, istride, idist, fftout, n, ostride, odist, fwd, FFTW_MEASURE);
        plan_inv = fftw_plan_many_r2r(rank, n, howmany, fftout, n, istride, idist, fftin, n, ostride, odist, inv, FFTW_MEASURE);
        //plan = fftw_plan_r2r_2d(xcount, ycount, fftin, fftout, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);      //Forward DCT
        //plan_inv = fftw_plan_r2r_2d(xcount, ycount, fftout, fftin, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);  //Backward DCT
    }

    ~DCTCPUCalculator2() {
        fftw_free(fftin);
        fftw_free(fftout);
        fftw_free(besselVals);
    }
    static std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
    create(const gridDomain<RealT> &g) {
        return std::make_unique<DCTCPUCalculator2>(g);
    }
    std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>> clone() {
        return std::make_unique<DCTCPUCalculator2>(*this);
    } /*maybe disable this functionality */

    functionGrid<rhocount, xcount, ycount, RealT> calculate(
        functionGrid<rhocount, xcount, ycount, RealT> &f) {
        functionGrid<rhocount, xcount, ycount, RealT> m = f;  // do we need to write a copy constructor?

        std::copy(f.gridValues.data.begin(), f.gridValues.data.begin() + xcount * ycount * rhocount + 1, fftin);
        fftw_execute(plan);
        std::copy(fftout, fftout + rhocount * xcount * ycount + 1, m.gridValues.data.begin());
        for (int i = 0; i < rhocount * xcount * ycount; ++i) {
            fftout[i] *= besselVals[i];
        }
        fftw_execute(plan_inv);
        std::copy(fftin, fftin + rhocount * xcount * ycount + 1, m.gridValues.data.begin());

        return m;
    }
};
//FFT2
//FFTPADDED
template <int rhocount, int xcount, int ycount, int padcount = 0, class RealT = double>
class DCTCPUPaddedCalculator
    : public GACalculator<rhocount, xcount, ycount, RealT> {
   private:
    double *fftin, *fftout, *besselVals;
    gridDomain<RealT> paddedg;
    fftw_plan plan;
    fftw_plan plan_inv;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> denseGAMatrix;
    std::unique_ptr<DCTCPUCalculator2<rhocount, xcount + padcount * 2, ycount + padcount * 2, RealT>> dctCalc;

   public:
    friend class GACalculator<rhocount, xcount, ycount, RealT>;

    DCTCPUPaddedCalculator(const gridDomain<RealT> &g) {
        paddedg = g;
        double deltax = (g.xmax - g.xmin) / (xcount - 1.0);
        double deltay = (g.ymax - g.ymin) / (ycount - 1.0);
        paddedg.xmin = g.xmin - deltax * padcount;
        paddedg.xmax = g.xmax + deltax * padcount;
        paddedg.ymin = g.xmin - deltay * padcount;
        paddedg.ymax = g.xmax + deltay * padcount;
        dctCalc = std::make_unique<DCTCPUCalculator2<rhocount, xcount + padcount * 2, ycount + padcount * 2, RealT>>(paddedg);
    }

    ~DCTCPUPaddedCalculator() {
        dctCalc.reset(nullptr);
    }
    static std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
    create(const gridDomain<RealT> &g) {
        return std::make_unique<DCTCPUPaddedCalculator>(g);
    }
    std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>> clone() {
        return std::make_unique<DCTCPUPaddedCalculator>(*this);
    } /*maybe disable this functionality */

    functionGrid<rhocount, xcount, ycount, RealT> calculate(
        functionGrid<rhocount, xcount, ycount, RealT> &f) {
        functionGrid<rhocount, xcount, ycount, RealT> m = f;
        double xdelta = f.xset[1] - f.xset[0];
        double ydelta = f.yset[1] - f.yset[0];
        double xmax = f.xset.back();
        double ymax = f.yset.back();
        double xmin = f.xset[0];
        double ymin = f.yset[0];
        auto xnew = LinearSpacedArray(xmin - xdelta * padcount, xmin - xdelta, padcount);
        auto ynew = LinearSpacedArray(ymin - ydelta * padcount, ymin - ydelta, padcount);

        auto xPost = LinearSpacedArray(xmin + xdelta, xmax + xdelta * padcount, padcount);
        auto yPost = LinearSpacedArray(ymin + ydelta, ymax + ydelta * padcount, padcount);
        xnew.insert(xnew.end(), f.xset.begin(), f.xset.end());
        xnew.insert(xnew.end(), xPost.begin(), xPost.end());
        ynew.insert(ynew.end(), f.yset.begin(), f.yset.end());
        ynew.insert(ynew.end(), yPost.begin(), yPost.end());

        functionGrid<rhocount, xcount + padcount * 2, ycount + padcount * 2, RealT>
            paddedf(f.rhoset, xnew, ynew);
        for (int i = 0; i < rhocount; ++i)
            for (int j = 0; j < xcount; ++j)
                for (int k = 0; k < ycount; ++k) {
                    paddedf.gridValues(i, j + padcount, k + padcount) = f.gridValues(i, j, k);
                }
        auto paddedf2 = dctCalc->calculate(paddedf);
        /*std::cout << "Padded Result:" << std::endl;
        paddedf2.csvPrinter(0);
        std::cout << std::endl
                  << std::endl;*/
        for (int i = 0; i < rhocount; ++i)
            for (int j = 0; j < xcount; ++j)
                for (int k = 0; k < ycount; ++k) {
                    m.gridValues(i, j, k) = paddedf2.gridValues(i, j + padcount, k + padcount);
                }

        return m;
    }
};
//FFT2
//FFTPADDED

template <int rhocount, int xcount, int ycount, class RealT = double>
class bicubicDotProductCPU
    : public GACalculator<rhocount, xcount, ycount, RealT> {
   private:
    Eigen::SparseMatrix<RealT, Eigen::RowMajor> BCSparseOperator;

   public:
    friend class GACalculator<rhocount, xcount, ycount, RealT>;
    bicubicDotProductCPU(const gridDomain<RealT> &g, functionGrid<rhocount, xcount, ycount, RealT> &f) {
        BCSparseOperator = assembleFastBCCalc(g, f);
    };
    ~bicubicDotProductCPU(){

    };

    static std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
    create(const gridDomain<RealT> &g, functionGrid<rhocount, xcount, ycount, RealT> &f) {
        return std::make_unique<bicubicDotProductCPU>(g, f);
    }
    std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>> clone() {
        return std::make_unique<bicubicDotProductCPU>(*this);
    } /*maybe disable this functionality */

    static Eigen::SparseMatrix<RealT, Eigen::RowMajor> assembleFastBCCalc(const gridDomain<double> &g, functionGrid<rhocount, xcount, ycount, RealT> &f) {
        Eigen::SparseMatrix<RealT, Eigen::RowMajor> BCOffsetTensor;
        BCOffsetTensor.setZero();
        auto ref = functionGrid<rhocount, xcount, ycount, RealT>::bicubicParameterGrid::internalRef;
        std::vector<std::vector<Eigen::Triplet<RealT>>>
            TripletVecVec(rhocount * xcount);
        for (auto i = 0; i < rhocount; i++) {
#pragma omp parallel for
            for (auto j = 0; j < xcount; j++)
                for (auto k = 0; k < ycount; k++) {
                    std::vector<indexedPoint<RealT>> intersections;
                    RealT rho = f.rhoset[i];
                    RealT xc = f.xset[j];
                    RealT yc = f.yset[k];
                    RealT xmin = f.xset[0], xmax = f.xset.back();
                    RealT ymin = f.yset[0], ymax = f.yset.back();
                    std::vector<RealT> xIntersections, yIntersections;
                    // these two loops calculate all potential intersection points
                    // between GA circle and the grid.
                    for (auto v : f.xset)
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
                    for (auto v : f.yset)
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
                            1e-12)     // TODO MAGIC NUMBER (here and another place )
                            continue;  // if two of our points are equal or very
                                       // close to one another, we make the arc
                                       // larger. this will probably happen for s=0
                                       // and s=pi/2, but doesn't cost us anything.
                        integrand(rho, xc, yc, (s0 + s1) / 2, xmid,
                                  ymid);  // this just calculates into (xmid,ymid)
                                          // the point half through the arc.
                        arcIntegralBicubic(coeffs, rho, xc, yc, s0, s1);
                        f.interpIndexSearch(xmid, ymid, xInterpIndex, yInterpIndex);
                        // begin look-thru code
                        if (!((xInterpIndex == (xcount - 1)) &&
                              (yInterpIndex == (ycount - 1)))) {
                            std::array<int, 16> LTSources, LTTargets;
                            std::array<RealT, 16> LTCoeffs;
                            for (int l = 0; l < 16; l++) {
                                LTSources[l] =
                                    ref(i, xInterpIndex, yInterpIndex, l) - ref(0, 0, 0, 0);
                                LTTargets[l] =
                                    &(f.gridValues(i, j, k)) - &(f.gridValues(0, 0, 0));
                                LTCoeffs[l] = coeffs[l] / (2.0 * pi);

                                TripletVecVec[i * xcount + j].emplace_back(
                                    Eigen::Triplet<RealT>(LTTargets[l], LTSources[l], LTCoeffs[l]));
                            }
                        }
                    }
                }
        }
        std::vector<Eigen::Triplet<RealT>> Triplets;
        for (int i = 0; i < rhocount * xcount; i++) {
            for (auto iter = TripletVecVec[i].begin();
                 iter != TripletVecVec[i].end(); ++iter) {
                Eigen::Triplet<RealT> lto(*iter);
                Triplets.emplace_back(*iter);
            }
        }
        BCOffsetTensor.resize(rhocount * xcount * ycount,
                              rhocount * xcount * ycount * 16);
        BCOffsetTensor.setFromTriplets(Triplets.begin(), Triplets.end());

        std::cout << "Number of RealT  products needed for BC calc: " << BCOffsetTensor.nonZeros() << " and rough memory usage is " << BCOffsetTensor.nonZeros() * (sizeof(RealT) + sizeof(long)) << std::endl;
        return BCOffsetTensor;
    }

    functionGrid<rhocount, xcount, ycount, RealT> calculate(
        functionGrid<rhocount, xcount, ycount, RealT> &f) {
        functionGrid<rhocount, xcount, ycount, RealT> m =
            f;  // do we need to write a copy constructor?

        m.clearGrid();
        typename functionGrid<rhocount, xcount, ycount, RealT>::bicubicParameterGrid b = functionGrid<rhocount, xcount, ycount, RealT>::setupBicubicGrid(f);

        Eigen::Map<Eigen::Matrix<RealT, rhocount * xcount * ycount * 16, 1>> source(b.data.data());
        Eigen::Map<Eigen::Matrix<RealT, rhocount * xcount * ycount, 1>> target(m.gridValues.data.data());

        target = BCSparseOperator * source;
        return m;
    }
};

// GPU Bicubic DP
template <int rhocount, int xcount, int ycount, class RealT = double>
class bicubicDotProductGPU
    : public GACalculator<rhocount, xcount, ycount, RealT> {
   private:
    viennacl::compressed_matrix<RealT> vcl_sparse_matrix;
    //viennacl::vector<RealT> gpu_source;
    //viennacl::vector<RealT> gpu_target;

   public:
    friend class GACalculator<rhocount, xcount, ycount, RealT>;
    bicubicDotProductGPU(const gridDomain<RealT> &g, functionGrid<rhocount, xcount, ycount, RealT> &f)
        : vcl_sparse_matrix(xcount * ycount * rhocount, xcount * ycount * rhocount * 16) {
        Eigen::SparseMatrix<RealT, Eigen::RowMajor> BCSparseOperator = bicubicDotProductCPU<rhocount, xcount, ycount, RealT>::assembleFastBCCalc(g, f);
        viennacl::copy(BCSparseOperator, vcl_sparse_matrix);
        typename functionGrid<rhocount, xcount, ycount, RealT>::bicubicParameterGrid b = functionGrid<rhocount, xcount, ycount, RealT>::setupBicubicGrid(f);
        //gpu_source.resize(b.data.size());
        //gpu_target.resize(f.gridValues.data.size());
        //gpu_source.clear();
        //gpu_target.clear();
        std::cout << "no fail 1" << std::endl;
        auto garbage = this->calculate(f);  //call to initialize compute kernel maybe?
        viennacl::backend::finish();
    };
    ~bicubicDotProductGPU(){

    };

    static std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
    create(const gridDomain<RealT> &g, functionGrid<rhocount, xcount, ycount, RealT> &f) {
        return std::make_unique<bicubicDotProductGPU>(g, f);
    }
    std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>> clone() {
        return std::make_unique<bicubicDotProductGPU>(*this);
    } /*maybe disable this functionality */

    functionGrid<rhocount, xcount, ycount, RealT> calculate(
        functionGrid<rhocount, xcount, ycount, RealT> &f) {
        functionGrid<rhocount, xcount, ycount, RealT> m =
            f;

        //        m.clearGrid();
        typename functionGrid<rhocount, xcount, ycount, RealT>::bicubicParameterGrid b = functionGrid<rhocount, xcount, ycount, RealT>::setupBicubicGrid(f);
        viennacl::vector<RealT> gpu_source(b.data.size());
        viennacl::copy(b.data.begin(), b.data.end(), gpu_source.begin());

        std::cout << "no fail 2" << std::endl;
        viennacl::backend::finish();
        //viennacl::copy(m.gridValues.data.begin(), m.gridValues.data.end(), gpu_target.begin());
        //viennacl::backend::finish();
        // this is garbage data, I just want to make sure  it's allocated.
        viennacl::vector<RealT> gpu_target = viennacl::linalg::prod(vcl_sparse_matrix, gpu_source);

        std::cout << "no fail 3" << std::endl;
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       m.gridValues.data.begin());
        viennacl::backend::finish();
        return m;
    }
};
// GPU Bicubic DP

// GPU Linear DP
template <int rhocount, int xcount, int ycount, class RealT = double>
class linearDotProductGPU
    : public GACalculator<rhocount, xcount, ycount, RealT> {
   private:
    viennacl::compressed_matrix<RealT> vcl_sparse_matrix;
    //viennacl::vector<RealT> gpu_source;
    //viennacl::vector<RealT> gpu_target;

   public:
    friend class GACalculator<rhocount, xcount, ycount, RealT>;
    linearDotProductGPU(const gridDomain<RealT> &g, functionGrid<rhocount, xcount, ycount, RealT> &f)
        : vcl_sparse_matrix(xcount * ycount * rhocount, xcount * ycount * rhocount * 16) {
        Eigen::SparseMatrix<RealT, Eigen::RowMajor> BCSparseOperator = linearDotProductCPU<rhocount, xcount, ycount, RealT>::assembleFastBCCalc(g, f);
        viennacl::copy(BCSparseOperator, vcl_sparse_matrix);
        typename functionGrid<rhocount, xcount, ycount, RealT>::linearParameterGrid b = functionGrid<rhocount, xcount, ycount, RealT>::setuplinearGrid(f);
        //gpu_source.resize(b.data.size());
        //gpu_target.resize(f.gridValues.data.size());
        //gpu_source.clear();
        //gpu_target.clear();
        std::cout << "no fail 1" << std::endl;
        auto garbage = this->calculate(f);  //call to initialize compute kernel maybe?
        viennacl::backend::finish();
    };
    ~linearDotProductGPU(){

    };

    static std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
    create(const gridDomain<RealT> &g, functionGrid<rhocount, xcount, ycount, RealT> &f) {
        return std::make_unique<linearDotProductGPU>(g, f);
    }
    std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>> clone() {
        return std::make_unique<linearDotProductGPU>(*this);
    } /*maybe disable this functionality */

    functionGrid<rhocount, xcount, ycount, RealT> calculate(
        functionGrid<rhocount, xcount, ycount, RealT> &f) {
        functionGrid<rhocount, xcount, ycount, RealT> m =
            f;

        //        m.clearGrid();
        typename functionGrid<rhocount, xcount, ycount, RealT>::linearParameterGrid b = functionGrid<rhocount, xcount, ycount, RealT>::setuplinearGrid(f);
        viennacl::vector<RealT> gpu_source(b.data.size());
        viennacl::copy(b.data.begin(), b.data.end(), gpu_source.begin());

        std::cout << "no fail 2" << std::endl;
        viennacl::backend::finish();
        //viennacl::copy(m.gridValues.data.begin(), m.gridValues.data.end(), gpu_target.begin());
        //viennacl::backend::finish();
        // this is garbage data, I just want to make sure  it's allocated.
        viennacl::vector<RealT> gpu_target = viennacl::linalg::prod(vcl_sparse_matrix, gpu_source);

        std::cout << "no fail 3" << std::endl;
        viennacl::backend::finish();
        viennacl::copy(gpu_target.begin(), gpu_target.end(),
                       m.gridValues.data.begin());
        viennacl::backend::finish();
        return m;
    }
};
// GPU Linear DP

template <int rhocount, int xcount, int ycount, class RealT, int padcount>
std::unique_ptr<GACalculator<rhocount, xcount, ycount, RealT>>
GACalculator<rhocount, xcount, ycount, RealT, padcount>::Factory::newCalculator(
    calculatorType c, gridDomain<RealT> &g, functionGrid<rhocount, xcount, ycount, RealT> &f) {
    if (c == calculatorType::linearCPU)
        return linearCPUCalculator<rhocount, xcount, ycount, RealT>::create();
    else if (c == calculatorType::bicubicCPU)
        return bicubicCPUCalculator<rhocount, xcount, ycount, RealT>::create();
    else if (c == calculatorType::DCTCPUCalculator)
        return DCTCPUCalculator<rhocount, xcount, ycount, RealT>::create(g);
    else if (c == calculatorType::DCTCPUCalculator2)
        return DCTCPUCalculator2<rhocount, xcount, ycount, RealT>::create(g);
    else if (c == calculatorType::linearDotProductCPU)
        return linearDotProductCPU<rhocount, xcount, ycount, RealT>::create(g, f);
    else if (c == calculatorType::bicubicDotProductCPU)
        return bicubicDotProductCPU<rhocount, xcount, ycount, RealT>::create(g, f);
    else if (c == calculatorType::DCTCPUPaddedCalculator2)
        return DCTCPUPaddedCalculator<rhocount, xcount, ycount, padcount, RealT>::create(g);
    else if (c == calculatorType::bicubicDotProductGPU)
        return bicubicDotProductGPU<rhocount, xcount, ycount, RealT>::create(g, f);
    else if (c == calculatorType::linearDotProductGPU)
        return linearDotProductGPU<rhocount, xcount, ycount, RealT>::create(g, f);
    else
        return linearCPUCalculator<rhocount, xcount, ycount, RealT>::create();
}

}  // namespace OOGA

#endif  // GYROAVERAGING_GA_H

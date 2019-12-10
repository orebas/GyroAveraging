// GyroDebugging.cpp : This file contains the 'main' function. Program execution begins and ends there.
////#include "pch.h"

#include "ga.h"
#include <algorithm>
#include <array>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/timer/timer.hpp>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <math.h>
#include <omp.h>
#include <vector>
#define NDEBUG

template <typename TFunc>
double TanhSinhIntegrate(double x, double y, TFunc f) { //used to be trapezoid rule.
    boost::math::quadrature::tanh_sinh<double> integrator;
    return integrator.integrate(f, x, y);
}

template <typename TFunc>
inline double TrapezoidIntegrate(double x, double y, TFunc f) { //used to be trapezoid rule.
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

//add handling for rho =0.
template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::assembleFastGACalc(void) {
    std::map<std::pair<int, int>, std::array<double, 4>> GAOffsetMap;
    for (auto i = 0; i < rhocount; i++) {
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
                    interpIndexSearch(i, xmid, ymid, xInterpIndex, yInterpIndex);
                    sparseOffset so; //fill this in
                    so.target = &(fastGACalcResultOffset(i, j, k)) - &(fastGACalcResultOffset(0, 0, 0));
                    so.source = &(interpParameters(i, xInterpIndex, yInterpIndex, 0)) - &(interpParameters(0, 0, 0, 0));
                    so.coeffs[0] = coeffs[0] / (2 * pi);
                    so.coeffs[1] = coeffs[1] / (2 * pi);
                    so.coeffs[2] = coeffs[2] / (2 * pi);
                    so.coeffs[3] = coeffs[3] / (2 * pi);
                    GAOffsetMap[std::pair<int, int>(so.source, so.target)] = GAOffsetMap[std::pair<int, int>(so.source, so.target)] + std::array<double, 4>({so.coeffs[0], so.coeffs[1], so.coeffs[2], so.coeffs[3]});
                }
            }
    }
    //std::cout << "Size: " << GAOffsetTensor.size() << " and sizeof is " << sizeof(sparseOffset) << std::endl;
    for (auto iter = GAOffsetMap.begin(); iter != GAOffsetMap.end(); ++iter) {
        sparseOffset so;
        so.source = (iter->first).first;
        so.target = (iter->first).second;
        so.coeffs[0] = (iter->second)[0];
        so.coeffs[1] = (iter->second)[1];
        so.coeffs[2] = (iter->second)[2];
        so.coeffs[3] = (iter->second)[3];
        GAOffsetTensor.push_back(so);
    }
    std::sort(GAOffsetTensor.begin(), GAOffsetTensor.end(), [](sparseOffset a, sparseOffset b) -> bool { 
				if(a.source == b.source)
                    return a.target < b.target;
                else
					return a.source < b.source; });
    /*int dupCounter = 0;
    for (int c = 0; c < GAOffsetTensor.size() - 1; c++) {
        if (GAOffsetTensor[c].source == GAOffsetTensor[c + 1].source)
            if (GAOffsetTensor[c].target == GAOffsetTensor[c + 1].target)
                dupCounter++;
    }*/
    std::cout << "Number of 4-vector dot-products needed for GA calc: " << GAOffsetTensor.size() << " and rough memory usage is " << GAOffsetTensor.size() * sizeof(sparseOffset) << std::endl;
}

template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::fastGACalcOffset() {
    clearGrid(fastGACalcResultOffset);
    int size = GAOffsetTensor.size();
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        auto interpPlace = interpParameters.data.begin() + GAOffsetTensor[i].source;
        fastGACalcResultOffset.data[GAOffsetTensor[i].target] += GAOffsetTensor[i].coeffs[0] * interpPlace[0];
        fastGACalcResultOffset.data[GAOffsetTensor[i].target] += GAOffsetTensor[i].coeffs[1] * interpPlace[1];
        fastGACalcResultOffset.data[GAOffsetTensor[i].target] += GAOffsetTensor[i].coeffs[2] * interpPlace[2];
        fastGACalcResultOffset.data[GAOffsetTensor[i].target] += GAOffsetTensor[i].coeffs[3] * interpPlace[3];
    }
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
    fillbyindex(almostExactGA, [&](int i, int j, int k) -> double { //adaptive trapezoid rule on actual input function.
        double xc = xset[j];
        double yc = yset[k];
        if (rhoset[i] == 0)
            return f(i, xc, yc);
        auto new_f = [&](double x) -> double { return f(i, xc + rhoset[i] * std::sin(x), yc - rhoset[i] * std::cos(x)); };
        double result = TrapezoidIntegrate(0, 2 * pi, new_f) / (2 * pi);
        return result;
    });
    t.report();
    std::cout << "That was the time required to calculate gyroaverages from the definition, with the trapezoid rule.\n";
    t.start();
    fillbyindex(truncatedAlmostExactGA, [&](int i, int j, int k) -> double {
        double xc = xset[j];
        double yc = yset[k];
        if (rhoset[i] == 0)
            return f(i, xc, yc);
        auto new_f = [&](double x) -> double {
            double ex = xc + rhoset[i] * std::sin(x);
            double why = yc - rhoset[i] * std::cos(x);
            if ((ex < xset[0]) || (ex > xset.back()))
                return 0;
            if ((why < yset[0]) || (why > yset.back()))
                return 0;
            return f(i, ex, why);
        };
        double result = TrapezoidIntegrate(0, 2 * pi, new_f) / (2 * pi);
        return result;
    });
    t.report();
    std::cout << "That was the time required to calculate gryoaverages by def (as above), except we hard truncated f() to 0 off-grid.\n";
    t.start();

    fillbyindex(trapezoidInterp, [&](int i, int j, int k) -> double {
        double xc = xset[j];
        double yc = yset[k];
        if (rhoset[i] == 0)
            return f(i, xc, yc);
        auto new_f = [&](double x) -> double { return interp2d(i, xc + rhoset[i] * std::sin(x), yc - rhoset[i] * std::cos(x)); };
        double result = TrapezoidIntegrate(0, 2 * pi, new_f) / (2 * pi);
        return result;
    });
    t.report();
    std::cout << "That was the time required to calc gyroaverages by def, replacing f() by its bilinear interpolant.\n";
    t.start();
    assembleFastGACalc();
    t.report();
    std::cout << "That was the time required to assemble the sparse matrix in the fast-GA dot product calculation.\n";
    t.start();
    int times = 100;
    for (int counter = 0; counter < times; counter++) {
        setupInterpGrid();
        fastGACalcOffset();
    }

    t.report();
    std::cout << "The was the time require to run dot-product gyroaverage calc " << times << " times. \n "
              << std::endl;
    t.start();
    std::cout << "Below are some summary statistics for various grid.  Under each header is a pair of values.  The first is the RMS of a matrix, the second is the max absolute entry in a matrix.\n";

    std::cout << "rho        Input Grid                   Analytic Estimates              Trapezoid Rule                  Impact of truncation            Trapezoid rule vs bilin interp  Fast dot-product GA\n";

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
                  << RMSNorm(fastGACalcResultOffset, i) << "\t"
                  << maxNorm(fastGACalcResultOffset, i) << "\n";
    }
    std::cout << "Diffs:\n";
    std::cout << "rho        Analytic vs Quadrature       Error due to truncation         Error due to interp             Analytic vs interp              interp vs dot-product GA \n";
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
                  << RMSNormDiff(trapezoidInterp, fastGACalcResultOffset, i) << "\t"
                  << maxNormDiff(trapezoidInterp, fastGACalcResultOffset, i) << "\n";
    }
}

//below function returns the indices referring to lower left point of the grid box containing (x,y)
template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::interpIndexSearch(const int rhoindex, const double x, const double y, int &xindex, int &yindex) {

    assert((rhoindex >= 0) && (rhoindex < rhocount));
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
    interpIndexSearch(rhoindex, x, y, xindex, yindex);
    double result = BilinearInterpolation(gridValues(rhoindex, xindex, yindex), gridValues(rhoindex, xindex + 1, yindex),
                                          gridValues(rhoindex, xindex, yindex + 1), gridValues(rhoindex, xindex + 1, yindex + 1),
                                          xset[xindex], xset[xindex + 1], yset[yindex], yset[yindex + 1],
                                          x, y);

    return result;
}

int main() {

    /*   const double rhomax = 3, rhomin = 0.0;                //
	constexpr int xcount = 64, ycount = 64, rhocount = 8; //bump up to 64x64x24 later
	const double xmin = -5, xmax = 5;
	const double ymin = -5, ymax = 5;
	const double A = 1;
	double B = 2.0;*/

    constexpr double rhomax = 3, rhomin = 0.0;             // set from 0 to 3 maybe?  remember to test annoying integral multiples. TEST 0. test negative.
    constexpr int xcount = 64, ycount = 64, rhocount = 35; //bump up to 64x64x24 later
    constexpr double xmin = -5, xmax = 5;
    constexpr double ymin = -5, ymax = 5;
    constexpr double A = 5;
    constexpr double B = 5.0;
    constexpr double Normalizer = 1.0;
    std::vector<double> rhoset;
    std::vector<double> xset;
    std::vector<double> yset;

    /*auto verySmoothFunc = [](double row, double ex, double why) -> double {
        double temp = ex * ex + why * why;
        if (temp >= 25)
            return 0;
        else
            return (15 * std::exp(1 / (temp / 25.0 - 1.0)));
    };*/
    //auto constantFuncAnalytic = [](double row, double ex, double why) -> double {return 2*pi;};
    //auto linearFunc = [](double row, double ex, double why) -> double {return std::max(0.0,5 - std::abs(ex) - std::abs(why));};
    //auto constantFuncAnalytic = [](double row, double ex, double why) -> double {return 2*pi;};
    auto testfunc1 = [A](double row, double ex, double why) -> double { return Normalizer * exp(-A * (ex * ex + why * why)); };
    auto testfunc1_analytic = [A](double row, double ex, double why) -> double {
        return Normalizer * exp(-A * (ex * ex + why * why + row * row)) * boost::math::cyl_bessel_i(0, 2 * A * row * std::sqrt(ex * ex + why * why));
    };
    auto testfunc2 = [Normalizer, A, B](double row, double ex, double why) -> double { return Normalizer * exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
    auto testfunc2_analytic = [Normalizer, A, B](double row, double ex, double why) -> double { return Normalizer * exp(-B * row * row) * exp(-A * (ex * ex + why * why + row * row)) * boost::math::cyl_bessel_i(0, 2 * A * row * std::sqrt(ex * ex + why * why)); };
    /*auto testfunc2_analytic = [A, B](double row, double ex, double why) -> double {
        double rsquared = ex * ex + why * why;  //THIS SEEMS WRONG.
        double alpha = (A * B) / (A + B);
        return (exp(-alpha * rsquared) / (2 * (A + B)));
    };*/
    rhoset = LinearSpacedArray(rhomin, rhomax, rhocount);
    xset = LinearSpacedArray(xmin, xmax, xcount);
    yset = LinearSpacedArray(ymin, ymax, ycount);

    GyroAveragingGrid<rhocount, xcount, ycount> g(rhoset, xset, yset);
    g.GyroAveragingTestSuite(testfunc1, testfunc1_analytic);
}
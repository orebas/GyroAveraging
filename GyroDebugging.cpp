// GyroDebugging.cpp : This file contains the 'main' function. Program execution begins and ends there.
////#include "pch.h"


#include <boost/timer/timer.hpp>

#include <iostream>
#include <algorithm>
#include <array>
#include <boost/math/special_functions/bessel.hpp>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <math.h>
#include <vector>
#include <eigen3/eigen/Eigen>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
//#include <boost/spirit/include/karma.hpp>

#define NDEBUG
//#undef NDEBUG
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
double TanhSinhIntegrate(double x, double y, TFunc f) { //used to be trapezoid rule.
	boost::math::quadrature::tanh_sinh<double> integrator;
	return integrator.integrate(f, x, y);

	/*double h = (y - x) / (n - 1);
	double sum = 0;
	double fx1 = f(x);
	for (int i = 1; i <= n; i++) {
		double x2 = x + i * h;
		double fx2 = f(x2);
		sum += (fx1 + fx2) * h;
		fx1 = fx2;
	}
	return 0.5 * sum;*/
	//using namespace boost;
}



template <typename TFunc>
inline double TrapezoidIntegrate(double x, double y, TFunc f) { //used to be trapezoid rule.
using boost::math::quadrature::trapezoidal;
  return trapezoidal(f,x,y);

	/*double h = (y - x) / (n - 1);
	double sum = 0;
	double fx1 = f(x);
	for (int i = 1; i <= n; i++) {
		double x2 = x + i * h;
		double fx2 = f(x2);
		sum += (fx1 + fx2) * h;
		fx1 = fx2;
	}
	return 0.5 * sum;*/
	//using namespace boost;
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


struct sparseEntry {
	std::array<int, 3> target; //target entry i,j,k
	std::array<double, 4> coeffs;
	std::array<int, 3> source;
	sparseEntry(std::array<int, 3> t, std::array<double, 4> c, std::array<int, 3> s) :
		target(t), coeffs(c), source(s) {}
};

struct sparseOffset{ //no constructor.
  int target, source;
  double coeffs[4];
};


template <int rhocount, int xcount, int ycount>
class GyroAveragingGrid {
public:
  typedef Array3d<rhocount, xcount, ycount, double> fullgrid;
  typedef Array4d<rhocount, xcount, ycount, 4, double> fullgridInterp;
  
private:
  std::vector<double> rhoset;
  std::vector<double> xset;
  std::vector<double> yset;
  fullgrid gridValues;    //input values of f
  fullgrid almostExactGA; //stores value of GA calculated as trapezoid rule on input f
  fullgrid truncatedAlmostExactGA; //above, except f hard truncated to 0 outside grid
  fullgrid trapezoidInterp;  //GA calculated as trapezoid rule on interpolated, truncated f
  fullgrid fastGACalcResult;
  fullgrid analytic_averages; // stores value of expected GA computed analytically
  fullgrid exactF;
  fullgridInterp interpParameters; //will store the bilinear interp parameters.
  std::vector<sparseEntry> GATensor;
  std::vector<sparseOffset> GAOffsetTensor;
  
  void csvPrinter(const fullgrid &m, int rho){
    for(int j=0;j<xcount;j++){
      for(int k=0; k<ycount; k++){
	std::cout << m(rho,j,k) <<",";
      }
      std::cout << std::endl;
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
	void interpIndexSearch(const int rhoindex, const double x, const double y, int & xindex, int & yindex);
	inline void integrand(const double rho, const double xc, const double yc, const double gamma, double &xn, double &yn) {
		xn = xc + rho * std::sin(gamma);
		yn = yc - rho * std::cos(gamma);
	}
	void setupInterpGrid();
	void assembleFastGACalc(void);
	void fastGACalc();
	std::array<double, 4> arcIntegral(double rho, double xc, double yc, double s0, double s1);
	template <typename TFunc1, typename TFunc2>
	void GyroAveragingTestSuite(TFunc1 f,
		TFunc2 analytic,
		int trapezoidRuleN); //last param ignored for now.

	double interp2d(int rhoindex, const double x, const double y);
};

/* The below function is going to fill a 4d array with the A,B,C,D coefficients such that
for each i,j,k with j,k not on the top or right edge
for every x,y inside the box xset[j],xset[j+1],yset[k],yset[k+1]
interp2d(i,x,y,etc) = A+Bx+Cy+Dxy (approx)
we can optimize this better later.
*/
template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::setupInterpGrid() {
	using namespace Eigen;
	for (int i = 0; i < rhocount; i++) {
		interpParameters(i, xcount - 1, ycount - 1, 0) = 0; // CHANGE BACK TO 0
		interpParameters(i, xcount - 1, ycount - 1, 1) = 0;
		interpParameters(i, xcount - 1, ycount - 1, 2) = 0;
		interpParameters(i, xcount - 1, ycount - 1, 3) = 0;
		for (int j = 0; j < xcount - 1; j++)
			for (int k = 0; k < ycount - 1; k++) {
				double x1 = xset[j],
					x2 = xset[j + 1],
					y1 = yset[k],
					y2 = yset[k + 1];
				//double denom = 1 / ((x1 - x2) * (y1 - y2));
				double Q11 = gridValues(i, j, k),
					Q12 = gridValues(i, j + 1, k),
					Q21 = gridValues(i, j, k + 1),
					Q22 = gridValues(i, j + 1, k + 1);

				//double a0 = (Q11 * x2 * y2 - Q12 * x2 * y1 - Q21 * x1 * y2 + Q22 * x1 * y1) / denom;
				//double a1 = (-Q11 * y2 + Q12 * y1 + Q21 * y2 - Q22 * y1) / denom;
				//double a2 = (-Q11 * x2 + Q12 * x2 + Q21 * x1 - Q22 * x1) / denom;
				//double a3 = (Q11 - Q12 - Q21 + Q22) / denom;
				Matrix4d mat;
				mat << 1, x1, y1, x1* y1,
					1, x1, y2, x1*y2,
					1, x2, y1, x2*y1,
					1, x2, y2, x2*y2;
				Vector4d B;
				B << Q11, Q12, Q21, Q22;
				Vector4d anums = mat.fullPivLu().solve(B); //HouseholderQr
				double a0 = anums(0);
				double a1 = anums(1);
				double a2 = anums(2);
				double a3 = anums(3);
				/*auto testf = [=](double x, double y) -> double { return a0 + a1 * x + a2 * y + a3 * x * y; };
				std::cout << mat << std::endl;
				std::cout << anums << std::endl;
				std::cout << B << std::endl;
				std::cout << (mat * anums) - B << std::endl;
				double tester = mat(0, 0)*anums(0) + mat(0, 1)*anums(1) + mat(0, 2)*anums(2) + mat(0, 3)*anums(3);
				double tester2 = 1*anums(0) + x1*anums(1) + y1*anums(2) + x1*y1*anums(3);

				std::cout << tester << " " << testf(x1, y1) << " " << Q11 << " " << testf(x1, y1) - Q11 << std::endl;
				bool testtrue = (testf(x1, y1) == Q11);
				if (!testtrue) {
					std::cout << testf(x1, y1) << " " << Q11;
				}

				assert(testf(x1, y1) == Q11);
				assert(testf(x1, y2) == Q12);
				assert(testf(x2, y1) == Q21);
				assert(testf(x2, y2) == Q22);
				*/
				interpParameters(i, j, k, 0) = a0;
				interpParameters(i, j, k, 1) = a1;
				interpParameters(i, j, k, 2) = a2;
				interpParameters(i, j, k, 3) = a3;
			}
	}
}

struct indexedPoint {
	double xvalue = 0;
	double yvalue = 0;
	double s = 0; // from 0 to 2*Pi only please.
	indexedPoint(double x = 0, double y = 0, double row = 0) : xvalue(x), yvalue(y), s(row) {}
};




template <int rhocount, int xcount, int ycount>

std::array<double, 4> GyroAveragingGrid<rhocount, xcount, ycount>::arcIntegral(
	double rho, double xc, double yc, double s0, double s1) {
	std::array<double, 4> coeffs;
	double coss1 = std::cos(s1), coss0 = std::cos(s0), sins0 = std::sin(s0), sins1 = std::sin(s1);
	coeffs[0] = s1 - s0;
	coeffs[1] = (s1*xc - rho * coss1) - (s0*xc - rho * coss0);
	coeffs[2] = (s1*yc - rho * sins1) - (s0*yc - rho * sins0);
	coeffs[3] = (s1*xc*yc - rho * yc*coss1 - rho * xc*sins1 + rho * rho*coss1*coss1 / 2.0) -
		(s0*xc*yc - rho * yc*coss0 - rho * xc*sins0 + rho * rho*coss0*coss0 / 2.0);
	return coeffs;
}

//add handling for rho =0.
template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::assembleFastGACalc(void) {
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
				for (auto v : xset)
					if (std::abs(v - xc) <= rho) {
						double deltax = v - xc;
						double deltay = std::sqrt(rho*rho - deltax * deltax);
						if ((yc + deltay >= ymin) && (yc + deltay <= ymax))
							intersections.push_back(indexedPoint(v, yc + deltay, 0));
						if ((yc - deltay >= ymin) && (yc - deltay <= ymax))
							intersections.push_back(indexedPoint(v, yc - deltay, 0));
					}
				for (auto v : yset)
					if (std::abs(v - yc) <= rho) {
						double deltay = v - yc;
						double deltax = std::sqrt(rho*rho - deltay * deltay);
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
				std::sort(intersections.begin(), intersections.end(), [](indexedPoint a, indexedPoint b) {return a.s < b.s; });
				assert(intersections.size() > 0);
				assert(intersections[0].s == 0);
				assert(intersections.back().s == (2 * pi));
				for (auto p = 0; p < intersections.size() - 1; p++) {
					double s0 = intersections[p].s, s1 = intersections[p + 1].s;
					double xmid, ymid, testIntegration = 0;
					std::array<double, 4> coeffs;
					int xInterpIndex = 0, yInterpIndex = 0;
					if (s1 - s0 < 1e-12) continue; //let's handle this better - don't want the double 0 there to begin with for instance.

					/* below code is debugging code*/
					/*double epsilon = 1e-8;
					double ns0 = s0 + epsilon;
					double ns1 = s1 - epsilon;
					double ox, oy, nx, ny;
					int N = 25;
					double step = (ns1 - ns0) / N;


					int oxi = 0, oyi = 0, nxi = 0, nyi = 0;
					integrand(rho, xc, yc, ns0, ox, oy);
					interpIndexSearch(i, ox, oy, oxi, oyi);
					integrand(rho, xc, yc, ns0, nx, ny);
					interpIndexSearch(i, nx, ny, nxi, nyi);
					while (ns0 < ns1) {
						integrand(rho, xc, yc, ns0, nx, ny);
						interpIndexSearch(i, nx, ny, nxi, nyi);
						assert(nxi == oxi);
						assert(nyi == oyi);
						ns0 += step;
					}
					ns0 = ns1;
					integrand(rho, xc, yc, ns0, nx, ny);
					interpIndexSearch(i, nx, ny, nxi, nyi);
					assert(nxi == oxi);
					assert(nyi == oyi);*/
					/*end debugging code*/
					integrand(rho, xc, yc, (s0 + s1) / 2, xmid, ymid);  //this just calculated into xmid,ymid the point half through the arc.
					coeffs = arcIntegral(rho, xc, yc, s0, s1);
					interpIndexSearch(i, xmid, ymid, xInterpIndex, yInterpIndex);
					/*testIntegration =
						(interpParameters(i, xInterpIndex, yInterpIndex, 0) * coeffs[0] +
							interpParameters(i, xInterpIndex, yInterpIndex, 1) * coeffs[1] +
							interpParameters(i, xInterpIndex, yInterpIndex, 2) * coeffs[2] +
							interpParameters(i, xInterpIndex, yInterpIndex, 3) * coeffs[3]) / (2 * pi);*/
					sparseEntry se({ i,j,k }, coeffs, { i,xInterpIndex,yInterpIndex });
					sparseOffset so; //fill this in
					
				       
					se.target[0] = i;
					se.target[1] = j;
					se.target[2] = k;
					se.source[0] = i;
					se.source[1] = xInterpIndex;
					se.source[2] = yInterpIndex;
					se.coeffs[0] = coeffs[0] / (2 * pi);
					se.coeffs[1] = coeffs[1] / (2 * pi);
					se.coeffs[2] = coeffs[2] / (2 * pi);
					se.coeffs[3] = coeffs[3] / (2 * pi);
					GATensor.push_back(se);
					//double temp = fastGACalcResult(i, j, k) + testIntegration;
					//fastGACalcResult(i, j, k) = temp;


					/*We believe that s0 and s1 both fall on the border of the same gridbox,
					so their midpoint should be in its interior.  We should add code to check this. */
					//different verification below:  we check integration vs trap rule.

					/*auto new_f = [&](double x) -> double {  return interp2d(i, xc + rhoset[i] * std::sin(x), yc - rhoset[i] * std::cos(x)); };
					double testAgainst = TanhSinhIntegrate(s0, s1, new_f) / (2 * pi);
					std::cout.precision(12);
					if (std::abs(testAgainst - testIntegration) > 1e-14) {
						std::cout << testAgainst - testIntegration << " " << testAgainst << " " << testIntegration << "\n";
						assert(false);
						//COME BACK HERE
						*/
				}
			}


	}
}



template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::fastGACalc() {
	//std::cout << fastGACalcResult.data << std::endl;
	//return;
	for (auto i = 0; i < GATensor.size(); i++) {
	  sparseEntry v = GATensor[i]; //store raw data indices for speed.
		fastGACalcResult(v.target[0], v.target[1], v.target[2]) += v.coeffs[0] * interpParameters(v.source[0], v.source[1], v.source[2], 0);
		fastGACalcResult(v.target[0], v.target[1], v.target[2]) += v.coeffs[1] * interpParameters(v.source[0], v.source[1], v.source[2], 1);
		fastGACalcResult(v.target[0], v.target[1], v.target[2]) += v.coeffs[2] * interpParameters(v.source[0], v.source[1], v.source[2], 2);
		fastGACalcResult(v.target[0], v.target[1], v.target[2]) += v.coeffs[3] * interpParameters(v.source[0], v.source[1], v.source[2], 3);
		//std::cout << fastGACalcResult(v.target[0], v.target[1], v.target[2]) << " "  };
	}

}



/* Run test suite.  We expect the test suite to:
1)     Calculate the gyroaverage transform of f, using f on a full grid, filling only [x,y]
2)     above, using f but truncating it to be 0 outside of our grid
3)     above, using bilinear interp of f (explicitly only gridpoint valuations are allowed)
		 but we trapezoid rule the interp-ed, truncated function  (similiar to (2), but only interp error introduced)
4)     above, but we split up piecwise bilinear f into different section on grid and analytically integrate them
5)     above, but we are only allowed to dotproduct with a Array3d that depends on our grid parameters and not on f.

Report evolution of error.  Later we add timings.

   */


template <int rhocount, int xcount, int ycount>
template <typename TFunc1, typename TFunc2>
void GyroAveragingGrid<rhocount, xcount, ycount>::GyroAveragingTestSuite(TFunc1 f,
	TFunc2 analytic,
									 int trapezoidRuleN) {
  boost::timer::auto_cpu_timer t;
  fill(gridValues, f);
  t.start();
  fill(analytic_averages, analytic);
  t.report(); std::cout << "That was analytic timing.\n";
  setupInterpGrid();
         t.start();
	fillbyindex(almostExactGA, [&](int i, int j, int k) -> double {
		double xc = xset[j];
		double yc = yset[k];
		if (rhoset[i] == 0)
			return f(i, xc, yc);
		auto new_f = [&](double x) -> double { return f(i, xc + rhoset[i] * std::sin(x), yc - rhoset[i] * std::cos(x)); };
		double result = TrapezoidIntegrate(0, 2 * pi, new_f) / (2 * pi);
		return result;
	});
	t.report(); std::cout<< "That was almostExact timing.\n";
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
	t.report();   std::cout<< "That was almostExact truncated timing.\n";
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
	t.report();  std::cout << "That was quadrature of interp timing.\n"; t.start();
	assembleFastGACalc();
	t.report();  std::cout << "That was assembly time.\n" ; t.start();
	std::cout << GATensor.size() << "\n";
	for(int counter =0; counter<100;counter++){
	setupInterpGrid();
	//std::cout << GATensor.size() << "\n";
	fastGACalc();
	//std::cout << GATensor.size() << "\n";
	}
	t.report(); std::cout << "That was fastGACalc timing. (100 runs)\n\n" << std::endl; t.start();


	for (int i = 0; i < rhocount; i++) {
		using namespace std;
		cout.precision(5);
		cout << fixed << setw(15)
			<< FrobNorm(gridValues, i) << "\t"
			<< maxNorm(gridValues, i) << "\t"
		  	<< FrobNorm(analytic_averages, i) << "\t"
		  	<< maxNorm(analytic_averages, i) << "\t"
			<< FrobNorm(almostExactGA, i) << "\t"
			<< maxNorm(almostExactGA, i) << "\t"
			<< FrobNorm(truncatedAlmostExactGA, i) << "\t"
			<< maxNorm(truncatedAlmostExactGA, i) << "\t"
			<< FrobNorm(trapezoidInterp, i) << "\t"
			<< maxNorm(trapezoidInterp, i) << "\t"
			<< FrobNorm(fastGACalcResult, i) << "\t"
			<< maxNorm(fastGACalcResult, i) << "\n";

	}
	for (int i = 0; i < rhocount; i++) {
		using namespace std;
		cout.precision(5);
		cout << fixed << setw(15)
		  << FrobNormDiff(analytic_averages, almostExactGA, i) << "\t"
		    << maxNormDiff(analytic_averages, almostExactGA, i) << "\t"
		     << FrobNormDiff(almostExactGA, truncatedAlmostExactGA, i) << "\t"
		     << maxNormDiff(almostExactGA, truncatedAlmostExactGA, i) << "\t"
		     << FrobNormDiff(truncatedAlmostExactGA, trapezoidInterp, i) << "\t"
		     << maxNormDiff(truncatedAlmostExactGA, trapezoidInterp, i) << "\t"
		   << FrobNormDiff(analytic_averages, trapezoidInterp, i) << "\t"
		    << maxNormDiff(analytic_averages, trapezoidInterp, i) << "\t"
		     << FrobNormDiff(trapezoidInterp, fastGACalcResult, i) << "\t"
		     << maxNormDiff(trapezoidInterp, fastGACalcResult, i) << "\t"
		     << FrobNormDiff(truncatedAlmostExactGA, fastGACalcResult, i) << "\t"
		     << maxNormDiff(truncatedAlmostExactGA, fastGACalcResult, i) << "\t"
		     << FrobNormDiff(analytic_averages, fastGACalcResult, i) << "\t"
		     << maxNormDiff(analytic_averages, fastGACalcResult, i) << "\n";		
	}
	//csvPrinter(almostExactGA,1);
	//csvPrinter(truncatedAlmostExactGA,1);
	//csvPrinter(trapezoidInterp,1);
	//csvPrinter(fastGACalcResult,1);
}


//below function returns the indices referring to lower left point of the grid box containing (x,y)
template <int rhocount, int xcount, int ycount>
void GyroAveragingGrid<rhocount, xcount, ycount>::interpIndexSearch(const int rhoindex, const double x, const double y, int & xindex, int & yindex) {

	assert((rhoindex >= 0) && (rhoindex < rhocount));
	if ((x < xset[0]) || (y < yset[0]) || (x > xset.back()) || (y > yset.back())) {
		xindex = xcount - 1;  // the top right corner should have zeros.
		yindex = ycount - 1;
		return;
	}
	xindex = std::upper_bound(xset.begin(),xset.end(),x) - xset.begin()-1;
	yindex = std::upper_bound(yset.begin(),yset.end(),y) - yset.begin()-1;
	xindex = std::min(std::max(xindex,0),xcount-2);
	yindex=std::min(ycount-2,std::max(yindex,0));
	//while ((xindex < xcount - 1) && (x > xset[xindex + 1]))
	//	xindex++;

	//while ((yindex < ycount - 1) && (y > yset[yindex + 1]))
	//	yindex++;


	/*
	xindex = ((x - xset[0]) * xcount) / (xset.back() - xset[0]);
	yindex = ((y - yset[0]) * ycount) / (yset.back() - yset[0]);
	if (xindex >= xcount-1)
		xindex--;
	if (yindex >= ycount-1)
		yindex--;
	while ((xset[xindex] >= x) && (xindex > 0))
		xindex--;
	while ((xindex < xcount - 1) && (xset[xindex + 1] < x))
		xindex++;
	assert((xset[xindex]) <= x && (xset[xindex + 1] >= x));
	while ((yset[yindex] > y) && (yindex > 0))
		yindex--;
	while ((yindex < ycount - 1) && (yset[yindex + 1] < y))
		yindex++;
		assert((yset[yindex]) <= y && (yset[yindex + 1] >= y));*/
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
	/*double testResult = interpParameters(rhoindex, xindex, yindex, 0) +
		interpParameters(rhoindex, xindex, yindex, 1) * x +
		interpParameters(rhoindex, xindex, yindex, 2) * y +
		interpParameters(rhoindex, xindex, yindex, 3) * x * y;
	if (std::abs(testResult - result) > 1e-15) {
		std::cout << rhoindex << " " << xindex << " " << yindex << " " << std::endl;
		//std::cout << rhoindex << " " << x << " " << y << " " << std::endl;
		//std::cout << rhoindex << " " << x << " " << y << " " << std::endl;

		std::cout << "Grid Values: " << gridValues(rhoindex, xindex, yindex) << " "
			<< gridValues(rhoindex, xindex + 1, yindex) << " "
			<< gridValues(rhoindex, xindex, yindex + 1) << " "
			<< gridValues(rhoindex, xindex + 1, yindex + 1) << " " << std::endl
			<< "Grid Coordinates: " << xset[xindex] << " "
			<< xset[xindex + 1] << " "
			<< yset[yindex] << " "
			<< yset[yindex + 1] << " " << std::endl
			<< "x,y: " << x << " "
			<< y << std::endl;
		std::cout << "Interp Parameters: " << interpParameters(rhoindex, xindex, yindex, 0) << " "
			<< interpParameters(rhoindex, xindex, yindex, 1) << " "
			<< interpParameters(rhoindex, xindex, yindex, 2) << " "
			<< interpParameters(rhoindex, xindex, yindex, 3) << std::endl;

		std::cout
			<< result << " "
			<< testResult << " " << result - testResult << "\n";
	}*/
	return result;
}

int main() {

	/*   const double rhomax = 3, rhomin = 0.0;                //
	constexpr int xcount = 64, ycount = 64, rhocount = 8; //bump up to 64x64x24 later
	const double xmin = -5, xmax = 5;
	const double ymin = -5, ymax = 5;
	const double A = 1;
	double B = 2.0;*/

	constexpr double rhomax = 3, rhomin = 0.0;                // set from 0 to 3 maybe?  remember to test annoying integral multiples. TEST 0. test negative.
	constexpr int xcount = 64, ycount = 64, rhocount = 24; //bump up to 64x64x24 later
	constexpr double xmin = -5, xmax = 5;
	constexpr double ymin = -5, ymax = 5;
	constexpr double A = 2; // 5
	constexpr double B = 2.0;
	constexpr double Normalizer = 5.0;
	std::vector<double> rhoset;
	std::vector<double> xset;
	std::vector<double> yset;

	auto verySmoothFunc = [](double row, double ex, double why) -> double {
			      double temp=ex*ex+why*why;
			      if (temp >=25)
				return 0;
			      else
				return(15*std::exp(1/(temp/25.0-1.0)));
				  };
	//  auto constantFuncAnalytic = [](double row, double ex, double why) -> double {return 2*pi;};
	//auto constantFunc = [](double row, double ex, double why) -> double {return std::max(0.0,5 - std::abs(ex) - std::abs(why));};
	//  auto constantFuncAnalytic = [](double row, double ex, double why) -> double {return 2*pi;};
	  auto testfunc1 = [A](double row, double ex, double why) -> double { return Normalizer*exp(-A * (ex * ex + why * why)); };
	  //auto testfunc2 = [A, B](double row, double ex, double why) -> double { return exp(-A * (ex * ex + why * why)) * exp(-B * row * row); };
	  auto testfunc1_analytic = [A](double row, double ex, double why) -> double {
	  	return Normalizer*exp(-A * (ex * ex + why * why + row * row)) * boost::math::cyl_bessel_i(0, 2 * A * row * std::sqrt(ex * ex + why * why));
	  };
//auto testfunc2_analytic = [A, B](double row, double ex, double why) -> double {
//		double rsquared = ex * ex + why * why;
//		double alpha = (A * B) / (A + B);
//		return (exp(-alpha * rsquared) / (2 * (A + B)));
//	};
	rhoset = LinearSpacedArray(rhomin, rhomax, rhocount);
	xset = LinearSpacedArray(xmin, xmax, xcount);
	yset = LinearSpacedArray(ymin, ymax, ycount);

	GyroAveragingGrid<rhocount, xcount, ycount> g(rhoset, xset, yset);
	g.GyroAveragingTestSuite(testfunc1, testfunc1_analytic, 500);

	//g.GyroAveragingTestSuite(testfunc1, testfunc1_analytic, 500);

	/* Run test suite.  We expect the test suite to:
1)     Calculate the gyroaverage transform of f, using f on a full grid, filling only [x,y]
2)     above, using f but truncating it to be 0 outside of our grid
3)     above, using bilinear interp of x (explicitly only gridpoint valuations are allowed)
		 but we trapezoid rule the interp-ed, truncated function  (similiar to (2), but only interp error introduced)
4)     above, but we split up piecwise bilinear f into different section on grid and analytically integrate them
5)     above, but we are only allowed to dotproduct with a Array3d that depends on our grid parameters and not on f.

Report evolution of error.  Later we add timings.

   */

   //  auto fsquare = [](double x) -> double {return x*x;};
   //  auto fexp = [](double x) -> double {return std::exp(x);} ;
	for (int i = 0; i < rhocount; i++) {
		//    std::cout << "For rho index " << i<< " f-norm is: "<< gridValues[i].norm() <<" naive norm is: "<< na[i].norm() << " analytic norm is: " << an[i].norm() << std::endl;
		//auto ret = (na[i] - an[i]).eval();
		//std::cout << "Norm of diff is: " << ret.norm() << " " << (ret).lpNorm<10> () << " " << ret.maxCoeff() << " " << ret.minCoeff() <<  std::endl;
	}
	std::cout << std::endl;
	//	assert(false);
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file


int oldmain() {
    using namespace OOGA;
    typedef double mainReal;

    gridDomain<mainReal> g;
    g.rhomax = 3;
    g.rhomin = 0;
    g.xmin = g.ymin = -5;
    g.xmax = g.ymax = 5;
    constexpr int xcount = 32, ycount = 32,
                  rhocount = 24;  // bump up to 64x64x35 later or 128x128x35
    constexpr mainReal A = 2;
    constexpr mainReal B = 2;
    constexpr mainReal Normalizer = 50.0;
    std::vector<mainReal> rhoset;
    std::vector<mainReal> xset;
    std::vector<mainReal> yset;

    /*auto verySmoothFunc = [](mainReal row, mainReal ex, mainReal why) -> mainReal {
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

    auto ZeroFunc = [](mainReal row, mainReal ex, mainReal why) -> mainReal {
        return 0;
    };
    auto constantFuncAnalytic = [](mainReal row, mainReal ex, mainReal why) -> mainReal { return 2 * pi; };
    auto linearFunc = [](mainReal row, mainReal ex, mainReal why) -> mainReal {
        return std::max(0.0, 5 - std::abs(ex) - std::abs(why));
    };
    auto testfunc1 = [A, Normalizer](mainReal row, mainReal ex,
                                     mainReal why) -> mainReal {
        return Normalizer * exp(-A * (ex * ex + why * why));
    };
    auto testfunc1_analytic = [A, Normalizer](mainReal row, mainReal ex, mainReal why) -> mainReal {
        return Normalizer * exp(-A * (ex * ex + why * why + row * row)) *
               boost::math::cyl_bessel_i(
                   0, 2 * A * row * std::sqrt(ex * ex + why * why));
    };*/
    auto testfunc2 = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal {
        return Normalizer * exp(-A * (ex * ex + why * why)) *
               exp(-B * row * row);
    };
    auto testfunc2_analytic = [Normalizer, A, B](mainReal row, mainReal ex, mainReal why) -> mainReal {
        return Normalizer * exp(-B * row * row) *
               exp(-A * (ex * ex + why * why + row * row)) *
               boost::math::cyl_bessel_i(
                   0, 2 * A * row * std::sqrt(ex * ex + why * why));
    };
    auto testfunc2_analytic_dx = [Normalizer, A, B](mainReal row, mainReal ex,
                                                    mainReal why) -> mainReal {
        return -2 * A * ex * Normalizer * exp(-A * (ex * ex + why * why)) *
               exp(-B * row * row);
    };
    auto testfunc2_analytic_dy = [Normalizer, A, B](mainReal row, mainReal ex,
                                                    mainReal why) -> mainReal {
        return -2 * A * why * Normalizer * exp(-A * (ex * ex + why * why)) *
               exp(-B * row * row);
    };
    auto testfunc2_analytic_dx_dy = [Normalizer, A, B](
                                        mainReal row, mainReal ex,
                                        mainReal why) -> mainReal {
        return 4 * A * A * ex * why * Normalizer *
               exp(-A * (ex * ex + why * why)) * exp(-B * row * row);
    };

    rhoset = LinearSpacedArray<mainReal>(g.rhomin, g.rhomax, rhocount);
    xset = LinearSpacedArray<mainReal>(g.xmin, g.xmax, xcount);
    yset = LinearSpacedArray<mainReal>(g.ymin, g.ymax, ycount);

    GyroAveragingGrid<rhocount, xcount, ycount> grid(rhoset, xset, yset);
    errorAnalysis(g, testfunc2, testfunc2_analytic);
    grid.GyroAveragingTestSuite(testfunc2, testfunc2_analytic);
    derivTest(g, testfunc2, testfunc2_analytic_dx, testfunc2_analytic_dy,
              testfunc2_analytic_dx_dy);
    interpAnalysis(g, testfunc2, testfunc2_analytic);
    testInterpImprovement();
    testArcIntegralBicubic();
    return 0;
}

template <int count, typename TFunc1, typename TFunc2, class RealT>
void interpAnalysisInnerLoop(const gridDomain<RealT> &g, TFunc1 f,
                             TFunc2 analytic) {
    constexpr int rhocount = 4;  // TODO MAGIC NUMBER SHOULD BE PASSED IN
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
    constexpr int rhocount = 1;  // TODO MAGIC NUMBER SHOULD BE PASSED IN
    std::vector<RealT> rhoset;
    std::vector<RealT> xset;
    std::vector<RealT> yset;
    rhoset.push_back(g.rhomax);
    xset = LinearSpacedArray<RealT>(g.xmin, g.xmax, count);
    yset = LinearSpacedArray<RealT>(g.ymin, g.ymax, count);
    GyroAveragingGrid<rhocount, count, count, RealT> grid(rhoset, xset, yset);
    grid.compactErrorAnalysis(f, analytic);
}

template <typename TFunc1, typename TFunc2, class RealT>
void interpAnalysis(const gridDomain<RealT> &g, TFunc1 f, TFunc2 analytic) {
    constexpr int counts[] = {6, 12, 24, 48, 96, 192};
    interpAnalysisInnerLoop<counts[0]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[1]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[2]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[3]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[4]>(g, f, analytic);
    interpAnalysisInnerLoop<counts[5]>(g, f, analytic);
}

template <typename TFunc1, typename TFunc2, class RealT>
void errorAnalysis(const gridDomain<RealT> &g, TFunc1 f, TFunc2 analytic) {
    constexpr int counts[] = {
        6, 12, 24, 48,
        96, 192, 384, 768};  // we skip the last one, it's too big/slow.
    std::cout << "        gridN           Input Grid      Analytic Estimate    "
                 "           Linear Interp                   Bicubic Interp    "
                 "             Lin rel err      Bicubic Rel err\n";
    errorAnalysisInnerLoop<counts[0]>(g, f, analytic);
    errorAnalysisInnerLoop<counts[1]>(g, f, analytic);
    errorAnalysisInnerLoop<counts[2]>(g, f, analytic);
    errorAnalysisInnerLoop<counts[3]>(g, f, analytic);
    errorAnalysisInnerLoop<counts[4]>(g, f, analytic);
    // errorAnalysisInnerLoop<counts[5]>(g, f, analytic);
    // errorAnalysisInnerLoop<counts[6]>(g, f, analytic);
}

template <typename TFunc1, typename TFunc2, typename TFunc3, typename TFunc4,
          class RealT>
void derivTest(const gridDomain<RealT> &g, TFunc1 f, TFunc2 f_x, TFunc3 f_y,
               TFunc4 f_xy) {
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

    constexpr int rhocount = 4;  // TODO MAGIC NUMBER SHOULD BE PASSED IN
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
    // auto testfunc2 = [Normalizer, A, B](RealT row, RealT ex, RealT why) ->
    // RealT { return (1) * row; };

    auto testfunc2 = [Normalizer, A, B](localReal row, localReal ex,
                                        localReal why) -> localReal {
        return Normalizer * exp(-A * (ex * ex + why * why)) *
               exp(-B * row * row);
    };
    /*auto testfunc2_analytic = [Normalizer, A, B](localReal row, localReal ex,
                                                 localReal why) -> localReal {
        return Normalizer * exp(-B * row * row) *
               exp(-A * (ex * ex + why * why + row * row)) *
               boost::math::cyl_bessel_i(
                   0, 2 * A * row * std::sqrt(ex * ex + why * why));
    };
    auto testfunc2_analytic_dx = [Normalizer, A, B](
                                     localReal row, localReal ex,
                                     localReal why) -> localReal {
        return -2 * A * ex * Normalizer * exp(-A * (ex * ex + why * why)) *
               exp(-B * row * row);
    };
    auto testfunc2_analytic_dy = [Normalizer, A, B](
                                     localReal row, localReal ex,
                                     localReal why) -> localReal {
        return -2 * A * why * Normalizer * exp(-A * (ex * ex + why * why)) *
               exp(-B * row * row);
    };
    auto testfunc2_analytic_dx_dy = [Normalizer, A, B](
                                        localReal row, localReal ex,
                                        localReal why) -> localReal {
        return 4 * A * A * ex * why * Normalizer *
               exp(-A * (ex * ex + why * why)) * exp(-B * row * row);
    };*/

    smallgrid.fill(
        smallgrid.gridValues,
        testfunc2);  // This is the base grid of values we will interpolate.
    // smallgrid.fill(smallgrid.analytic_averages, testfunc2_analytic);
    // //analytic formula for gyroaverages
    smallgrid.setupInterpGrid();
    smallgrid.setupDerivsGrid();
    smallgrid.setupBicubicGrid();

    smallgrid.setupBicubicGrid();

    GyroAveragingGrid<rhocount, xcountb, ycountb>::fullgrid exact, lin, bic;
    for (int i = 0; i < rhocount; ++i) std::cout << rhoset;
    for (int i = 0; i < rhocount; ++i)
        for (int j = 0; j < xcountb; ++j)
            for (int k = 0; k < ycountb; ++k) {
                exact(i, j, k) = testfunc2(rhoset[i], xsetb[j], ysetb[k]);
                lin(i, j, k) = smallgrid.interp2d(i, xsetb[j], ysetb[k]);
                // bic(i, j, k) = smallgrid.interpNaiveBicubic(i, xsetb[j],
                // ysetb[k]);
                bic(i, j, k) =
                    smallgrid.interpNaiveBicubic(i, xsetb[j], ysetb[k]);
            }

    for (int i = 1; i < rhocount; ++i) {
        // smallgrid.csvPrinter(smallgrid.gridValues, i);
        // std::cout << "\n";

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

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>
#include <future>
#include <execution>

#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <limits>
#include <algorithm>

const float PI = 3.14159f;
const float RAD = 3.14159f / 180.0f;
const float HALF = 1.0f / 2.0f;
const int GRID_SIZE = 16;
const float COEFF = 1.3f;

struct vec2
{
    float x = 0.0f;
    float y = 0.0f;
};

struct unit
{
    vec2 position; // position of unit (-10^5...10^5, -10^5...10^5)
    vec2 direction; // normalized view direction
    float fov_deg = 0.0f; // horizontal field-of-view in degrees (0...180)
    float distance = 0.0f; // view distance (0...10^5)
};

struct Cell {
    std::vector<int> units;
};

struct Grid {
    Cell grid[GRID_SIZE + 1][GRID_SIZE + 1];
    float stepX;
    float stepY;
    float xBorder;
    float yBorder;
    __m128 gridBounds;
    __m128 gridStep;
};

void rotateSSE(__m128& borderVecs, const vec2& vec, float angle) {
    float cosine = cosf(angle * RAD);
    float sinus = sinf(angle * RAD);
    __m128 border = _mm_castpd_ps(_mm_load1_pd((double*)&vec));
    __m128 rotateMx = _mm_setr_ps(cosine, -sinus, sinus, cosine);
    __m128 rotated1 = _mm_mul_ps(border, rotateMx);
    rotateMx = _mm_permute_ps(rotateMx, _MM_SHUFFLE(3, 1, 2, 0));
    __m128 rotated2 = _mm_mul_ps(border, rotateMx);
    borderVecs = _mm_add_ps(_mm_permute_ps(rotated1, _MM_SHUFFLE(0, 3, 1, 2)), _mm_permute_ps(rotated2, _MM_SHUFFLE(1, 2, 0, 3)));
}

bool checkInCircleSSE(const __m128& sub, const __m128& squareRadius) {
    __m128 mulVecs = _mm_mul_ps(sub, sub);
    __m128 addVecs = _mm_add_ps(mulVecs, _mm_permute_ps(mulVecs, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_comile_ss(addVecs, squareRadius);
}

bool checkSectorSSE(const __m128& border, const __m128& vec) {
    __m128 mulVecs = _mm_mul_ps(border, vec);
    __m128 subVecs = _mm_sub_ps(mulVecs, _mm_movehdup_ps(mulVecs));
    __m128 cmp = _mm_cmplt_ps(subVecs, _mm_setzero_ps());
    return _mm_movemask_ps(cmp) == 4;
}

__m128 subtractSSE(const __m128& v1, const __m128& v2) {
    return _mm_sub_ps(v1, v2);
}

void fillGrid(Grid& grid, const std::vector<unit>& input_units) {
    float xmin = std::numeric_limits<float>::max();
    float xmax = std::numeric_limits<float>::lowest();
    float ymin = std::numeric_limits<float>::max();
    float ymax = std::numeric_limits<float>::lowest();

    for (const auto& unit : input_units) {
        xmin = std::min(xmin, unit.position.x);
        xmax = std::max(xmax, unit.position.x);
        ymin = std::min(ymin, unit.position.y);
        ymax = std::max(ymax, unit.position.y);
    }

    grid.stepX = (xmax - xmin) / GRID_SIZE;
    grid.stepY = (ymax - ymin) / GRID_SIZE;
    grid.xBorder = xmin;
    grid.yBorder = ymin;

    for (int i = 0; i < input_units.size(); ++i) {
        unsigned int xpos = (input_units[i].position.x - xmin) / grid.stepX;
        unsigned int ypos = (input_units[i].position.y - ymin) / grid.stepY;
        grid.grid[xpos][ypos].units.push_back(i);
    }
}

void checkUnitVision(const std::vector<unit>& input_units, std::vector<int>& result, int startFrom, int step, std::vector<__m128>& xmmpos, Grid& grid) {
    int size = input_units.size();
    for (int i = startFrom; i < size; i += step) {
        __m128 borderVecs128;
        rotateSSE(borderVecs128, input_units[i].direction, input_units[i].fov_deg / 2);

        float borders[4];
        _mm_store_ps(borders, borderVecs128);

        float xmin = input_units[i].position.x;
        float xmax = xmin;
        float ymin = input_units[i].position.y;
        float ymax = ymin;

        vec2 vr{
            (borders[1] * input_units[i].distance * COEFF) + input_units[i].position.x,
            (borders[0] * input_units[i].distance * COEFF) + input_units[i].position.y
        };

        vec2 vl{
            (borders[3] * input_units[i].distance * COEFF) + input_units[i].position.x,
            (borders[2] * input_units[i].distance * COEFF) + input_units[i].position.y
        };

        vec2 vd{
            (input_units[i].direction.x * input_units[i].distance * COEFF) + input_units[i].position.x,
            (input_units[i].direction.y * input_units[i].distance * COEFF) + input_units[i].position.y
        };

        xmin = std::min({ xmin, vr.x, vl.x, vd.x });
        xmax = std::max({ xmax, vr.x, vl.x, vd.x });
        ymin = std::min({ ymin, vr.y, vl.y, vd.y });
        ymax = std::max({ ymax, vr.y, vl.y, vd.y });

        int xposmin = std::max(0, int((xmin - grid.xBorder) / grid.stepX));
        int xposmax = std::min(GRID_SIZE, int((xmax - grid.xBorder) / grid.stepX));
        int yposmin = std::max(0, int((ymin - grid.yBorder) / grid.stepY));
        int yposmax = std::min(GRID_SIZE, int((ymax - grid.yBorder) / grid.stepY));

        float squareRadius = input_units[i].distance * input_units[i].distance;
        __m128 radius2 = _mm_set1_ps(squareRadius);

        for (int ix = xposmin; ix <= xposmax; ++ix) {
            for (int jx = yposmin; jx <= yposmax; ++jx) {
                for (int sz : grid.grid[ix][jx].units) {
                    __m128 sub = subtractSSE(xmmpos[sz], xmmpos[i]);
                    if (checkInCircleSSE(sub, radius2) && checkSectorSSE(borderVecs128, sub)) {
                        ++result[i];
                    }
                }
            }
        }
    }
}

void castVec2ToXmm(std::vector<__m128>& dst, const vec2& src) {
    dst.push_back(_mm_castpd_ps(_mm_load1_pd((double*)&src)));
}

void checkVisible(const std::vector<unit>& input_units, std::vector<int>& result) {
    Grid grid;
    fillGrid(grid, input_units);

    grid.gridBounds = _mm_castpd_ps(_mm_load_pd1((double*)&grid.xBorder));
    grid.gridStep = _mm_castpd_ps(_mm_load_pd1((double*)&grid.stepX));

    result.resize(input_units.size());

    int thread_num = std::thread::hardware_concurrency();
    std::vector<__m128> xmmpos;
    xmmpos.reserve(input_units.size());

    for (const auto& unit : input_units) {
        castVec2ToXmm(xmmpos, unit.position);
    }

    std::vector<std::thread> threads(thread_num - 1);

    for (int threadIdx = 0; threadIdx < thread_num - 1; ++threadIdx) {
        threads[threadIdx] = std::thread(checkUnitVision, std::cref(input_units), std::ref(result), threadIdx, thread_num, std::ref(xmmpos), std::ref(grid));
    }

    checkUnitVision(input_units, result, thread_num - 1, thread_num, xmmpos, grid);

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}


int main() {
    std::ifstream inputFile("input.txt");
    std::ofstream outputFile("output.txt");

    int N;
    double angle;
    double range;
    bool isFile;

    std::vector<unit> units;
    std::vector<int> results;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1000.0, 1000.0);

    std::cout << "Choose random data (0) or load file (1): ";
    std::cin >> isFile;

    if (isFile) {
        if (!inputFile.is_open()) {
            std::cout << "input.txt not exist";
            return 0;
        }

        inputFile >> N >> angle >> range;
        units.resize(N);
        results.resize(N);

        for (int i = 0; i < N; ++i) {
            inputFile >> units[i].position.x >> units[i].position.y >> units[i].direction.x >> units[i].direction.y;
            units[i].distance = range;
            units[i].fov_deg = angle;
        }
    }
    else {
        std::cout << "Input N: ";
        std::cin >> N;
        std::cout << "Input Angle: ";
        std::cin >> angle;
        std::cout << "Input Range: ";
        std::cin >> range;

        units.resize(N);
        results.resize(N);

        for (int i = 0; i < N; ++i) {
            units[i].position.x = dis(gen);
            units[i].position.y = dis(gen);
            units[i].direction.x = dis(gen);
            units[i].direction.y = dis(gen);
            units[i].distance = range;
            units[i].fov_deg = angle;
        }
    }
    auto start_time = std::chrono::high_resolution_clock::now();

    checkVisible(units, results);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

    for (int i = 0; i < N; ++i) {
        outputFile << results[i] << std::endl;
    }

    outputFile.close();

    return 0;
}
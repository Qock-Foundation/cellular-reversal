#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

const int kCACntCellTypes = 7;
const int N = 21, M = 19, H = 55, V = H * N * M, A = N * M;
const int kCAIOi0 = 4, kCAIOj0 = 5, kCAIOlen = 9;

std::atomic<int64_t> aTotalDiffScore, aTotalVolumeScore, aTotalBorderScore, aTotalSubstringsScore, aTotalTrashScore, aTotalAreaDiffScore, aTotalFiberizationScore, aTotalGenesScore, aTotalLastAreaScore;

class field {  // N * M
  uint8_t arr[N][M];
public:
  const uint8_t (&operator[](int index) const)[M] {
    return arr[index];
  }
  uint8_t (&operator[](int index))[M] {
    return arr[index];
  }
  friend std::ostream& operator<< (std::ostream& out, const field& f) {
    out << "field:\n";
    for (int i = 0; i < N; ++i) {
      out << " ";
      for (int j = 0; j < M; ++j) {
        out << char(f[i][j] == 0 ? '.' : 'A' + (f[i][j] - 1)) << " ";
      }
      out << "\n";
    }
    return out;
  }
};
void print_spacetime(std::ostream& out, const uint8_t arr[H][N][M]) {
  out << "spacetime:\n";
  for (int i = 0; i < N; ++i) {
    for (int t = 0; t < H; t += 6) {
      out << "  ";
      for (int j = 0; j < M; ++j) {
        out << char(arr[t][i][j] == 0 ? '.' : 'A' + (arr[t][i][j] - 1));
      }
    }
    out << "\n";
  }
}

class creature {  // 3D CA
  static inline const int kCACntGenes = kCACntCellTypes * kCACntCellTypes * kCACntCellTypes
                                      * kCACntCellTypes * kCACntCellTypes;
  static inline const int kCARecommendedCntGenes = 40;
  uint8_t genes[kCACntGenes];
  static int min(int a, int b, int c) {
    return std::min(a, std::min(b, c));
  }
  static int levenstein(const uint8_t *s, const uint8_t *t) {
    int n = kCAIOlen;  // length of both s and t
    int dst[n+1][n+1];
    for (int i = 0; i <= n; ++i) {
      dst[i][0] = i;
    }
    for (int j = 0; j <= n; ++j) {
      dst[0][j] = j;
    }
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        dst[i+1][j+1] = s[i] == t[j] ? dst[i][j] : min(dst[i][j], dst[i+1][j], dst[i][j+1]) + 1;
      }
    }
    return dst[n][n];
  }
  static int strdiff(const uint8_t *s, const uint8_t *t) {
    int diff = 0;
    int n = kCAIOlen;  // length of both s and t
    for (int i = 0; i < n; ++i) {
      diff += s[i] != t[i];
    }
    return diff;
  }
  static std::pair<int, int> rotated_substrings_heuristics(const uint8_t arr[H][N][M], const uint8_t str[kCAIOlen]) {
    int score = 0;
    int direct[kCAIOlen+1] = {}, diagonal_easy[kCAIOlen+1] = {}, perpendicular[kCAIOlen+1] = {}, diagonal_hard[kCAIOlen+1] = {}, reversed[kCAIOlen+1] = {};
    for (int t = 0; t < H; ++t) {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          for (int offset = 0; offset < kCAIOlen - 1; ++offset) {  // trying to match arr[t][i][j].. against str[offset]..
            int len = 0;
            for (int u = 0; j + u < M && offset + u < kCAIOlen && arr[t][i][j+u] == str[offset+u]; ++u) {  // matching rightwards
              len++;
            }
            direct[len]++;
            len = 0;
            for (int u = 0; i - u >= 0 && offset + u < kCAIOlen && arr[t][i-u][j] == str[offset+u]; ++u) {  // matching upwards
              len++;
            }
            perpendicular[len]++;
            len = 0;
            for (int u = 0; i + u < N && offset + u < kCAIOlen && arr[t][i+u][j] == str[offset+u]; ++u) {  // matching downwards
              len++;
            }
            perpendicular[len]++;
            len = 0;
            for (int u = 0; j - u >= 0 && offset + u < kCAIOlen && arr[t][i][j-u] == str[offset+u]; ++u) {  // matching leftwards!
              len++;
            }
            reversed[len]++;
            len = 0;
            for (int u = 0; i + u < N && j - u >= 0 && offset + u < kCAIOlen && arr[t][i+u][j-u] == str[offset+u]; ++u) {  // matching right-up-wards
              len++;
            }
            diagonal_easy[len]++;
            len = 0;
            for (int u = 0; i + u < N && j + u < N && offset + u < kCAIOlen && arr[t][i+u][j+u] == str[offset+u]; ++u) {  // matching right-down-wards
              len++;
            }
            diagonal_easy[len]++;
            len = 0;
            for (int u = 0; i - u >= 0 && j - u >= 0 && offset + u < kCAIOlen && arr[t][i-u][j-u] == str[offset+u]; ++u) {  // matching left-up-wards
              len++;
            }
            diagonal_hard[len]++;
            len = 0;
            for (int u = 0; i - u >= 0 && j + u < N && offset + u < kCAIOlen && arr[t][i-u][j+u] == str[offset+u]; ++u) {  // matching left-down-wards
              len++;
            }
            diagonal_hard[len]++;
          }
        }
      }
    }
    for (int len = 4; len <= kCAIOlen; ++len) {
      score += direct[len] <= H * kCAIOlen / len ?
          (len <= 3 ? 0 : len == 4 ? 1 : len == 5 ? 2 : len == 6 ? 4 : len == 7 ? 8 : 16) * std::min(H * kCAIOlen / len, direct[len]) : 0;  //-direct[len] * direct[len] / H / H * 5;
      score += diagonal_easy[len] <= 2 * H * kCAIOlen / len ?
          (len == 3 ? 1 : len == 4 ? 2 : len == 5 ? 4 : len == 6 ? 8 : len == 7 ? 16 : 32) * std::min(H * kCAIOlen / len, diagonal_easy[len]) : 0;
      /*score += perpendicular[len] <= 2 * H * kCAIOlen / len ?
          (len == 3 ? 2 : len == 4 ? 4 : len == 5 ? 8 : len == 6 ? 16 : len == 7 ? 32 : 64) * std::min(H * kCAIOlen / len, perpendicular[len]) : 0;  //-perpendicular[len] * perpendicular[len] / H / H * 3;
      score += diagonal_hard[len] <= 2 * H * kCAIOlen / len ?
          (len == 3 ? 4 : len == 4 ? 8 : len == 5 ? 16 : len == 6 ? 32 : len == 7 ? 64 : 128) * std::min(H * kCAIOlen / len, diagonal_hard[len]) : 0;
      score += reversed[len] <= 2 * H * kCAIOlen / len ?
          (len == 3 ? 8 : len == 4 ? 16 : len == 5 ? 32 : len == 6 ? 64 : len == 7 ? 128 : 256) * std::min(H * kCAIOlen / len, reversed[len]) : 0;  //-reversed[len] * reversed[len] / H / H * 3;*/
      // aiming at rotating by pi/2 only, for now; we don't want to mess up with so many bonuses interventing each other
    }
    int penalty = 0;  // for excessive chunks
    for (int t = 0; t < H; ++t) {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          if (arr[t][i][j] == 0 or arr[t][i][j] > 3) {
            continue;
          }
          penalty += i < N - 1 && 1 <= arr[t][i+1][j] && arr[t][i+1][j] <= 3;
          penalty += i < N - 1 && j < M - 1 && 1 <= arr[t][i+1][j+1] && arr[t][i+1][j+1] <= 3;
          penalty += j < M - 1 && 1 <= arr[t][i][j+1] && arr[t][i][j+1] <= 3;
          penalty += i > 0 && j < M - 1 && 1 <= arr[t][i-1][j+1] && arr[t][i-1][j+1] <= 3;
        }
      }
    }
    return {score, penalty};
  }
  static double rms_area_diff(const uint8_t arr[H][N][M]) {
    int cnt[H] = {};
    for (int t = 0; t < H; ++t) {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          cnt[t] += arr[t][i][j] != 0;
        }
      }
      if (t >= H / 4 && t < 3 * H / 4) {  // kostyl
        cnt[t] /= 2;
      }
    }
    double avg = 0;
    for (int t = 0; t < H; ++t) {
      avg += cnt[t];
    }
    avg /= double(H);
    double rms = 0;
    for (int t = 0; t < H; ++t) {
      rms += (cnt[t] - avg) * (cnt[t] - avg);
    }
    rms = sqrt(rms / H);
    return rms;
  }
  static int fiberization_score(const uint8_t arr[H][N][M]) {
    int score = 0;
    for (int t = 0; t < H; ++t) {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          if (arr[t][i][j] == 0) {
            continue;
          }
          bool mask[3][3] = {
            {i > 0 && j > 0 && arr[t][i-1][j-1], i > 0 && arr[t][i-1][j], i > 0 && j < M - 1 && arr[t][i-1][j+1]},
            {j > 0 && arr[t][i][j-1], 1, j < M - 1 && arr[t][i][j+1]},
            {i < N - 1 && j > 0 && arr[t][i+1][j-1], i < N - 1 && arr[t][i+1][j], i < N - 1 && j < M - 1 && arr[t][i+1][j+1]}
          };
          static const int bonus[9] = {-10, -3, 5, 2, -1, -2, -6, -10, -20};
          score += bonus[mask[0][0]+mask[0][1]+mask[0][2]+mask[1][0]+mask[1][2]+mask[2][0]+mask[2][1]+mask[2][2]];
          score += mask[0][0] == 0 && mask[0][1] == 0 && mask[0][2] == 0 && mask[1][0] == 1 && mask[1][2] == 1 && mask[2][0] == 0 && mask[2][1] == 0 && mask[2][2] == 0 ? 4 : 0;
          score += mask[0][0] == 0 && mask[0][1] == 1 && mask[0][2] == 0 && mask[1][0] == 0 && mask[1][2] == 0 && mask[2][0] == 0 && mask[2][1] == 1 && mask[2][2] == 0 ? 4 : 0;
          score += mask[0][0] == 1 && mask[0][1] == 0 && mask[0][2] == 0 && mask[1][0] == 0 && mask[1][2] == 0 && mask[2][0] == 0 && mask[2][1] == 0 && mask[2][2] == 1 ? 3 : 0;
          score += mask[0][0] == 0 && mask[0][1] == 0 && mask[0][2] == 1 && mask[1][0] == 0 && mask[1][2] == 0 && mask[2][0] == 1 && mask[2][1] == 0 && mask[2][2] == 0 ? 3 : 0;
        }
      }
    }
    if (rand() % 30000 == 0)
    std::cout << "fibscore " << score << std::endl;
    return score;
  }
  double personal_loss() {  // like "penalize nonzero genes", etc
    int64_t cnt_nonzero = 0;
    for (int i = 0; i < kCACntGenes; ++i) {
      cnt_nonzero += genes[i] != 0;
    }
    if (cnt_nonzero > 2 * kCARecommendedCntGenes) {  // 324 of 16807
      return cnt_nonzero * cnt_nonzero / (2.0 * kCARecommendedCntGenes) / (2 * kCARecommendedCntGenes);
    }
    return 0;
  }
public:
  creature(std::mt19937& generator) {  // random, 2x oversampling compared to recommended size
    for (int i = 0; i < kCACntGenes; ++i) {
      int u = i;
      bool ok = true;
      while (u > 0) {
        ok &= u % kCACntCellTypes < 4;
        u /= kCACntCellTypes;
      }
      genes[i] = ok || generator() % kCACntGenes < kCARecommendedCntGenes ? generator() % kCACntCellTypes : 0;
      //genes[i] = generator() % kCACntCellTypes;
    }
  }
  uint8_t operator()(uint8_t a01, uint8_t a10, uint8_t a11, uint8_t a12, uint8_t a21) {
    return genes[(((a01 * 3 + a10) * 3 + a11) * 3 + a12) * 3 + a21];
  }
  field run(const field F, uint8_t arr[H][N][M]) {  // field, cnt nonzero cells in the run, cnt border cells in the run
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        arr[0][i][j] = F[i][j];
      }
    }
    for (int t = 0; t < H - 1; ++t) {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          arr[t+1][i][j] = (*this)(
              i > 0 ? arr[t][i-1][j] : 0,
              j > 0 ? arr[t][i][j-1] : 0, arr[t][i][j], j < M - 1 ? arr[t][i][j+1] : 0,
              i < N - 1 ? arr[t][i+1][j] : 0
          );
        }
      }
    }
    field R;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        R[i][j] = arr[H-1][i][j];
      }
    }
    if (rand() % 100000 == 0) {
      print_spacetime(std::cout, arr);
    }
    return R;
  }
  creature sex(const creature& another, std::mt19937& generator) {  // another is const
    creature x(generator);
    int cnt_nonzero = 0;
    for (int i = 0; i < kCACntGenes; ++i) {
      x.genes[i] = generator() % 2 ? genes[i] : another.genes[i];
      cnt_nonzero += x.genes[i] != 0;
    }
    // and mutations
    static const double kCAFracMutations = 0.01;
    for (int i = 0; i < kCACntGenes; ++i) {
      if (generator() % 100000 < kCAFracMutations * 100000) {
        x.genes[i] = cnt_nonzero < kCARecommendedCntGenes ? generator() % kCACntCellTypes : 0;
      }
    }
    return x;
  }
  void learn(int time, std::mt19937& generator) {  // randomly change genes to maximize the score
    int prev_score = score(1, generator);
    for (int iter = 0; iter < time; ++iter) {
      int i = generator() % kCACntGenes, r = rand() % kCACntCellTypes;
      int oldr = genes[i];
      if (r == oldr) {
        continue;
      }
      genes[i] = r;
      int new_score = score(1, generator);
      if (new_score < prev_score) {
        genes[i] = oldr;
        prev_score = new_score;
      }
    }
  }
  //creature operator&(const creature& another) {
  //  return sex(another);
  //}
  int64_t score(int trials, std::mt19937& generator) {  // samples situations, estimates the creature
    aTotalDiffScore += 1;
    int64_t score = 0;
    for (int situation = 0; situation < trials; ++situation) {
      field F;
      uint8_t s[M], t[M];
      for (int i = 0; i < kCAIOlen; ++i) {
        s[i] = 1 + generator() % 2;
      }
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          F[i][j] = 0;
        }
      }
      for (int j = 0; j < kCAIOlen; ++j) {
        F[kCAIOi0][kCAIOj0+j] = s[j];
      }
      assert(kCAIOi0 < N - 1 && kCAIOj0 >= 1 && kCAIOj0 + kCAIOlen < M);
      F[kCAIOi0+1][kCAIOj0-1] = 4;        // start codon: 'D'
      F[kCAIOi0+1][kCAIOj0+kCAIOlen] = 5;  // stop codon: 'E'
      uint8_t spacetime[H][N][M];
      const field G = run(F, spacetime);
      {
        static const int kCADiffLoss = 15;
        for (int j = 0; j < kCAIOlen; ++j) {
          t[j] = G[kCAIOi0][kCAIOj0+j];
        }
        for (int i = 0; i < kCAIOlen / 2; ++i) {
          std::swap(t[i], t[kCAIOlen-1-i]);
        }
        auto tmp = -kCADiffLoss * levenstein(s, t);
        score += tmp;
        aTotalDiffScore += tmp;
      }
      int cnt_nonzero_cells = 0;
      for (int t = 0; t < H; ++t) {
        for (int i = 0; i < N; ++i) {
          for (int j = 0; j < M; ++j) {
            cnt_nonzero_cells += spacetime[t][i][j] != 0;
          }
        }
      }
      static const double kCAFracVolume = 0.1;
      {
        static const int kCAVolumeLoss = 1;
        auto tmp = (double)cnt_nonzero_cells / V < kCAFracVolume ? -10000
            : -(log((double)cnt_nonzero_cells / V) - log(kCAFracVolume))
            * (log((double)cnt_nonzero_cells / V) - log(kCAFracVolume))
            * kCAVolumeLoss;
        score += tmp;
        aTotalVolumeScore += tmp;
      }
      {
        static const int BA = 2 * (H * N + N * M + M * H);
        static const int kCABorderAreaLoss = 3000;
        int cnt_border_cells = 0;  // they are strongly condemned
        for (int t = 0; t < H; ++t) {
          for (int i = 0; i < N; ++i) {
            cnt_border_cells += (spacetime[t][i][0] != 0) + (spacetime[t][i][M-1] != 0);
          }
          for (int j = 0; j < M; ++j) {
            cnt_border_cells += (spacetime[t][0][j] != 0) + (spacetime[t][N-1][j] != 0);
          }
        }
        auto tmp = -cnt_border_cells / double(BA) * kCABorderAreaLoss;
        score += tmp;
        aTotalBorderScore += tmp;
      }
      {
        static const int kCALastAreaLoss = 10;
        int area = 0;
        for (int i = 0; i < N; ++i) {
          for (int j = 0; j < M; ++j) {
            if (i == kCAIOi0 && j >= kCAIOj0 && j - kCAIOj0 < kCAIOlen) {
              continue;
            }
            area += G[i][j] != 0;
            /*if (G[i][j] != 0) {
              int di = std::abs(i - kCAIOi0), dj = j < kCAIOj0 ? kCAIOj0 - j : j >= kCAIOj0 + kCAIOlen ? j - kCAIOj0 - kCAIOlen - 1 : 0;
              score -= (di * di * di + dj * dj * dj) * kCALastLayerEccentricLoss;
              if (di * di + dj * dj < 3 * 3) {
                score += kCALastLayerCentricBonus;
              }
            }*/
          }
        }
        score -= area * kCALastAreaLoss;
        aTotalLastAreaScore -= area * kCALastAreaLoss;
      }
      /*static const double kCAFracLastArea = 0.3;
      static const int kCALastAreaLoss = 500;
      score -= ((double)cnt_nonzero_last_layer / A - kCAFracLastArea) * kCAFracLastArea
          * ((double)cnt_nonzero_last_layer / A - kCAFracLastArea) * kCAFracLastArea
          * kCALastAreaLoss;*/
      {
        static const double kCARotatedSubstringsBonus = 200;
        static const double kCARotatedTrashLoss = 9;
        auto score_and_penalty = rotated_substrings_heuristics(spacetime, s);
        auto tmp = score_and_penalty.first / double(H) * kCARotatedSubstringsBonus;
        score += tmp;
        aTotalSubstringsScore += tmp;
        tmp = -score_and_penalty.second / double(H) * kCARotatedTrashLoss;
        score += tmp;
        aTotalTrashScore += tmp;
      }
      {
        static const double kCAAreaDiffLoss = 40;
        auto tmp = -rms_area_diff(spacetime) / (kCAFracVolume * A) * kCAAreaDiffLoss;
        score += tmp;
        aTotalAreaDiffScore += tmp;
      }
      {
        static const double kCAFiberizationBonus = 80;
        if (rand() % 30000 == 0) {
          std::cout << "fs " << fiberization_score(spacetime) << " / " << cnt_nonzero_cells << " * " << kCAFiberizationBonus << "\n";
        }
        auto tmp = fiberization_score(spacetime) / cnt_nonzero_cells * kCAFiberizationBonus;
        score += tmp;
        aTotalFiberizationScore += tmp;
      }
      {
        static const double kCAGenesLoss = 0.1;
        auto tmp = -personal_loss() * kCAGenesLoss;
        score += tmp;
        aTotalGenesScore += tmp;
      }
    }
    return score;
  }
};

int main() {
  const int kCAGenerationSize = 1000;
  const double kCAFracGenerationLeave = 0.20;
  const double kCAFracGenerationEligible = 0.40;
  const int kCACreatureAssessments = 10;
  const int kCALearningTime = 10;
  unsigned int threads_cnt = std::thread::hardware_concurrency();
  std::mt19937 host_generator, thread_generators[threads_cnt];
  std::vector<creature> generation(kCAGenerationSize, host_generator);
  for (int T = 0;; ++T) {
    const int kCAGenerationLeave = kCAFracGenerationLeave * kCAGenerationSize;
    const int kCAGenerationEligible = kCAFracGenerationEligible * kCAGenerationSize;
    std::cout << " GENERATION " << T << std::endl;
    std::cout << "phase assessment" << std::endl;
    std::vector<std::pair<int64_t, int>> scores;
    std::vector<std::thread> pool;
    std::mutex mutex;
    aTotalDiffScore = 0, aTotalVolumeScore = 0, aTotalBorderScore = 0, aTotalSubstringsScore = 0, aTotalTrashScore = 0,
                    aTotalAreaDiffScore = 0, aTotalFiberizationScore = 0, aTotalGenesScore = 0, aTotalLastAreaScore = 0;
    for (int thread_id = 0; thread_id < threads_cnt; ++thread_id) {
      pool.emplace_back([&](int thread_i) {
        aTotalBorderScore += 2;
        for (int i = 0; i < kCAGenerationSize; ++i) {
          if (i % threads_cnt != thread_i) {
            continue;
          }
          int64_t score = generation[i].score(kCACreatureAssessments, thread_generators[thread_i]);
          mutex.lock();
          scores.emplace_back(score, i);
          mutex.unlock();
        }
      }, thread_id);
    }
    for (auto& j : pool) {
      j.join();
    }
    pool.clear();
    std::sort(scores.begin(), scores.end());
    std::reverse(scores.begin(), scores.end());
    std::cout << "scores: ";
    for (int i = 0; i < 6; ++i) {
      std::cout << scores[i].first << "," << scores[i].second << "\t";
    }
    std::cout << "...";
    for (int i = 3; i > 0; --i) {
      std::cout << "\t" << scores[kCAGenerationSize-i].first << "," << scores[kCAGenerationSize-i].second;
    }
    std::cout << "\n";
    std::cout << "Total Score: " << std::accumulate(scores.begin(), scores.end(), 0ll, [](int64_t acc, auto pair) { return acc + pair.first; }) << "\n";
    static const double cnt_assessments = kCAGenerationSize * kCACreatureAssessments;
    std::cout << "Avg Diff Score (per assessment):\t\t" << aTotalDiffScore / cnt_assessments << "\n";
    std::cout << "Avg Volume Score (per assessment):\t\t" << aTotalVolumeScore / cnt_assessments << "\n";
    std::cout << "Avg Border Score (per assessment):\t\t" << aTotalBorderScore / cnt_assessments << "\n";
    std::cout << "Avg Substrings Score (per assessment):\t\t" << aTotalSubstringsScore / cnt_assessments << "\n";
    std::cout << "Avg Trash Score (per assessment):\t\t" << aTotalTrashScore / cnt_assessments << "\n";
    std::cout << "Avg Area Diff Score (per assessment):\t\t" << aTotalAreaDiffScore / cnt_assessments << "\n";
    std::cout << "Avg Fiberization Score (per assessment):\t" << aTotalFiberizationScore / cnt_assessments << "\n";
    std::cout << "Avg Genes Score (per assessment):\t\t" << aTotalGenesScore / cnt_assessments << "\n";
    std::cout << "Avg Last Area Score (per assessment):\t\t" << aTotalLastAreaScore / cnt_assessments << "\n";
    std::cout << "phase sex" << std::endl;
    std::vector<creature> new_generation;
    for (int i = 0; i < kCAGenerationLeave; ++i) {
      new_generation.push_back(generation[scores[i].second]);
    }
    int left = kCAGenerationSize - new_generation.size();
    std::vector<creature> new_generation_part[threads_cnt];
    for (int thread_id = 0; thread_id < threads_cnt; ++thread_id) {
      pool.emplace_back([&](int thread_i) {
        for (int i = 0; i < left; ++i) {
          if (i % threads_cnt != thread_i) {
            continue;
          }
          creature c = generation[scores[thread_generators[thread_i]()%kCAGenerationEligible].second]
              .sex(generation[scores[thread_generators[thread_i]()%kCAGenerationEligible].second], thread_generators[thread_i]);
          new_generation_part[thread_i].push_back(c);
        }
      }, thread_id);
    }
    for (auto& j : pool) {
      j.join();
    }
    pool.clear();
    for (int thread_id = 0; thread_id < threads_cnt; ++thread_id) {
      std::copy(new_generation_part[thread_id].begin(), new_generation_part[thread_id].end(), std::back_inserter(new_generation));
      new_generation_part[thread_id].clear();
    }
    generation.swap(new_generation);
    std::cout << "phase learning" << std::endl;
    for (int thread_id = 0; thread_id < threads_cnt; ++thread_id) {
      pool.emplace_back([&](int thread_i) {
        for (int i = 0; i < kCAGenerationSize; ++i) {
          if (i % threads_cnt != thread_i) {
            continue;
          }
          generation[i].learn(kCALearningTime, thread_generators[thread_i]);
        }
      }, thread_id);
    }
    for (auto& j : pool) {
      j.join();
    }
  }
}


#include <algorithm>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

const int kCACntCellTypes = 6;
const int N = 5, M = 5, H = 6;
const int kCAIOi0 = 2, kCAIOj0 = 1, kCAIOlen = 3;

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
      out << "\t";
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
    out << "\t";
    for (int t = 0; t < H; ++t) {
      for (int j = 0; j < M; ++j) {
        out << char(arr[t][i][j] == 0 ? '.' : 'A' + (arr[t][i][j] - 1)) << " ";
      }
      out << "\t\t";
    }
    out << "\n";
  }
}

class creature {  // 3D CA
  static inline const int kCACntGenes = kCACntCellTypes * kCACntCellTypes * kCACntCellTypes
                                      * kCACntCellTypes * kCACntCellTypes;
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
public:
  creature(std::mt19937& generator) {  // random
    for (int i = 0; i < kCACntGenes; ++i) {
      genes[i] = generator() % kCACntCellTypes;
    }
  }
  /*uint8_t operator()(uint8_t a00, uint8_t a01, uint8_t a02, uint8_t a10, uint8_t a11, uint8_t a12, uint8_t a20, uint8_t a21, uint8_t a22) const {
    //std::cout << (int)a00 << " " << (int)a01 << " " << (int)a02 << " " << (int)a10 << " " << (int)a11 << " " << (int)a12 << " " << (int)a20 << " " << (int)a21 << " " << (int)a22 << std::endl;
    return genes[(((((((a00 * 3 + a01) * 3 + a02) * 3 + a10) * 3 + a11) * 3 + a12) * 3 + a20) * 3 + a21) * 3 + a22];
  }*/
  uint8_t operator()(uint8_t a01, uint8_t a10, uint8_t a11, uint8_t a12, uint8_t a21) {
    return genes[(((a01 * 3 + a10) * 3 + a11) * 3 + a12) * 3 + a21];
  }
  field run(const field F) {
    uint8_t arr[H][N][M];
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        arr[0][i][j] = F[i][j];
      }
    }
    for (int t = 0; t < H - 1; ++t) {
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          /*arr[t+1][i][j] = (*this)(
              i > 0 && j > 0 ? arr[t][i-1][j-1] : 0, i > 0 ? arr[t][i-1][j] : 0, i > 0 && j < M - 1 ? arr[t][i-1][j+1] : 0,
              j > 0 ? arr[t][i][j-1] : 0, arr[t][i][j], j < M - 1 ? arr[t][i][j+1] : 0,
              i < N - 1 && j > 0 ? arr[t][i+1][j-1] : 0, i < N - 1 ? arr[t][i+1][j] : 0, i < N - 1 && j < M - 1 ? arr[t][i+1][j+1] : 0
          );*/
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
    if (rand() % 10000000 == 0) {
      //std::cout << F << "to\n" << R << "\n";
      print_spacetime(std::cout, arr);
    }
    return R;
  }
  field operator()(const field F) {
    return run(F);
  }
  creature sex(const creature& another, std::mt19937& generator) {  // another is const
    creature x(generator);
    for (int i = 0; i < kCACntGenes; ++i) {
      x.genes[i] = generator() % 2 ? genes[i] : another.genes[i];
    }
    // and mutations
    static const double kCAFracMutations = 0.01;
    for (int i = 0; i < kCACntGenes; ++i) {
      if (generator() % 100000 < kCAFracMutations * 100000) {
        x.genes[i] = generator() % kCACntCellTypes;
      }
    }
    return x;
  }
  //creature operator&(const creature& another) {
  //  return sex(another);
  //}
  int64_t score(int trials, std::mt19937& generator) {  // samples situations, estimates the creature
    int64_t score = 0;
    for (int situation = 0; situation < trials; ++situation) {
      field F;
      uint8_t s[kCAIOlen], t[kCAIOlen];
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
      field G = run(F);
      for (int j = 0; j < kCAIOlen; ++j) {
        t[j] = G[kCAIOi0][kCAIOj0+j];
      }
      for (int i = 0; i < kCAIOlen / 2; ++i) {
        std::swap(t[i], t[kCAIOlen-1-i]);
      }
      score -= 10 * levenstein(s, t);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
          if (i == kCAIOi0 && j >= kCAIOj0 && j - kCAIOj0 < kCAIOlen) {
            continue;
          }
          score -= G[i][j] != 0;
        }
      }
    }
    return score;
  }
};

int main() {
  const int kCAGenerationSize = 100000;
  const int kCAGenerationsCnt = 10000000;
  const int kCAGenerationLeave = 20000;
  const int kCAGenerationEligible = 40000;
  const int kCACreatureAssessment = 30;
  unsigned int threads_cnt = std::thread::hardware_concurrency();
  //unsigned int threads_cnt = 1;
  std::mt19937 host_generator, thread_generators[threads_cnt];
  std::vector<creature> generation(kCAGenerationSize, host_generator);
  for (int T = 0; T < kCAGenerationsCnt; ++T) {
    std::cout << "phase assessment" << std::endl;
    std::vector<std::pair<int64_t, int>> scores;
    std::vector<std::jthread> pool;
    std::mutex mutex;
    for (int thread_id = 0; thread_id < threads_cnt; ++thread_id) {
      pool.emplace_back([&](int thread_i) {
        for (int i = 0; i < kCAGenerationSize; ++i) {
          if (i % threads_cnt != thread_i) {
            continue;
          }
          int64_t score = generation[i].score(kCACreatureAssessment, thread_generators[thread_i]);
          mutex.lock();
          scores.emplace_back(score, i);
          mutex.unlock();
        }
      }, thread_id);
    }
    pool.clear();  // execute them all
    std::sort(scores.begin(), scores.end());
    std::reverse(scores.begin(), scores.end());
    //for (const auto [score, i] : scores) {
    //  std::cout << score << " ";
    //}
    std::cout << "scores: ";
    for (int i = 0; i < 7; ++i) {
      std::cout << scores[i].first << "," << scores[i].second << "\t";
    }
    std::cout << "...\n";
    std::cout << std::accumulate(scores.begin(), scores.end(), 0ll, [](int64_t acc, auto pair) { return acc + pair.first; }) << std::endl;
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
    pool.clear();  // execute them all
    for (int thread_id = 0; thread_id < threads_cnt; ++thread_id) {
      std::copy(new_generation_part[thread_id].begin(), new_generation_part[thread_id].end(), std::back_inserter(new_generation));
      new_generation_part[thread_id].clear();
    }
    generation.swap(new_generation);
  }
}


#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

const int M = 40;  // cnt memory states (~2^(log n)^2)
const int n = 6;   // cnt vertices, source=0, target=n-1
const int N = 10; // cnt graphs (tests)
double p_edge;

const int cnt_vars_per_mem_state = 2 * M + 2 * n;
int var_q_from(int mem_state, int v) {
  return mem_state * cnt_vars_per_mem_state + v + 1;
}
int var_q_to(int mem_state, int v) {
  return mem_state * cnt_vars_per_mem_state + n + v + 1;
}
int var_transition_if_0(int mem_from, int mem_to) {
  return mem_from * cnt_vars_per_mem_state + n + n + mem_to + 1;
}
int var_transition_if_1(int mem_from, int mem_to) {
  return mem_from * cnt_vars_per_mem_state + n + n + M + mem_to + 1;
}
const int cnt_vars_algo = M * cnt_vars_per_mem_state;

const int max_solving_time = M;
const int cnt_vars_per_graph = (max_solving_time + 1) * (M + 2 * n + 1);
int var_graph_being(int graph_id, int time_moment, int mem_state) {
  return cnt_vars_algo + graph_id * cnt_vars_per_graph + time_moment * (M + 2 * n + 1) + mem_state + 1;
}
int var_graph_qfrom(int graph_id, int time_moment, int v) {
  return cnt_vars_algo + graph_id * cnt_vars_per_graph + time_moment * (M + 2 * n + 1) + M + v + 1;
}
int var_graph_qto(int graph_id, int time_moment, int v) {
  return cnt_vars_algo + graph_id * cnt_vars_per_graph + time_moment * (M + 2 * n + 1) + M + n + v + 1;
}
int var_graph_qans(int graph_id, int time_moment) {
  return cnt_vars_algo + graph_id * cnt_vars_per_graph + time_moment * (M + 2 * n + 1) + M + n + n + 1;
}
const int cnt_vars_graph_runs = N * cnt_vars_per_graph;

bool is_reachable(const bool graph[n][n], int s, int t) {
  bool used[n] = {};
  used[s] = true;
  std::queue<int> q;
  q.push(s);
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    for (int v = 0; v < n; ++v) {
      if (graph[u][v] && !used[v]) {
        used[v] = true;
        q.push(v);
      }
    }
  }
  return used[t];
}

void set_p_edge(std::mt19937& gen) {
  double l = 0, r = 1;
  while (r - l > 1e-6) {
    p_edge = (l + r) / 2;
    int cnt = 0;
    for (int id = 0; id < 10000; ++id) {
      bool graph[n][n];
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          graph[i][j] = gen() % 1000000 / 1e6 < p_edge;
        }
      }
      cnt += is_reachable(graph, 0, 1);
    }
    (cnt < 5000 ? l : r) = p_edge;
  }
}

int main(int argc, char **argv) {
  assert(argc == 1);
  srand(time(0));
  std::mt19937 gen(rand());
  set_p_edge(gen);
  std::cout << "p_edge = " << p_edge << std::endl;
  std::vector<std::vector<int>> problem;

  /// DESCRIPTION OF OUR ALGORITHM
  std::cout << "description of our algorithm..." << std::endl;
  // for each memory state, exactly one edge should be questioned about, O(M * n^2):
  for (int mem_state = 0; mem_state < M; ++mem_state) {
    problem.emplace_back();
    for (int v = 0; v < n; ++v) {
      problem.back().push_back(var_q_from(mem_state, v));
    }
    for (int u = 0; u < n; ++u) {
      for (int v = 0; v < u; ++v) {
        problem.push_back({-var_q_from(mem_state, u), -var_q_from(mem_state, v)});
      }
    }
    problem.emplace_back();
    for (int v = 0; v < n; ++v) {
      problem.back().push_back(var_q_to(mem_state, v));
    }
    for (int u = 0; u < n; ++u) {
      for (int v = 0; v < u; ++v) {
        problem.push_back({-var_q_to(mem_state, u), -var_q_to(mem_state, v)});
      }
    }
  }
  // for each memory state, exactly one transition should happen for both answers, O(M^3):
  for (int mem_from = 0; mem_from < M; ++mem_from) {
    problem.emplace_back();
    for (int mem_to = 0; mem_to < M; ++mem_to) {
      problem.back().push_back(var_transition_if_0(mem_from, mem_to));
    }
    for (int mem_to_1 = 0; mem_to_1 < M; ++mem_to_1) {
      for (int mem_to_2 = 0; mem_to_2 < mem_to_1; ++mem_to_2) {
        problem.push_back({-var_transition_if_0(mem_from, mem_to_1), -var_transition_if_0(mem_from, mem_to_2)});
      }
    }
    problem.emplace_back();
    for (int mem_to = 0; mem_to < M; ++mem_to) {
      problem.back().push_back(var_transition_if_1(mem_from, mem_to));
    }
    for (int mem_to_1 = 0; mem_to_1 < M; ++mem_to_1) {
      for (int mem_to_2 = 0; mem_to_2 < mem_to_1; ++mem_to_2) {
        problem.push_back({-var_transition_if_1(mem_from, mem_to_1), -var_transition_if_1(mem_from, mem_to_2)});
      }
    }
  }

  /// DESCRIPTION OF OUR TEST GRAPHS
  std::cout << "description of our test graphs..." << std::endl;
  bool test_graphs[N][n][n], test_graphs_answers[N];
  for (int graph_id = 0; graph_id < N; ++graph_id) {
    for (int u = 0; u < n; ++u) {
      for (int v = 0; v < n; ++v) {
        test_graphs[graph_id][u][v] = gen() % 1000000 / 1e6 < p_edge;
      }
    }
    test_graphs_answers[graph_id] = is_reachable(test_graphs[graph_id], 0, n - 1);
  }

  /// DESCRIPTION OF OUR ALGORITHM RUN ON OUR TEST GRAPHS
  std::cout << "description of our algorithm run on our test graphs..." << std::endl;
  // for each graph and time moment...
  for (int graph_id = 0; graph_id < N; ++graph_id) {
    problem.push_back({var_graph_being(graph_id, 0, 0)});  // otherwise it was kidding me with 0-1-0-1 chains
    for (int time_moment = 0; time_moment < max_solving_time; ++time_moment) {
      // we should be in exactly one state
      problem.emplace_back();
      for (int mem_state = 0; mem_state < M; ++mem_state) {
        problem.back().push_back(var_graph_being(graph_id, time_moment, mem_state));
      }
      for (int mem_state_1 = 0; mem_state_1 < M; ++mem_state_1) {
        for (int mem_state_2 = 0; mem_state_2 < mem_state_1; ++mem_state_2) {
          problem.push_back({-var_graph_being(graph_id, time_moment, mem_state_1), -var_graph_being(graph_id, time_moment, mem_state_2)});
        }
      }
      // we should question about exactly one edge
      problem.emplace_back();
      for (int q_from = 0; q_from < n; ++q_from) {
        problem.back().push_back(var_graph_qfrom(graph_id, time_moment, q_from));
      }
      for (int q_from_1 = 0; q_from_1 < n; ++q_from_1) {
        for (int q_from_2 = 0; q_from_2 < q_from_1; ++q_from_2) {
          problem.push_back({-var_graph_qfrom(graph_id, time_moment, q_from_1), -var_graph_qfrom(graph_id, time_moment, q_from_2)});
        }
      }
      problem.emplace_back();
      for (int q_to = 0; q_to < n; ++q_to) {
        problem.back().push_back(var_graph_qto(graph_id, time_moment, q_to));
      }
      for (int q_to_1 = 0; q_to_1 < n; ++q_to_1) {
        for (int q_to_2 = 0; q_to_2 < q_to_1; ++q_to_2) {
          problem.push_back({-var_graph_qto(graph_id, time_moment, q_to_1), -var_graph_qto(graph_id, time_moment, q_to_2)});
        }
      }
      // our question should be as documented
      for (int mem_state = 0; mem_state < M; ++mem_state) {
        for (int q_from = 0; q_from < n; ++q_from) {
          problem.push_back({-var_graph_being(graph_id, time_moment, mem_state), -var_q_from(mem_state, q_from), var_graph_qfrom(graph_id, time_moment, q_from)});
        }
        for (int q_to = 0; q_to < n; ++q_to) {
          problem.push_back({-var_graph_being(graph_id, time_moment, mem_state), -var_q_to(mem_state, q_to), var_graph_qto(graph_id, time_moment, q_to)});
        }
      }
      // our question answer should be according to the graph
      for (int q_from = 0; q_from < n; ++q_from) {
        for (int q_to = 0; q_to < n; ++q_to) {
          problem.push_back({-var_graph_qfrom(graph_id, time_moment, q_from), -var_graph_qto(graph_id, time_moment, q_to), var_graph_qans(graph_id, time_moment) * (test_graphs[graph_id][q_from][q_to] ? +1 : -1)});
        }
      }
      // our next state should be as documented
      for (int mem_from = 0; mem_from < M; ++mem_from) {
        for (int mem_to = 0; mem_to < M; ++mem_to) {
          problem.push_back({-var_graph_being(graph_id, time_moment, mem_from), var_graph_qans(graph_id, time_moment), -var_transition_if_0(mem_from, mem_to), var_graph_being(graph_id, time_moment + 1, mem_to)});
          problem.push_back({-var_graph_being(graph_id, time_moment, mem_from), -var_graph_qans(graph_id, time_moment), -var_transition_if_1(mem_from, mem_to), var_graph_being(graph_id, time_moment + 1, mem_to)});
        }
      }
    }
    // except for the last time moment, here we should be in the correct memory state
    for (int mem_state = 0; mem_state < M - 2; ++mem_state) {
      problem.push_back({-var_graph_being(graph_id, max_solving_time, mem_state)});
    }
    if (test_graphs_answers[graph_id]) {
      problem.push_back({-var_graph_being(graph_id, max_solving_time, M - 2)});
      problem.push_back({var_graph_being(graph_id, max_solving_time, M - 1)});
    } else {
      problem.push_back({var_graph_being(graph_id, max_solving_time, M - 2)});
      problem.push_back({-var_graph_being(graph_id, max_solving_time, M - 1)});
    }
  }

  std::string filename("p.cnf");
  std::cout << "writing to file: " << filename << std::endl;
  std::ofstream fout(filename, std::ios::out);
  fout << "p cnf " << cnt_vars_algo + cnt_vars_graph_runs << " " << problem.size() << "\n";
  for (const std::vector<int>& clause : problem) {
    for (int literal : clause) {
      fout << literal << " ";
    }
    fout << "0\n";
  }
  std::cout << "done!\n";
}

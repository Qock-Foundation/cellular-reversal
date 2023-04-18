#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <queue>
#include <random>
#include <vector>

const int n = 5;  // cnt vertices, source=0, target=n-1
const int N = 100;  // cnt graphs (tests)
const int M = 2;  // cnt memory states (~2^(log n)^2)
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

const int cnt_set = 100'000'000;
int value[cnt_set];

int main(int argc, char **argv) {
  assert(argc == 2);
  std::fill(value, value + cnt_set, -1);
  std::string filename(argv[1]);
  std::ifstream fin(filename);
  int var = 1;
  while (var != 0) {
    fin >> var;
    if (var > 0) {
      value[var] = 1;
    } else if (var < 0) {
      value[-var] = 0;
    }
  }
  std::cout << "\n";
  std::cout << "SATISFIABLE with n = " << n << ", N = " << N << ", M = " << M << "\n";
  std::cout << "\n";
  std::cout << "possible question types:\n";
  std::cout << "low-level:\n";
  std::cout << "\tvar_q_from(mem_state, v)\n";
  std::cout << "\tvar_q_to(mem_state, v)\n";
  std::cout << "\tvar_transition_if_0(mem_from, mem_to)\n";
  std::cout << "\tvar_transition_if_1(mem_from, mem_to)\n";
  std::cout << "\tvar_graph_being(graph_id, time_moment, mem_state)\n";
  std::cout << "\tvar_graph_qfrom(graph_id, time_moment, v)\n";
  std::cout << "\tvar_graph_qto(graph_id, time_moment, v)\n";
  std::cout << "\tvar_graph_qans(graph_id, time_moment)\n";
  std::cout << "high-level:\n";
  std::cout << "\tquestioned_edge(mem_state)\n";
  std::cout << "\tproposed_transitions(mem_from)\n";
  std::cout << "\tgraph_run_info(graph_id)\n";
  std::cout << "\n";
  std::string var_type;
  while (std::cout << "VAR_TYPE: " && std::cin >> var_type) {
    if (var_type == "var_q_from" || var_type == "var_q_to") {
      int mem_state, v;
      std::cout << "mem_state (0.." << M - 1 << "): ";
      std::cin >> mem_state;
      std::cout << "v (0.." << n - 1 << "): ";
      std::cin >> v;
      std::cout << "\tvalue: " << value[(var_type == "var_q_from" ? var_q_from : var_q_to)(mem_state, v)] << std::endl;
    } else if (var_type == "var_transition_if_0" || var_type == "var_transition_if_1") {
      int mem_from, mem_to;
      std::cout << "mem_from (0.." << M - 1 << "): ";
      std::cin >> mem_from;
      std::cout << "mem_to: (0.." << M - 1 << "): ";
      std::cin >> mem_to;
      std::cout << "\tvalue: " << value[(var_type == "var_transition_if_0" ? var_transition_if_0 : var_transition_if_1)(mem_from, mem_to)] << std::endl;
    } else if (var_type == "var_graph_being") {
      int graph_id, time_moment, mem_state;
      std::cout << "graph_id (0.." << N - 1 << "): ";
      std::cin >> graph_id;
      std::cout << "time_moment (0.." << max_solving_time << "): ";
      std::cin >> time_moment;
      std::cout << "mem_state (0.." << M - 1 << "): ";
      std::cin >> mem_state;
      std::cout << "\tvalue: " << value[var_graph_being(graph_id, time_moment, mem_state)] << std::endl;
    } else if (var_type == "var_graph_qfrom" || var_type == "var_graph_qto") {
      int graph_id, time_moment, v;
      std::cout << "graph_id (0.." << N - 1 << "): ";
      std::cin >> graph_id;
      std::cout << "time_moment (0.." << max_solving_time << "): ";
      std::cin >> time_moment;
      std::cout << "v (0.." << n - 1 << "): ";
      std::cin >> v;
      std::cout << "\tvalue: " << value[(var_type == "var_graph_qfrom" ? var_graph_qfrom : var_graph_qto)(graph_id, time_moment, v)] << std::endl;
    } else if (var_type == "var_graph_qans") {
      int graph_id, time_moment;
      std::cout << "graph_id (0.." << N - 1 << "): ";
      std::cin >> graph_id;
      std::cout << "time_moment (0.." << max_solving_time << "): ";
      std::cin >> time_moment;
      std::cout << "\tvalue: " << value[var_graph_qans(graph_id, time_moment)] << std::endl;
    } else if (var_type == "questioned_edge") {
      int mem_state;
      std::cout << "mem_state (0.." << M - 1 << "): ";
      std::cin >> mem_state;
      std::vector<int> q_from, q_to;
      for (int v = 0; v < n; ++v) {
        int var = var_q_from(mem_state, v), val = value[var];
        if (val == -1) {
          std::cout << "\twarning: variable " << var << " [aka var_q_from(mem_state=" << mem_state << ", v=" << v << ")] not set!\n";
        } else if (val == 1) {
          q_from.push_back(v);
        }
        var = var_q_to(mem_state, v), val = value[var];
        if (val == -1) {
          std::cout << "\twarning: variable " << var << " [aka var_q_to(mem_state=" << mem_state << ", v=" << v << ")] not set!\n";
        } else if (val == 1) {
          q_to.push_back(v);
        }
      }
      std::cout << "\tedge: from ";
      std::copy(q_from.begin(), q_from.end(), std::ostream_iterator<int>(std::cout, " "));
      std::cout << "to ";
      std::copy(q_to.begin(), q_to.end(), std::ostream_iterator<int>(std::cout, " "));
      std::cout << std::endl;
    } else if (var_type == "proposed_transitions") {
      int mem_from;
      std::cout << "mem_from: ";
      std::cin >> mem_from;
      std::vector<int> if_0, if_1;
      for (int mem_to = 0; mem_to < M; ++mem_to) {
        int var = var_transition_if_0(mem_from, mem_to), val = value[var];
        if (val == -1) {
          std::cout << "\twarning: variable " << var << " [aka var_transition_if_0(mem_from=" << mem_from << ", mem_to=" << mem_to << ")] not set!\n";
        } else if (val == 1) {
          if_0.push_back(mem_to);
        }
        var = var_transition_if_1(mem_from, mem_to), val = value[var];
        if (val == -1) {
          std::cout << "\twarning: variable " << var << " [aka var_transition_if_1(mem_from=" << mem_from << ", mem_to=" << mem_to << ")] not set!\n";
        } else if (val == 1) {
          if_1.push_back(mem_to);
        }
      }
      std::cout << "\tif the answer is false, proceeds to state ";
      std::copy(if_0.begin(), if_0.end(), std::ostream_iterator<int>(std::cout, " "));
      std::cout << "\n";
      std::cout << "\tif the answer is true, proceeds to state ";
      std::copy(if_1.begin(), if_1.end(), std::ostream_iterator<int>(std::cout, " "));
      std::cout << std::endl;
    } else if (var_type == "graph_run_info") {
      int graph_id;
      std::cout << "graph_id: ";
      std::cin >> graph_id;
      for (int time_moment = 0; time_moment <= max_solving_time; ++time_moment) {
        std::cout << "\tat time moment " << time_moment << ":\n";
        std::vector<int> being, qfrom, qto;
        for (int mem_state = 0; mem_state < M; ++mem_state) {
          int var = var_graph_being(graph_id, time_moment, mem_state), val = value[var];
          if (val == -1) {
            std::cout << "\twarning: variable " << var << " [aka var_graph_being(graph_id=" << graph_id << ", time_moment=" << time_moment << ", mem_state=" << mem_state << ")] not set!\n";
          } else if (val == 1) {
            being.push_back(mem_state);
          }
        }
        std::cout << "\t\twe are in memory state ";
        std::copy(being.begin(), being.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << "\n";
        for (int v = 0; v < n; ++v) {
          int var = var_graph_qfrom(graph_id, time_moment, v), val = value[var];
          if (val == -1) {
            std::cout << "\twarning: variable " << var << " [aka var_graph_qfrom(graph_id=" << graph_id << ", time_moment=" << time_moment << ", v=" << v << ")] not set!\n";
          } else if (val == 1) {
            qfrom.push_back(v);
          }
          var = var_graph_qto(graph_id, time_moment, v), val = value[var];
          if (val == -1) {
            std::cout << "\twarning: variable " << var << " [aka var_graph_qto(graph_id=" << graph_id << ", time_moment=" << time_moment << ", v=" << v << ")] not set!\n";
          } else if (val == 1) {
            qto.push_back(v);
          }
        }
        std::cout << "\t\t..questioning about whether there is an edge from ";
        std::copy(qfrom.begin(), qfrom.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << "to ";
        std::copy(qto.begin(), qto.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << "\n";
        std::cout << "\t\t..and receiving answer " << value[var_graph_qans(graph_id, time_moment)] << std::endl;
      }
    } else {
      std::cout << "\tunknown request type" << std::endl;
    }
  }
}

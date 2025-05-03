#include "all/kholin_k_multidimensional_integrals_rectangle/include/ops_all.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

void kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::DivideWork(std::vector<double>& local_l_limits,
                                                                                std::vector<double>& local_u_limits,
                                                                                int rank, int size, size_t dim) {
  for (size_t i = 0; i < dim; ++i) {
    double range = upper_limits_[i] - lower_limits_[i];
    local_l_limits[i] = lower_limits_[i] + rank * (range / size);
    local_u_limits[i] = lower_limits_[i] + (rank + 1) * (range / size);
  }
}

double kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::Integrate(
    const Function& f, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
    const std::vector<double>& h, std::vector<double> f_values, int curr_index_dim, size_t dim, double n) {
  if (curr_index_dim == static_cast<int>(dim)) {
    return f(f_values);
  }

  double sum = 0.0;
  const double h_curr = h[curr_index_dim];
  const double l_limit_curr = l_limits[curr_index_dim];

#pragma omp parallel for reduction(+ : sum) schedule(guided)
  for (int i = 0; i < static_cast<int>(n); ++i) {
    f_values[curr_index_dim] = l_limit_curr + (static_cast<double>(i) + 0.5) * h_curr;
    sum += Integrate(f, l_limits, u_limits, h, f_values, curr_index_dim + 1, dim, n);
  }
  return sum * h[curr_index_dim];
}

double kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::IntegrateWithRectangleMethod(
    const Function& f, std::vector<double>& f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim, double n, std::vector<double> h) {
  for (size_t i = 0; i < dim; ++i) {
    h[i] = (u_limits[i] - l_limits[i]) / n;
  }
  return Integrate(f, l_limits, u_limits, h, f_values, 0, dim, n);
}

double kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::RunMultistepSchemeMethodRectangle(
    const Function& f, std::vector<double> f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim, double n) {
  int rank = world.rank();
  int size = world.size();

  std::vector<double> local_l_limits = l_limits;
  std::vector<double> local_u_limits = u_limits;
  DivideWork(local_l_limits, local_u_limits, rank, size, dim);

  std::vector<double> h(dim);
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(dim); ++i) {
    h[i] = (local_u_limits[i] - local_l_limits[i]) / n;
  }

  double local_result = IntegrateWithRectangleMethod(f, f_values, local_l_limits, local_u_limits, dim, n, h);

  double global_result = 0.0;
  MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return (rank == 0) ? global_result : 0.0;
}

bool kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::PreProcessingImpl() {
  int rank = world.rank();
  int size = world.size();
  if (rank == 0) {
    // Init value for input and output
    sz_values_ = task_data->inputs_count[0];
    sz_lower_limits_ = task_data->inputs_count[1];
    sz_upper_limits_ = task_data->inputs_count[2];

    auto* ptr_dim = reinterpret_cast<size_t*>(task_data->inputs[0]);
    dim_ = *ptr_dim;

    auto* ptr_f_values = reinterpret_cast<double*>(task_data->inputs[1]);
    f_values_.assign(ptr_f_values, ptr_f_values + sz_values_);

    auto* ptr_f = reinterpret_cast<std::function<double(const std::vector<double>&)>*>(task_data->inputs[2]);
    f_ = *ptr_f;

    auto* ptr_lower_limits = reinterpret_cast<double*>(task_data->inputs[3]);
    lower_limits_.assign(ptr_lower_limits, ptr_lower_limits + sz_lower_limits_);

    auto* ptr_upper_limits = reinterpret_cast<double*>(task_data->inputs[4]);
    upper_limits_.assign(ptr_upper_limits, ptr_upper_limits + sz_upper_limits_);

    auto* ptr_start_n = reinterpret_cast<double*>(task_data->inputs[5]);
    start_n_ = *ptr_start_n;
  }

  MPI_Bcast(&dim_, 1, mpi_size_t, 0, MPI_COMM_WORLD);
  MPI_Bcast(&start_n_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sz_values_, 1, mpi_size_t, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sz_lower_limits_, 1, mpi_size_t, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sz_upper_limits_, 1, mpi_size_t, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    f_values_.resize(sz_values_);
    lower_limits_.resize(sz_lower_limits_);
    upper_limits_.resize(sz_upper_limits_);
  }

  MPI_Bcast(f_values_.data(), sz_values_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(lower_limits_.data(), sz_lower_limits_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(upper_limits_.data(), sz_upper_limits_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    omp_set_num_threads(omp_get_max_threads() / size);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  result_ = 0.0;
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::ValidationImpl() {
  bool valid = true;
  if (world.rank() == 0) {
    valid = task_data->inputs_count[1] > 0U && task_data->inputs_count[2] > 0U;
  }
  boost::mpi::broadcast(world, valid, 0);
  return valid;
}

bool kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::RunImpl() {
  result_ = RunMultistepSchemeMethodRectangle(f_, f_values_, lower_limits_, upper_limits_, dim_, start_n_);
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_all::TestTaskALL::PostProcessingImpl() {
  if (world.rank() == 0) {
    reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  }
  MPI_Type_free(&mpi_size_t);
  return true;
}
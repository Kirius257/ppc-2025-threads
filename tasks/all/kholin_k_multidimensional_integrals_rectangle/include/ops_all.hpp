#pragma once

#include <mpi.h>

#include <boost/mpi/collectives.hpp>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kholin_k_multidimensional_integrals_rectangle_all {
using Function = std::function<double(const std::vector<double>&)>;
class TestTaskALL : public ppc::core::Task {
  MPI_Datatype GetMpiType();

 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data, std::function<double(const std::vector<double>&)> f)
      : Task(std::move(task_data)), f_(std::move(f)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  ~TestTaskALL() override;

 private:
  std::vector<double> f_values_;
  Function f_;
  std::vector<double> lower_limits_;
  std::vector<double> upper_limits_;
  double start_n_;

  std::vector<double> local_l_limits_;
  std::vector<double> local_u_limits_;

  size_t dim_;
  size_t sz_values_;
  size_t sz_lower_limits_;
  size_t sz_upper_limits_;
  double I_2n_;

  double Integrate(const Function& f, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                   const std::vector<double>& h, std::vector<double>& f_values, size_t curr_index_dim, size_t dim,
                   double n);
  double IntegrateWithRectangleMethod(const Function& f, std::vector<double>& f_values,
                                      const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                                      size_t dim, double n);
  double RunMultistepSchemeMethodRectangle(const Function& f, std::vector<double>& f_values,
                                           const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                                           size_t dim, double n);
  MPI_Datatype mpi_size_t_;
};

}  // namespace kholin_k_multidimensional_integrals_rectangle_all
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "omp/kholin_k_multidimensional_integrals_rectangle/include/ops_omp.hpp"

TEST(kholin_k_multidimensional_integrals_rectangle_omp, test_validation) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double n = 10.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, test_pre_processing) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double n = 10.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, test_run) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double n = 10.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  ASSERT_EQ(test_task_omp.Run(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, test_post_processing) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double n = 10.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  ASSERT_EQ(test_task_omp.Run(), true);
  ASSERT_EQ(test_task_omp.PostProcessing(), true);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, single_integral_one_var) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double n = 1002.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  ASSERT_EQ(test_task_omp.Run(), true);
  ASSERT_EQ(test_task_omp.PostProcessing(), true);

  double ref_i = 0.46;
  ASSERT_NEAR(ref_i, out_i[0], 1e-2);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, single_integral_two_var) {
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0, 3.0};
  auto f = [](const std::vector<double> &f_values) { return std::exp(-f_values[0] + f_values[1]); };
  std::vector<double> in_lower_limits{-1};
  std::vector<double> in_upper_limits{5};
  double n = 1012.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  ASSERT_EQ(test_task_omp.Run(), true);
  ASSERT_EQ(test_task_omp.PostProcessing(), true);

  double ref_i = 54;
  ASSERT_EQ(ref_i, std::round((out_i[0])));
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, double_integral_two_var) {
  // Create data
  size_t dim = 2;
  std::vector<double> values{0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) {
    return (f_values[0] * f_values[0]) + (f_values[1] * f_values[1]);
  };
  std::vector<double> in_lower_limits{-10, 3};
  std::vector<double> in_upper_limits{10, 4};
  double n = 700.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  ASSERT_EQ(test_task_omp.Run(), true);
  ASSERT_EQ(test_task_omp.PostProcessing(), true);

  double ref_i = 913;
  ASSERT_EQ(ref_i, std::round(out_i[0]));
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, double_integral_one_var) {
  // Create data
  size_t dim = 2;
  std::vector<double> values{-17.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return 289 + (f_values[1] * f_values[1]); };
  std::vector<double> in_lower_limits{-10, 3};
  std::vector<double> in_upper_limits{10, 4};
  double n = 405.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  ASSERT_EQ(test_task_omp.Run(), true);
  ASSERT_EQ(test_task_omp.PostProcessing(), true);

  double ref_i = 6027;
  ASSERT_EQ(ref_i, std::round(out_i[0]));
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, triple_integral_three_var) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + f_values[1] + f_values[2]; };
  std::vector<double> in_lower_limits{-4, 6, 7};
  std::vector<double> in_upper_limits{4, 13, 8};
  double n = 176.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  ASSERT_EQ(test_task_omp.Run(), true);
  ASSERT_EQ(test_task_omp.PostProcessing(), true);

  double ref_i = 952;
  ASSERT_EQ(ref_i, std::round(out_i[0]));
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, triple_integral_two_var) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 5.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + 5.0 + f_values[2]; };
  std::vector<double> in_lower_limits{0, 5, -3};
  std::vector<double> in_upper_limits{12, 20, 2};
  double n = 150.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  ASSERT_EQ(test_task_omp.Run(), true);
  ASSERT_EQ(test_task_omp.PostProcessing(), true);

  double ref_i = 9450;
  ASSERT_EQ(ref_i, std::round(out_i[0]));
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, triple_integral_one_var) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 5.0, -10.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + 5.0 + (-10.0); };
  std::vector<double> in_lower_limits{0, 5, -3};
  std::vector<double> in_upper_limits{12, 20, 2};
  double n = 175.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  ASSERT_EQ(test_task_omp.Run(), true);
  ASSERT_EQ(test_task_omp.PostProcessing(), true);

  double ref_i = 900;
  ASSERT_EQ(ref_i, std::round(out_i[0]));
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, triple_integral_three_var_high_acc) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) {
    return (-2.4925829 * f_values[0] + sin(f_values[1]) + f_values[2]);
  };
  std::vector<double> in_lower_limits{2, -2, -5};
  std::vector<double> in_upper_limits{3, 0, 1};
  double n = 200.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  ASSERT_EQ(test_task_omp.Run(), true);
  ASSERT_EQ(test_task_omp.PostProcessing(), true);

  double ref_i = -107.2743680;
  ASSERT_NEAR(ref_i, out_i[0], 10 * 1e-3);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, double_integral_two_var_high_acc) {
  // Create data
  size_t dim = 2;
  std::vector<double> values{0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) {
    return ((-2.4925829 + f_values[0]) * f_values[0]) + (f_values[1] * f_values[1]);
  };
  std::vector<double> in_lower_limits{3, 0};
  std::vector<double> in_upper_limits{5, 1};
  double n = 1500.0;
  std::vector<double> out_i(1, 0.0);

  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_omp->inputs_count.emplace_back(values.size());
  task_data_omp->inputs_count.emplace_back(in_lower_limits.size());
  task_data_omp->inputs_count.emplace_back(in_upper_limits.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_i.data()));
  task_data_omp->outputs_count.emplace_back(out_i.size());

  // Create Task
  kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  ASSERT_EQ(test_task_omp.PreProcessing(), true);
  ASSERT_EQ(test_task_omp.Run(), true);
  ASSERT_EQ(test_task_omp.PostProcessing(), true);

  double ref_i = 28.34816754;
  ASSERT_NEAR(ref_i, out_i[0], 10 * 1e-3);
  delete f_object;
}
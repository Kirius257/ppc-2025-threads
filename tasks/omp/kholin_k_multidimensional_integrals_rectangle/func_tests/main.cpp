#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
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
  auto f = [](const std::vector<double> &f_values) { return (f_values[0] + (2.0 * f_values[1])); };
  std::vector<double> in_lower_limits{0, 0};
  std::vector<double> in_upper_limits{4, 2};
  double n = 350.0;
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

  double ref_i = 32.0;
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
  std::vector<double> in_lower_limits{-2, -2, 0};
  std::vector<double> in_upper_limits{4, 6, 3};
  double n = 100.0;
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

  double ref_i = 648;
  std::cout << "       " << out_i[0] << '\n';
  ASSERT_EQ(ref_i, std::round(out_i[0]));
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, triple_integral_two_var) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 5.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return (f_values[0] * f_values[0]) + f_values[1]; };
  std::vector<double> in_lower_limits{-2, 1, 0};
  std::vector<double> in_upper_limits{2, 4, 3};
  double n = 180.0;
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

  double ref_i = 138;
  std::cout << "       " << out_i[0] << '\n';
  ASSERT_EQ(ref_i, std::round(out_i[0]));
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, triple_integral_one_var) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 5.0, -10.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + 5.0 + (-10.0); };
  std::vector<double> in_lower_limits{0, 0, 0};
  std::vector<double> in_upper_limits{2, 1, 3};
  double n = 100.0;
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

  double ref_i = -24;
  std::cout << "       " << out_i[0] << '\n';
  ASSERT_EQ(ref_i, std::round(out_i[0]));
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, triple_integral_three_var_high_acc) {
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return ((1.0 / 3.0 * f_values[0] * f_values[1] * f_values[2])); };
  std::vector<double> in_lower_limits{0, 0, 0};
  std::vector<double> in_upper_limits{1, 1, 1};
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

  double ref_i = 0.04166666667;
  std::cout << "       " << out_i[0] << '\n';
  ASSERT_NEAR(ref_i, out_i[0], 1e-3);
  delete f_object;
}

TEST(kholin_k_multidimensional_integrals_rectangle_omp, double_integral_two_var_high_acc) {
  // Create data
  size_t dim = 2;
  std::vector<double> values{0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return (1.0 / 9.0 * f_values[0] * f_values[1]); };
  std::vector<double> in_lower_limits{0, 0};
  std::vector<double> in_upper_limits{1, 1};
  double n = 300.0;
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

  double ref_i = 0.02777777778;
  std::cout << "       " << out_i[0] << '\n';
  ASSERT_NEAR(ref_i, out_i[0], 1e-3);
  delete f_object;
}
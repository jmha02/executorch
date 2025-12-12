/*
 * Minimal portable CPU runner for stateless ZO LoRA training.
 * Assumes model signature:
 * inputs: [0] ids(int64), [1] labels(int64), [2] lr(float scalar),
 *         [3] lora_B(float tensor), [4] noise(float tensor), [5] eps(float scalar)
 * outputs: [0] loss(float scalar), [1] grad(float scalar)
 */

#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <fstream>
#include <string>
#include <vector>

using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;

static void fill_noise(std::vector<float>& buf) {
  static thread_local std::mt19937 rng{std::random_device{}()};
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (auto& v : buf) {
    v = dist(rng);
  }
}

static void fill_small_uniform(std::vector<float>& buf, float scale = 1e-3f) {
  static thread_local std::mt19937 rng{std::random_device{}()};
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (auto& v : buf) {
    v = dist(rng);
  }
}

static std::vector<int64_t> load_int64_bin(const std::string& path) {
  std::vector<int64_t> data;
  if (path.empty()) {
    return data;
  }
  std::ifstream fin(path, std::ios::binary);
  if (!fin.is_open()) {
    std::fprintf(stderr, "Failed to open %s\n", path.c_str());
    return data;
  }
  fin.seekg(0, fin.end);
  size_t bytes = fin.tellg();
  fin.seekg(0, fin.beg);
  size_t elems = bytes / sizeof(int64_t);
  data.resize(elems);
  fin.read(reinterpret_cast<char*>(data.data()), bytes);
  return data;
}

static bool load_lora(const std::string& path, std::vector<float>& buf) {
  if (path.empty()) {
    return false;
  }
  std::ifstream fin(path, std::ios::binary);
  if (!fin.is_open()) {
    std::fprintf(stderr, "Failed to open lora file %s\n", path.c_str());
    return false;
  }
  fin.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
  if (!fin.good()) {
    std::fprintf(stderr, "Failed to read lora file %s (size mismatch?)\n", path.c_str());
    return false;
  }
  return true;
}

static bool save_lora(const std::string& path, const std::vector<float>& buf) {
  if (path.empty()) {
    return false;
  }
  std::ofstream fout(path, std::ios::binary);
  if (!fout.is_open()) {
    std::fprintf(stderr, "Failed to open lora save file %s\n", path.c_str());
    return false;
  }
  fout.write(reinterpret_cast<const char*>(buf.data()), buf.size() * sizeof(float));
  if (!fout.good()) {
    std::fprintf(stderr, "Failed to write lora save file %s\n", path.c_str());
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <model.pte> [--epochs=N] [--train_data_inputs=path] [--train_data_labels=path]\n", argv[0]);
    return 1;
  }
  std::string model_path = argv[1];
  int epochs = 1;
  std::string data_inputs_path, data_labels_path;
  std::string load_lora_path, save_lora_path;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    auto eat_prefix = [&](const std::string& pfx) {
      if (arg.rfind(pfx, 0) == 0) {
        return arg.substr(pfx.size());
      }
      return std::string{};
    };
    if (arg.rfind("--epochs=", 0) == 0) {
      epochs = std::atoi(arg.substr(9).c_str());
    } else if (arg.rfind("--train_data_inputs=", 0) == 0) {
      data_inputs_path = eat_prefix("--train_data_inputs=");
    } else if (arg.rfind("--train_data_labels=", 0) == 0) {
      data_labels_path = eat_prefix("--train_data_labels=");
    } else if (arg.rfind("--load_lora=", 0) == 0) {
      load_lora_path = eat_prefix("--load_lora=");
    } else if (arg.rfind("--save_lora=", 0) == 0) {
      save_lora_path = eat_prefix("--save_lora=");
    }
  }

  executorch::runtime::runtime_init();

  Module module(model_path, Module::LoadMode::File);
  Error st = module.load_forward();
  if (st != Error::Ok) {
    std::fprintf(stderr, "Failed to load forward method: 0x%x\n", (uint32_t)st);
    return 1;
  }
  Result<Method*> meth_res = module.method("forward");
  if (!meth_res.ok()) {
    std::fprintf(stderr, "Failed to get method: 0x%x\n", (uint32_t)meth_res.error());
    return 1;
  }
  Method* method = *meth_res;
  MethodMeta meta = method->method_meta();
  std::fprintf(stderr, "Planned buffers: %zu, planned buffer[0]=%zu bytes\n",
               meta.num_memory_planned_buffers(),
               meta.memory_planned_buffer_size(0).get());

  // Allocate input buffers according to meta
  const int num_inputs = method->inputs_size();
  if (num_inputs != 6) {
    std::fprintf(stderr, "Unexpected input count %d\n", num_inputs);
    return 1;
  }
  std::vector<std::vector<uint8_t>> input_storage(num_inputs);
  std::vector<TensorImpl> input_impls;
  input_impls.reserve(num_inputs);

  auto make_tensor = [&](int idx, void* data_ptr) {
    Result<TensorInfo> tinfo = meta.input_tensor_meta(idx);
    TensorImpl impl(
        tinfo->scalar_type(),
        /*dim=*/tinfo->sizes().size(),
        const_cast<TensorImpl::SizesType*>(tinfo->sizes().data()),
        data_ptr,
        const_cast<TensorImpl::DimOrderType*>(tinfo->dim_order().data()));
    return impl;
  };

  // Prepare host buffers
  for (int i = 0; i < num_inputs; ++i) {
    auto tinfo = meta.input_tensor_meta(i);
    input_storage[i].resize(tinfo->nbytes());
  }

  // Shapes for lora_B/noise
  auto lora_info = meta.input_tensor_meta(3);
  size_t lora_elems = 1;
  for (auto s : lora_info->sizes()) {
    lora_elems *= s;
  }
  std::vector<float> lora_B(lora_elems, 0.0f);
  std::vector<float> noise(lora_elems, 0.0f);
  if (!load_lora(load_lora_path, lora_B)) {
    fill_small_uniform(lora_B, 1e-3f); // break symmetry
  } else {
    std::fprintf(stderr, "Loaded lora_B from %s\n", load_lora_path.c_str());
  }

  // Dummy ids/labels (zeros)
  for (int i = 0; i < num_inputs; ++i) {
    input_impls.push_back(make_tensor(i, input_storage[i].data()));
  }

  // Scalars lr/eps
  float lr = 1e-2f;
  float eps = 2e-2f; // larger epsilon for clearer signal

  // Load datasets if provided
  auto input_data = load_int64_bin(data_inputs_path);
  auto label_data = load_int64_bin(data_labels_path);
  int64_t batch = meta.input_tensor_meta(0)->sizes()[0];
  int64_t seq_len = meta.input_tensor_meta(0)->sizes()[1];
  size_t sample_elems = static_cast<size_t>(batch * seq_len);
  size_t num_samples = 0;
  if (!input_data.empty() && input_data.size() >= sample_elems) {
    num_samples = input_data.size() / sample_elems;
  }

  for (int epoch = 0; epoch < epochs; ++epoch) {
    size_t steps = std::max<size_t>(1, num_samples);
    for (size_t step = 0; step < steps; ++step) {
      auto t_begin = std::chrono::steady_clock::now();
      // ids/labels: from dataset if available
      if (!input_data.empty()) {
        size_t off = step % num_samples;
        std::memcpy(
            input_storage[0].data(),
            input_data.data() + off * sample_elems,
            sample_elems * sizeof(int64_t));
      } else {
        std::memset(input_storage[0].data(), 0, sample_elems * sizeof(int64_t));
      }
      if (!label_data.empty()) {
        size_t off = step % num_samples;
        std::memcpy(
            input_storage[1].data(),
            label_data.data() + off * sample_elems,
            sample_elems * sizeof(int64_t));
      } else {
        std::memset(input_storage[1].data(), 0, sample_elems * sizeof(int64_t));
      }

      std::memcpy(input_storage[2].data(), &lr, sizeof(float));
      std::memcpy(input_storage[5].data(), &eps, sizeof(float));

      std::memcpy(input_storage[3].data(), lora_B.data(), lora_B.size() * sizeof(float));
      fill_noise(noise);
      std::memcpy(input_storage[4].data(), noise.data(), noise.size() * sizeof(float));

      for (int i = 0; i < num_inputs; ++i) {
        Tensor t(&input_impls[i]);
        st = method->set_input(t, i);
        if (st != Error::Ok) {
          std::fprintf(stderr, "set_input %d failed: 0x%x\n", i, (uint32_t)st);
          return 1;
        }
      }

      st = method->execute();
      if (st != Error::Ok) {
        std::fprintf(stderr, "execute failed: 0x%x\n", (uint32_t)st);
        return 1;
      }

      std::vector<EValue> outputs(method->outputs_size());
      st = method->get_outputs(outputs.data(), outputs.size());
      if (st != Error::Ok) {
        std::fprintf(stderr, "get_outputs failed: 0x%x\n", (uint32_t)st);
        return 1;
      }
      float loss = outputs[0].toTensor().const_data_ptr<float>()[0];
      float grad = outputs[1].toTensor().const_data_ptr<float>()[0];

      float grad_abs_sum = 0.0f;
      for (float v : noise) {
        grad_abs_sum += std::fabs(grad * v);
      }

      for (size_t i = 0; i < lora_B.size(); ++i) {
        lora_B[i] -= lr * grad * noise[i];
      }
      auto t_end = std::chrono::steady_clock::now();
      double step_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count() / 1000.0;
      double tokens = static_cast<double>(batch * seq_len);
      double tok_per_s = step_ms > 0.0 ? tokens / (step_ms / 1000.0) : 0.0;
      std::printf("[Epoch %d Step %zu/%zu] loss=%.6f grad=%.6f grad_noise_abs_sum=%.6f step_ms=%.3f tok/s=%.2f\n",
                  epoch, step + 1, steps, loss, grad, grad_abs_sum, step_ms, tok_per_s);
    }
  }

  if (!save_lora_path.empty()) {
    if (save_lora(save_lora_path, lora_B)) {
      std::fprintf(stderr, "Saved lora_B to %s\n", save_lora_path.c_str());
    }
  }

  return 0;
}

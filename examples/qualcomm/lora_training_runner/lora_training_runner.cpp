/*
 * LoRA On-Device Training Runner
 * With proper portable ops registration
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/core/memory_allocator.h>

#include <cmath>
#include <iostream>
#include <memory>

using executorch::aten::ScalarType;
using executorch::etdump::ETDumpGen;
using executorch::extension::FileDataLoader;
using executorch::extension::from_blob;
using executorch::runtime::Error;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

int main(int argc, char** argv) {
    executorch::runtime::runtime_init();

    // Note: Portable operators are automatically registered via static initialization
    // when linking with full_portable_ops_lib
    ET_LOG(Info, "Runtime initialized with portable operators");

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.pte> [num_epochs] [num_steps]" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];
    int num_epochs = (argc > 2) ? std::atoi(argv[2]) : 3;
    int num_steps = (argc > 3) ? std::atoi(argv[3]) : 10;

    ET_LOG(Info, "========================================");
    ET_LOG(Info, "LoRA On-Device Training - Real Implementation");
    ET_LOG(Info, "========================================");
    ET_LOG(Info, "Model: %s", model_path);
    ET_LOG(Info, "Epochs: %d", num_epochs);
    ET_LOG(Info, "Steps per epoch: %d", num_steps);
    ET_LOG(Info, "========================================");

    // Load model
    Result<FileDataLoader> loader = FileDataLoader::from(model_path);
    if (!loader.ok()) {
        ET_LOG(Error, "Failed to load model file");
        return 1;
    }

    Result<Program> program = Program::load(&loader.get());
    if (!program.ok()) {
        ET_LOG(Error, "Failed to parse program");
        return 1;
    }
    ET_LOG(Info, "Model loaded successfully");

    // Get method
    const char* method_name = nullptr;
    {
        const auto method_name_result = program->get_method_name(0);
        if (!method_name_result.ok()) {
            ET_LOG(Error, "Program has no methods");
            return 1;
        }
        method_name = *method_name_result;
    }
    ET_LOG(Info, "Using method: %s", method_name);

    // Get method metadata
    Result<MethodMeta> method_meta = program->method_meta(method_name);
    if (!method_meta.ok()) {
        ET_LOG(Error, "Failed to get method metadata");
        return 1;
    }

    // Setup memory
    MemoryAllocator method_allocator{
        MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

    std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
    std::vector<Span<uint8_t>> planned_spans;
    size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();

    for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
        size_t buffer_size =
            static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
        ET_LOG(Info, "Planned buffer %zu: %zu bytes", id, buffer_size);
        planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
        planned_spans.push_back({planned_buffers.back().get(), buffer_size});
    }

    HierarchicalAllocator planned_memory(
        {planned_spans.data(), planned_spans.size()});
    MemoryManager memory_manager(&method_allocator, &planned_memory);
    // Load method
    ETDumpGen etdump_gen;
    auto method_res = program->load_method(method_name, &memory_manager, &etdump_gen);
    if (!method_res.ok()) {
        ET_LOG(Error, "Failed to load method");
        return 1;
    }
    ET_LOG(Info, "Method loaded successfully");

    // Get method reference
    Method& method = method_res.get();

    // Prepare input tensors
    ET_LOG(Info, "Preparing input tensors...");

    // Get input metadata
    size_t num_inputs = method_meta->num_inputs();
    ET_LOG(Info, "Method expects %zu inputs", num_inputs);

    // 1. Load Module
    auto module = executorch::extension::Module::load(model_path);
    if (!module.ok()) {
        ET_LOG(Error, "Failed to load module: %s", model_path.c_str());
        return 1;
    }
    ET_LOG(Info, "Module loaded successfully");

    // 2. Prepare Inputs
    // Input 0: input_ids [2, 32] (int64)
    const int32_t batch_size = 2;
    const int32_t seq_len = 32;
    const int32_t vocab_size = 50272; // OPT-125M vocab size
    
    int64_t input_ids_data[batch_size * seq_len];
    // Initialize with dummy data (e.g., token ID 100)
    for (int i = 0; i < batch_size * seq_len; i++) input_ids_data[i] = 100 + (i % 100);
    
    std::vector<executorch::aten::SizesType> input_sizes = {batch_size, seq_len};
    auto input_tensor_impl = executorch::runtime::TensorImpl(
        executorch::aten::ScalarType::Long,
        std::size(input_sizes),
        input_sizes.data(),
        input_ids_data,
        std::vector<executorch::aten::DimOrderType>().data()
    );
    auto input_tensor = executorch::runtime::Tensor(&input_tensor_impl);

    // Input 1: learning_rate [1] (float32)
    float lr_data[1] = {0.01f};
    std::vector<executorch::aten::SizesType> lr_sizes = {1};
    auto lr_tensor_impl = executorch::runtime::TensorImpl(
        executorch::aten::ScalarType::Float,
        std::size(lr_sizes),
        lr_sizes.data(),
        lr_data,
        std::vector<executorch::aten::DimOrderType>().data()
    );
    auto lr_tensor = executorch::runtime::Tensor(&lr_tensor_impl);

    // Input 2: projected_grad [1] (float32)
    float grad_data[1] = {0.0f};
    std::vector<executorch::aten::SizesType> grad_sizes = {1};
    auto grad_tensor_impl = executorch::runtime::TensorImpl(
        executorch::aten::ScalarType::Float,
        std::size(grad_sizes),
        grad_sizes.data(),
        grad_data,
        std::vector<executorch::aten::DimOrderType>().data()
    );
    auto grad_tensor = executorch::runtime::Tensor(&grad_tensor_impl);

    // 3. Training Loop
    ET_LOG(Info, "Starting training loop...");
    
    // Dummy targets (shifted input)
    int32_t targets[seq_len];
    for (int i = 0; i < seq_len; i++) targets[i] = 101 + (i % 100);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (int step = 0; step < num_steps; step++) {
            
            // Set inputs
            // Note: The order must match the exported model (input_ids, lr, grad)
            auto set_input_err = module->set_input(input_tensor, 0);
            if (set_input_err != executorch::runtime::Error::Ok) {
                ET_LOG(Error, "Failed to set input 0"); return 1;
            }
            
            module->set_input(lr_tensor, 1);
            module->set_input(grad_tensor, 2);

            // Execute Forward
            auto execute_err = module->execute();
            if (execute_err != executorch::runtime::Error::Ok) {
                ET_LOG(Error, "Execution failed at step %d", step);
                return 1;
            }

            // Get Output (Logits)
            auto outputs = module->get_outputs();
            if (!outputs.ok() || outputs->size() == 0) {
                ET_LOG(Error, "Failed to get outputs"); return 1;
            }
            
            auto logits_tensor = outputs->at(0).toTensor();
            float* logits_ptr = logits_tensor.const_data_ptr<float>();
            
            // Calculate Loss for both batch items (Positive & Negative perturbation)
            // Logits shape: [2, seq_len, vocab_size]
            float* logits_0 = logits_ptr;
            float* logits_1 = logits_ptr + (seq_len * vocab_size);
            
            float loss_0 = calculate_loss(logits_0, targets, vocab_size, seq_len);
            float loss_1 = calculate_loss(logits_1, targets, vocab_size, seq_len);
            
            // Estimate Gradient (Zeroth-Order)
            // grad = (Loss+ - Loss-) / (2 * eps)
            // eps is 1e-3 in our model
            float eps = 1e-3f;
            float estimated_grad = (loss_0 - loss_1) / (2 * eps);
            
            // Update grad for NEXT step
            grad_data[0] = estimated_grad;
            
            ET_LOG(Info, "Step %d: Loss0=%.4f, Loss1=%.4f, Grad=%.6f", 
                   step, loss_0, loss_1, estimated_grad);
        }
    }

    ET_LOG(Info, "Training complete!");
    return 0;
}

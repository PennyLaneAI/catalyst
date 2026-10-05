// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// catalyst_onnx_coprocessor: a per-message Backline coprocessor function that runs an ONNX model.
//
// The model has one input and one output. Each message's payload is the input tensor as raw bytes
// in row-major order, and the reply is the output tensor, likewise. A dynamic dimension of the
// input is taken as 1, so a model exported with a dynamic batch runs one sample per message.
//
// Built into libcatalyst_onnx_coprocessor.so. onnxruntime is loaded at run time, so the library
// neither links against it nor depends on a particular build: only onnxruntime's C API header is
// needed to compile it.
// The function is configured through the coprocessor's `fn.`-prefixed config keys, which reach the
// init hook of catalyst_onnx_coprocessor_info() without the prefix:
//
//   model=<path>      the .onnx file (required)
//   ort_lib=<path>    the onnxruntime shared library (default: libonnxruntime.so, or
//                     libonnxruntime.dylib on macOS)
//   provider=<name>   auto (default), cpu, migraphx, cuda or tensorrt
//   device=<index>    the GPU a GPU provider runs on (default 0)
//   threads=<count>   onnxruntime's intra-op threads (default 1), with one inter-op thread and
//                     intra-op spinning off, so the pool does not compete with the transport's
//                     spinning threads
//
// A GPU provider attaches through onnxruntime's OrtSessionOptionsAppendExecutionProvider_<Name>
// export, which only an onnxruntime build containing that provider has: onnxruntime-migraphx for
// migraphx on AMD GPUs, onnxruntime-gpu for cuda and tensorrt on NVIDIA GPUs. `auto` attaches the
// first GPU provider the loaded onnxruntime has, in the order migraphx then cuda, and fails if
// that provider cannot attach (for example when its GPU libraries are missing). Only an
// onnxruntime with no GPU provider runs on the CPU under `auto`. The provider in use is printed
// when the model loads.
//
// One context serves one coprocessor, whose worker calls the function from a single thread.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ConfigParser.hpp"
#include "Transport.hpp"
#include "onnxruntime_c_api.h"

namespace {

// The onnxruntime library loaded when the config names none, from the dynamic loader's search path.
#ifdef __APPLE__
constexpr const char *kDefaultOrtLib = "libonnxruntime.dylib";
#else
constexpr const char *kDefaultOrtLib = "libonnxruntime.so";
#endif

// The u32 decoder_id and u32 seq_num that end every request frame (see WireProtocol.hpp).
constexpr std::size_t kFrameTrailerBytes = 8;

// A GPU provider: its config name and the name onnxruntime exports its attach function under.
struct GpuProvider {
    const char *key;
    const char *export_name;
};
// The first kAutoProviders entries are the ones `auto` tries, in this order.
constexpr GpuProvider kGpuProviders[] = {
    {"migraphx", "MIGraphX"}, {"cuda", "CUDA"}, {"tensorrt", "Tensorrt"}};
constexpr std::size_t kAutoProviders = 2;

std::size_t element_bytes(ONNXTensorElementDataType type) {
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return 8;
    default:
        throw std::runtime_error("unsupported ONNX tensor element type " +
                                 std::to_string(static_cast<int>(type)));
    }
}

// The onnxruntime environment of the library `api` belongs to, created on first use and kept for
// the life of the process. onnxruntime has one environment per process, shared by every user of the
// library, including its Python package: releasing it, or unloading the library, while another user
// holds sessions leaves them with freed state (a later MIGraphX session then crashes). So no
// context releases it, and the library is never unloaded.
OrtEnv *shared_env(const OrtApi *api) {
    static std::mutex mutex;
    static std::map<const OrtApi *, OrtEnv *> envs;
    std::lock_guard<std::mutex> lock(mutex);
    OrtEnv *&env = envs[api];
    if (!env) {
        if (OrtStatus *status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "catalyst", &env)) {
            std::string message = api->GetErrorMessage(status);
            api->ReleaseStatus(status);
            envs.erase(api);
            throw std::runtime_error(message);
        }
    }
    return env;
}

class OnnxCoprocessor {
  public:
    explicit OnnxCoprocessor(std::string_view config) {
        namespace cfg = catalyst::transport::common::configparser;
        std::string model, ort_lib = kDefaultOrtLib, provider = "auto";
        int device = 0;
        int threads = 1;
        cfg::for_each_kv(config, [&](std::string_view key, std::string_view value) {
            if (key == "model") {
                model.assign(value);
            } else if (key == "ort_lib") {
                ort_lib.assign(value);
            } else if (key == "provider") {
                provider.assign(value);
            } else if (key == "device") {
                device = cfg::parse_index(value, "device");
            } else if (key == "threads") {
                threads = cfg::parse_index(value, "threads");
            } else if (key == "in_bytes") {
                expected_in_bytes_ = static_cast<std::size_t>(cfg::parse_index(value, "in_bytes"));
            } else if (key == "out_bytes") {
                expected_out_bytes_ =
                    static_cast<std::size_t>(cfg::parse_index(value, "out_bytes"));
            } else {
                throw std::runtime_error("unknown config key '" + std::string(key) + "'");
            }
        });
        // Validate everything the config controls before onnxruntime is loaded.
        if (model.empty()) {
            throw std::runtime_error("config needs model=<path to .onnx>");
        }
        if (!is_known_provider(provider)) {
            throw std::runtime_error("unknown provider '" + provider +
                                     "', expected auto, cpu, migraphx, cuda or tensorrt");
        }
        if (threads < 1) {
            throw std::runtime_error("threads must be at least 1");
        }
        if (!std::filesystem::is_regular_file(model)) {
            throw std::runtime_error("no model file at '" + model + "'");
        }

        handle_ = dlopen(ort_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            throw std::runtime_error("cannot load onnxruntime: " + std::string(dlerror()));
        }
        using GetApiBase = const OrtApiBase *(*)();
        auto get_api_base = reinterpret_cast<GetApiBase>(dlsym(handle_, "OrtGetApiBase"));
        if (!get_api_base) {
            throw std::runtime_error(ort_lib + " does not export OrtGetApiBase");
        }
        const OrtApiBase *api_base = get_api_base();
        api_ = api_base->GetApi(ORT_API_VERSION);
        if (!api_) {
            // C API version N ships with onnxruntime 1.N.
            throw std::runtime_error(ort_lib + " is onnxruntime " + api_base->GetVersionString() +
                                     ", which does not provide C API version " +
                                     std::to_string(ORT_API_VERSION) + ": onnxruntime 1." +
                                     std::to_string(ORT_API_VERSION) + " or newer is required");
        }

        env_ = shared_env(api_);
        OrtSessionOptions *options = nullptr;
        check(api_->CreateSessionOptions(&options));
        try {
            check(api_->SetIntraOpNumThreads(options, threads));
            check(api_->SetInterOpNumThreads(options, 1));
            check(api_->AddSessionConfigEntry(options, "session.intra_op.allow_spinning", "0"));
            std::string in_use = "cpu";
            if (provider == "auto") {
                in_use = append_first_gpu_provider(options, device);
            } else if (provider != "cpu") {
                append_provider(options, provider, device);
                in_use = provider;
            }
            check(api_->CreateSession(env_, model.c_str(), options, &session_));
            std::cerr << "[onnx] running " << model << " on "
                      << (in_use == "cpu" ? std::string("the CPU")
                                          : in_use + " (device " + std::to_string(device) + ")")
                      << "\n";
        } catch (...) {
            api_->ReleaseSessionOptions(options);
            throw;
        }
        api_->ReleaseSessionOptions(options);
        check(api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_));
        describe_io();
    }

    ~OnnxCoprocessor() {
        if (api_) {
            if (memory_) {
                api_->ReleaseMemoryInfo(memory_);
            }
            if (session_) {
                api_->ReleaseSession(session_);
            }
        }
        // The environment and the library stay for the life of the process (see shared_env).
    }

    OnnxCoprocessor(const OnnxCoprocessor &) = delete;
    OnnxCoprocessor &operator=(const OnnxCoprocessor &) = delete;

    // Throw if the message sizes the config names (in_bytes, out_bytes) differ from the model's
    // input and output tensors. A size the config does not name is not checked.
    void check_message_sizes() const {
        if (expected_in_bytes_ && *expected_in_bytes_ != input_bytes_) {
            throw std::runtime_error("the model input is " + std::to_string(input_bytes_) +
                                     " B, but the controller sends " +
                                     std::to_string(*expected_in_bytes_) + " B (in_bytes)");
        }
        if (expected_out_bytes_ && *expected_out_bytes_ != output_bytes_) {
            throw std::runtime_error("the model output is " + std::to_string(output_bytes_) +
                                     " B, but the controller expects " +
                                     std::to_string(*expected_out_bytes_) + " B (out_bytes)");
        }
    }

    // Run the model on `in_bytes` of payload, write the output tensor to `out`, and return its size
    // in bytes, or COPROCESSOR_FN_ERROR if the payload is too small, the output does not fit, or
    // inference fails.
    std::size_t run(const void *in, std::size_t in_bytes, void *out, std::size_t out_cap) {
        if (in_bytes < input_bytes_) {
            std::cerr << "[onnx] " << in_bytes << " B of input is smaller than the model input of "
                      << input_bytes_ << " B\n";
            return catalyst::transport::COPROCESSOR_FN_ERROR;
        }
        OrtValue *input = nullptr;
        OrtValue *output = nullptr;
        std::size_t written = 0;
        try {
            check(api_->CreateTensorWithDataAsOrtValue(memory_, const_cast<void *>(in),
                                                       input_bytes_, input_shape_.data(),
                                                       input_shape_.size(), input_type_, &input));
            const char *input_name = input_name_.c_str();
            const char *output_name = output_name_.c_str();
            check(api_->Run(session_, nullptr, &input_name, &input, 1, &output_name, 1, &output));
            written = copy_output(output, out, out_cap);
        } catch (const std::exception &e) {
            std::cerr << "[onnx] " << e.what() << "\n";
            written = catalyst::transport::COPROCESSOR_FN_ERROR;
        }
        if (output) {
            api_->ReleaseValue(output);
        }
        if (input) {
            api_->ReleaseValue(input);
        }
        return written;
    }

  private:
    void check(OrtStatus *status) const {
        if (status) {
            std::string message = api_->GetErrorMessage(status);
            api_->ReleaseStatus(status);
            throw std::runtime_error(message);
        }
    }

    using Append = OrtStatus *(*)(OrtSessionOptions *, int);

    // The attach function of a GPU provider, or null if this onnxruntime has none.
    Append provider_attach(const GpuProvider &gpu) const {
        const std::string symbol =
            std::string("OrtSessionOptionsAppendExecutionProvider_") + gpu.export_name;
        return reinterpret_cast<Append>(dlsym(handle_, symbol.c_str()));
    }

    void append_provider(OrtSessionOptions *options, const std::string &provider, int device) {
        for (const GpuProvider &gpu : kGpuProviders) {
            if (provider == gpu.key) {
                Append append = provider_attach(gpu);
                if (!append) {
                    throw std::runtime_error("this onnxruntime has no " + provider + " provider");
                }
                check(append(options, device));
                return;
            }
        }
        throw std::runtime_error("unknown provider '" + provider +
                                 "', expected auto, cpu, migraphx, cuda or tensorrt");
    }

    static bool is_known_provider(const std::string &provider) {
        if (provider == "auto" || provider == "cpu") {
            return true;
        }
        for (const GpuProvider &gpu : kGpuProviders) {
            if (provider == gpu.key) {
                return true;
            }
        }
        return false;
    }

    // Attach the first GPU provider, in kGpuProviders order, that this onnxruntime has, and return
    // its name. A provider it has but that fails to attach is an error. With no GPU provider at
    // all, return "cpu": the model runs on onnxruntime's CPU provider.
    std::string append_first_gpu_provider(OrtSessionOptions *options, int device) {
        for (std::size_t i = 0; i < kAutoProviders; ++i) {
            Append append = provider_attach(kGpuProviders[i]);
            if (!append) {
                continue;
            }
            if (OrtStatus *status = append(options, device)) {
                std::string reason = api_->GetErrorMessage(status);
                api_->ReleaseStatus(status);
                throw std::runtime_error(std::string("provider=auto: this onnxruntime has the ") +
                                         kGpuProviders[i].key +
                                         " provider, but it failed to attach: " + reason +
                                         ". Pass provider=cpu to run on the CPU");
            }
            return kGpuProviders[i].key;
        }
        return "cpu";
    }

    // Record the single input's name, type and shape, the single output's name, and both tensors'
    // sizes in bytes.
    void describe_io() {
        std::size_t inputs = 0, outputs = 0;
        check(api_->SessionGetInputCount(session_, &inputs));
        check(api_->SessionGetOutputCount(session_, &outputs));
        if (inputs != 1 || outputs != 1) {
            throw std::runtime_error("the model must have one input and one output, it has " +
                                     std::to_string(inputs) + " and " + std::to_string(outputs));
        }
        OrtAllocator *allocator = nullptr;
        check(api_->GetAllocatorWithDefaultOptions(&allocator));
        char *name = nullptr;
        check(api_->SessionGetInputName(session_, 0, allocator, &name));
        input_name_ = name;
        check(api_->AllocatorFree(allocator, name));
        check(api_->SessionGetOutputName(session_, 0, allocator, &name));
        output_name_ = name;
        check(api_->AllocatorFree(allocator, name));

        OrtTypeInfo *type_info = nullptr;
        check(api_->SessionGetInputTypeInfo(session_, 0, &type_info));
        try {
            const OrtTensorTypeAndShapeInfo *tensor = nullptr;
            check(api_->CastTypeInfoToTensorInfo(type_info, &tensor));
            check(api_->GetTensorElementType(tensor, &input_type_));
            std::size_t rank = 0;
            check(api_->GetDimensionsCount(tensor, &rank));
            input_shape_.resize(rank);
            check(api_->GetDimensions(tensor, input_shape_.data(), rank));
        } catch (...) {
            api_->ReleaseTypeInfo(type_info);
            throw;
        }
        api_->ReleaseTypeInfo(type_info);

        std::size_t elements = 1;
        for (std::int64_t &dim : input_shape_) {
            if (dim < 0) {
                dim = 1; // a dynamic dimension, such as the batch, runs one sample
            }
            elements *= static_cast<std::size_t>(dim);
        }
        input_bytes_ = elements * element_bytes(input_type_);

        check(api_->SessionGetOutputTypeInfo(session_, 0, &type_info));
        try {
            const OrtTensorTypeAndShapeInfo *tensor = nullptr;
            check(api_->CastTypeInfoToTensorInfo(type_info, &tensor));
            ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            check(api_->GetTensorElementType(tensor, &type));
            std::size_t rank = 0;
            check(api_->GetDimensionsCount(tensor, &rank));
            std::vector<std::int64_t> shape(rank);
            check(api_->GetDimensions(tensor, shape.data(), rank));
            std::size_t out_elements = 1;
            for (std::int64_t dim : shape) {
                out_elements *= static_cast<std::size_t>(dim < 0 ? 1 : dim);
            }
            output_bytes_ = out_elements * element_bytes(type);
        } catch (...) {
            api_->ReleaseTypeInfo(type_info);
            throw;
        }
        api_->ReleaseTypeInfo(type_info);
    }

    std::size_t copy_output(const OrtValue *output, void *out, std::size_t out_cap) const {
        OrtTensorTypeAndShapeInfo *info = nullptr;
        check(api_->GetTensorTypeAndShape(output, &info));
        std::size_t elements = 0;
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        OrtStatus *status = api_->GetTensorShapeElementCount(info, &elements);
        if (!status) {
            status = api_->GetTensorElementType(info, &type);
        }
        api_->ReleaseTensorTypeAndShapeInfo(info);
        check(status);
        const std::size_t bytes = elements * element_bytes(type);
        if (bytes > out_cap) {
            throw std::runtime_error("the model output of " + std::to_string(bytes) +
                                     " B does not fit the " + std::to_string(out_cap) + " B reply");
        }
        void *data = nullptr;
        check(api_->GetTensorMutableData(const_cast<OrtValue *>(output), &data));
        std::memcpy(out, data, bytes);
        return bytes;
    }

    void *handle_ = nullptr;
    const OrtApi *api_ = nullptr;
    OrtEnv *env_ = nullptr;
    OrtSession *session_ = nullptr;
    OrtMemoryInfo *memory_ = nullptr;
    std::string input_name_;
    std::string output_name_;
    ONNXTensorElementDataType input_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<std::int64_t> input_shape_;
    std::size_t input_bytes_ = 0;
    std::size_t output_bytes_ = 0; // the declared output tensor, a dynamic dimension taken as 1
    std::optional<std::size_t> expected_in_bytes_;
    std::optional<std::size_t> expected_out_bytes_;
};

// Load the model named by `config` and return the context the function is called with, or null
// with the reason on stderr.
void *onnx_init(const char *config) {
    try {
        auto ctx = std::make_unique<OnnxCoprocessor>(config ? config : "");
        ctx->check_message_sizes();
        return ctx.release();
    } catch (const std::exception &e) {
        std::cerr << "[onnx] " << e.what() << "\n";
        return nullptr;
    }
}

void onnx_fini(void *ctx) { delete static_cast<OnnxCoprocessor *>(ctx); }

// Version 1: the fields this library fills.
constexpr catalyst::transport::CoprocessorFnInfo kInfo{
    .abi_version = 1,
    .init = &onnx_init,
    .fini = &onnx_fini,
};

} // namespace

#define CATALYST_ONNX_EXPORT __attribute__((visibility("default")))

extern "C" {

// The function's lifecycle hooks: init loads the model, fini releases it.
CATALYST_ONNX_EXPORT const catalyst::transport::CoprocessorFnInfo *
catalyst_onnx_coprocessor_info() {
    return &kInfo;
}

// The model runs on the frame's payload, which ends where the frame's decoder_id and seq_num begin.
CATALYST_ONNX_EXPORT std::size_t catalyst_onnx_coprocessor(const void *in, std::size_t in_len,
                                                           void *out, std::size_t out_cap,
                                                           void *ctx) {
    if (!ctx || !in || !out || in_len < kFrameTrailerBytes) {
        return catalyst::transport::COPROCESSOR_FN_ERROR;
    }
    return static_cast<OnnxCoprocessor *>(ctx)->run(in, in_len - kFrameTrailerBytes, out, out_cap);
}

} // extern "C"

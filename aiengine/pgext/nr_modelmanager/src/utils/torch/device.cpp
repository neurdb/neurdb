#include "device.h"

#include <torch/torch.h>

void *device = nullptr;

void initialize_device() {
  // if CUDA is available, set the device to kCUDA
  if (torch::cuda::is_available()) {
    device = new torch::Device(torch::kCUDA);
  } else {
    // otherwise, set the device to CPU
    device = new torch::Device(torch::kCPU);
  }
}

bool device_is_cuda() {
  const auto device_type = static_cast<torch::Device *>(device)->type();
  return device_type == torch::kCUDA;
}

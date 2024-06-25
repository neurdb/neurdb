#ifndef DEVICE_H
#define DEVICE_H

#include <stdbool.h>


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// global variable to store the device, use void* to avoid including torch header file
extern void *device; // torch::Device

/**
 * @description: Initialize the device to be used
 * Note: If CUDA is available, the device will be set to kCUDA, otherwise, it will be set to CPU
 */
void
initialize_device();

/**
 * @description: Check if the device is CUDA
 * @return {bool} - true if the device is CUDA, false otherwise
 */
bool
device_is_cuda();

#ifdef __cplusplus
}
#endif // __cplusplus
#endif //DEVICE_H

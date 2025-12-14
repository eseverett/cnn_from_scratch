#pragma once

#ifndef CNN_FROM_SCRATCH_H
#define CNN_FROM_SCRATCH_H

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

typedef struct {
	int num_channels;
	int* channel_dims;
	int total_size;
	float* data;
} tensor_t; 


DLL_EXPORT tensor_t* create_tensor(int num_channels, int* channel_dims);
DLL_EXPORT void free_tensor(tensor_t* tensor);

#endif // CNN_FROM_SCRATCH_H
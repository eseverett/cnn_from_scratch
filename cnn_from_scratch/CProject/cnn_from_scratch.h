#pragma once

#ifndef CNN_FROM_SCRATCH_H
#define CNN_FROM_SCRATCH_H

typedef struct {
	int num_channels;
	int* channel_dims;
	int total_size;
	float* data;
} tensor_t; 


tensor_t* create_tensor(int num_channels, int* channel_dims);
void free_tensor(tensor_t* tensor);

#endif // CNN_FROM_SCRATCH_H
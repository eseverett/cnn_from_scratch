#include "cnn_from_scratch.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

tensor_t* create_tensor(int num_channels, int* channel_dims) {
	// Creates a tensor with specified number of channels and dimensions.
	// Allocates memory for the tensor structure, channel dimensions, and data.
	// Initializes the data to zero.

	if (num_channels <= 0 || channel_dims == NULL) {
		return NULL; 
	}


	tensor_t* tensor = (tensor_t*)malloc(sizeof(tensor_t));

	if (tensor == NULL) {
		return NULL;
	}


	tensor->num_channels = num_channels;
	tensor->channel_dims = (int*)malloc(num_channels * sizeof(int));

	if (tensor->channel_dims == NULL) {
		free(tensor);
		return NULL;
	}
	
	int total_data_size = 1;
	for (int i = 0; i < num_channels; i++) {
		tensor->channel_dims[i] = channel_dims[i];
		total_data_size *= channel_dims[i];
	}

	tensor->total_size = total_data_size; 

	if (total_data_size <= 0) {
		free(tensor->channel_dims);
		free(tensor);
		return NULL;
	}

	tensor->data = (float*)malloc(total_data_size * sizeof(float));

	if (tensor->data == NULL) {
		free(tensor->channel_dims);
		free(tensor->data);
		free(tensor);
		return NULL;
	}

	for (int i = 0; i < total_data_size; i++) {
		tensor->data[i] = 0.0f; 
	}

	return tensor;

}

void free_tensor(tensor_t* tensor) {
	//Frees the memory allocated for the tensor structure, channel dimensions, and data.
	if (tensor != NULL) {
		free(tensor->data);
		free(tensor->channel_dims);
		free(tensor);
	}
}

#include <cnpy.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dnnl.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "device.h"
#define ONE_BYTE 8

using namespace dnnl;
// function to save the output from dnnl memory to npy
void save_to_npy(memory &output_mem, memory::dims &output_dims,
                 const std::string &filename);
// function to save the output from float array to npy
void save_to_npy(float *final_output_data, const std::string &filename,
                 memory::dims &output_dims);
// util functions to read and write data to and from dnnl memory
inline void write_to_dnnl_memory(const void *handle, dnnl::memory &mem);
inline void read_from_dnnl_memory(void *handle, const dnnl::memory &mem);
// function to print the shapes of operation and save the output
void print_and_save(memory &output_mem, std::string node_name,
                    std::string op_type, memory::dims &ip_dims,
                    memory::dims &op_dims);
// function to initialize the PE array
void intialize_pe_array(void *pe_arrays[], int total_pe);
// util functions to write data to dnnl memory
inline void write_to_dnnl_memory(const void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *output = static_cast<uint8_t *>(mem.get_data_handle());
    if (output != nullptr) std::memcpy(output, handle, size);
  }
}
// util functions to read data from dnnl memory
inline void read_from_dnnl_memory(void *handle, const dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
    if (src != nullptr) std::memcpy(handle, src, size);
  }
}
// function to print the shapes of operation and save the output
void print_and_save(memory &output_mem, std::string node_name,
                    std::string op_type, memory::dims &ip_dims,
                    memory::dims &op_dims) {
  std::cout << "\n\n===== Node Name: " << node_name << " =====";
  std::cout << "\n Node OpType: " << op_type;
  std::cout << "\n Input Shape : ";
  for (auto dim : ip_dims) {
    std::cout << dim << " ";
  }
  std::cout << "\n Output Shape : ";
  for (auto dim : op_dims) {
    std::cout << dim << " ";
  }
  save_to_npy(output_mem, op_dims, node_name);
  std::cout << std::endl;
}
// function to save the output from dnnl memory to npy
void save_to_npy(memory &output_mem, memory::dims &output_dims,
                 const std::string &filename) {
  try {
    std::vector<size_t> vec_dims(output_dims.begin(), output_dims.end());
    unsigned long total_size = 1;
    for (auto dim : output_dims) {
      total_size *= dim;
    }
    std::vector<float> final_output_data(total_size);
    read_from_dnnl_memory(final_output_data.data(), output_mem);

    std::string npy_filename = "./outputs/cpp_" + filename + ".npy";
    cnpy::npy_save(npy_filename, &final_output_data[0], vec_dims, "w");
  } catch (...) {
    std::cout << "Error while saving dnnl memory" << std::endl;
  }
}
// function to save the output from float array to npy
void save_to_npy(float *final_output_data, const std::string &filename,
                 memory::dims &output_dims) {
  try {
    std::string npy_filename = "./outputs/cpp_" + filename + ".npy";
    std::vector<size_t> vec_dims(output_dims.begin(), output_dims.end());
    cnpy::npy_save(npy_filename, final_output_data, vec_dims, "w");
  } catch (...) {
    std::cout << "Error while saving float array" << std::endl;
  }
}

// function to initialize the PE array
int initialize_pe_array_for_convolution(void *pe_arrays[], float *input,
                                        float *weight, float *bias, int height,
                                        int width, int kernel_size, int stride,
                                        int padding, int input_channels,
                                        int output_channels) {
  int pe_counter = 1;
  int new_height = (((height) + padding) - kernel_size) / stride + 1;
  int new_width = (((width) + padding) - kernel_size) / stride + 1;
  double input_size = input_channels * ((height) + padding) * (width + padding);
  double output_size = output_channels * new_height * new_width;
  double weight_size =
      output_channels * input_channels * kernel_size * kernel_size;
  double bias_size = output_channels;
  double pe_required = ceil((input_size + output_size) /
                            (SIZE_PER_PE - (weight_size - bias_size)));

  // Round up pe_required to nearest multiple of 4
  // temp fix to make sure pe_required is a multiple of 4 and is a divisor of
  // HEIGHT
  pe_required = ceil(pe_required / 4.0) * 4;
  while (height % static_cast<int>(pe_required) != 0) {
    pe_required += 4;
  }
  // Ensure pe_required is a divisor of HEIGHT
  int split = static_cast<int>(pe_required);
  for (int i = 0; i < input_channels; i++) {
    for (int j = 0; j < (height / split) + (padding); j++) {
      for (int k = 0; k < width + (padding); k++) {
        for (int l = 0; l < split; l++) {
          float *pe = static_cast<float *>(pe_arrays[pe_counter + l]);
          if (l == 0) {
            int jj = j - (kernel_size / 2);
            int kk = k - (kernel_size / 2);
            if (jj >= 0 && kk >= 0 && kk < width) {
              pe[i * (height / split + (padding)) * (width + (padding)) +
                 j * (width + (padding)) + k] =
                  input[i * height * width + jj * width + kk];
            } else {
              pe[i * (height / split + (padding)) * (width + (padding)) +
                 j * (width + (padding)) + k] = 0;
            }
          } else {
            int jj = j + (l * (height / split)) - (kernel_size / 2);
            int kk = k - (kernel_size / 2);
            if (jj >= 0 && jj < height && kk >= 0 && kk < width) {
              pe[i * ((height / split) + padding) * (width + padding) +
                 j * (width + padding) + k] =
                  input[i * height * width + jj * width + kk];
            } else {
              pe[i * ((height / split) + padding) * (width + padding) +
                 j * (width + padding) + k] = 0;
            }
          }
        }
      }
    }
  }
  // TODO:  split the weights incase if it doesn't fit in the memory and fit it
  // in same PE as input and output Initialize the PE's with the weights
  int inputs_size =
      input_channels * ((height / split) + padding) * (width + padding);
  for (int npe = 0; npe < split; npe++) {
    float *pe = static_cast<float *>(pe_arrays[pe_counter + npe]);
    for (int i = 0; i < output_channels; i++) {
      for (int j = 0; j < input_channels; j++) {
        for (int k = 0; k < kernel_size; k++) {
          for (int l = 0; l < kernel_size; l++) {
            pe[i * input_channels * kernel_size * kernel_size +
               j * kernel_size * kernel_size + k * kernel_size + l +
               inputs_size] =
                weight[i * input_channels * kernel_size * kernel_size +
                       j * kernel_size * kernel_size + k * kernel_size + l];
          }
        }
      }
    }
  }
  // Initialize the PE's with the bias
  int weights_size =
      output_channels * input_channels * kernel_size * kernel_size;
  for (int npe = 0; npe < split; npe++) {
    float *pe = static_cast<float *>(pe_arrays[pe_counter + npe]);
    pe += (inputs_size + weights_size);
    for (int i = 0; i < output_channels; i++) {
      pe[i] = bias[i];
    }
  }
  return split;
}
void read_from_npy(std::string filename, float *data) {
  cnpy::NpyArray arr = cnpy::npy_load(filename);
  if (arr.word_size != sizeof(float)) {
    if (filename.find("input") != std::string::npos) {
    }
    throw std::runtime_error("Unsupported data type in .npy file");
  }
  memcpy(data, arr.data<float>(), arr.num_bytes());
}

// Common function for convolution with dnnl kernel
memory dnnl_conv(engine &eng, stream &eng_stream, memory::dims &src_dims,
                 memory::dims &weights_dims, memory::dims &bias_dims,
                 memory::dims &output_dims, int st, int pad, float *src_data,
                 float *weights_data, float *bias_data) {
  auto src_md =
      memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
  auto weights_md = memory::desc(weights_dims, memory::data_type::f32,
                                 memory::format_tag::oihw);
  auto bias_md =
      memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
  auto output_md = memory::desc(output_dims, memory::data_type::f32,
                                memory::format_tag::nchw);
  // Create memory objects
  auto src_mem = memory(src_md, eng);
  auto weights_mem = memory(weights_md, eng);
  auto bias_mem = memory(bias_md, eng);
  auto conv_output_mem = memory(output_md, eng);

  // Write data to memory objects
  write_to_dnnl_memory(weights_data, weights_mem);
  write_to_dnnl_memory(bias_data, bias_mem);
  write_to_dnnl_memory(src_data, src_mem);
  // Create convolution primitive
  auto conv_pd = convolution_forward::primitive_desc(
      eng, prop_kind::forward_inference, algorithm::convolution_direct, src_md,
      weights_md, bias_md, output_md, {st, st}, {pad, pad}, {pad, pad});

  auto conv_prim = convolution_forward(conv_pd);

  // Execute convolution primitive
  conv_prim.execute(eng_stream, {{DNNL_ARG_SRC, src_mem},
                                 {DNNL_ARG_WEIGHTS, weights_mem},
                                 {DNNL_ARG_BIAS, bias_mem},
                                 {DNNL_ARG_DST, conv_output_mem}});
  eng_stream.wait();
  return conv_output_mem;
}
memory dnnl_relu(engine &eng, stream &eng_stream, memory::dims &input_dims,
                 float *src_data) {
  auto src_mem_desc = memory::desc(input_dims, memory::data_type::f32,
                                   memory::format_tag::nchw);
  auto src_mem = memory(src_mem_desc, eng, src_data);
  auto output_mem_desc = memory::desc(input_dims, memory::data_type::f32,
                                      memory::format_tag::nchw);
  auto output_mem = memory(output_mem_desc, eng);
  auto relu_pd = eltwise_forward::primitive_desc(
      eng, prop_kind::forward_inference, algorithm::eltwise_relu,
      src_mem.get_desc(), src_mem.get_desc(), 0.0f,
      0.0f);  // alpha and beta for ReLU are typically 0
  auto relu_prim = eltwise_forward(relu_pd);
  write_to_dnnl_memory(src_data, src_mem);
  relu_prim.execute(eng_stream,
                    {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, output_mem}});
  eng_stream.wait();
  return output_mem;
}
// Common function for convolution with dfa
void convolution_dfa(engine &eng, stream &eng_stream, memory::dims &input_dims,
                     memory::dims &weights_dims, memory::dims &bias_dims,
                     memory::dims &output_dims, void *pe_arrays[], int st,
                     int pad, int pe_start_index, int split) {
  std::string conv_name = "conv_output";
  int input_size =
      input_dims[3] * input_dims[2] * input_dims[1] * input_dims[0];
  int weight_size =
      weights_dims[3] * weights_dims[2] * weights_dims[1] * weights_dims[0];
  int bias_size = weights_dims[0];
  /* Perfom convolution for all the 4 split parts */
  for (int i = 0; i < split; i++) {
    float *pe = static_cast<float *>(pe_arrays[pe_start_index + i]);
    auto conv_output = dnnl_conv(
        eng, eng_stream, input_dims, weights_dims, bias_dims, output_dims, st,
        0, (pe), (pe + input_size), (pe + input_size + weight_size));
    float *conv_output_data = (pe + input_size + weight_size + bias_size);
    read_from_dnnl_memory(conv_output_data, conv_output);
    print_and_save(conv_output, conv_name + std::to_string(i + 1), "conv",
                   input_dims, output_dims);
  }
}
// Common function for relu with dfa
void relu_dfa(engine &eng, stream &eng_stream, memory::dims &input_dims,
              memory::dims &output_dims, void *pe_arrays[], int pe_start_index,
              int split, int offset) {
  std::string relu_name = "relu_output";
  /* Perfom ReLU for all the 4 split parts */
  for (int i = 0; i < split; i++) {
    float *pe =
        static_cast<float *>(pe_arrays[pe_start_index + i]);  // for input
    float *relu_output_data =
        static_cast<float *>(pe_arrays[pe_start_index + split + i]) +
        offset;  // output next pe
    auto relu_output = dnnl_relu(eng, eng_stream, input_dims, pe + offset);
    read_from_dnnl_memory(relu_output_data, relu_output);
    print_and_save(relu_output, relu_name + std::to_string(i + 1), "relu",
                   input_dims, output_dims);
  }
}

void dump_merged(int size, int pe_start_index, std::string layer_name,
                 void *pe_arrays[], int split, memory::dims &output_dims,
                 int offset, engine &eng) {
  //  Merge the outputs into a single array
  float *merged_pe = (float *)std::malloc(size * sizeof(float));
  int channels = output_dims[1];
  int height = output_dims[2] * split;
  int width = output_dims[3];

  for (int i = 0; i < channels; i++) {
    for (int j = 0; j < height / split; j++) {
      for (int k = 0; k < width; k++) {
        for (int l = 0; l < split; l++) {
          float *pe =
              static_cast<float *>(pe_arrays[pe_start_index + l]) + (offset);
          if (l == 0) {
            merged_pe[i * height * width + j * width + k] =
                pe[i * ((height / split) * width) + j * width + k];
          } else {
            merged_pe[i * height * width +
                      (j + (l * (height / split))) * width + k] =
                pe[(i * ((height / split) * width) + j * width + k)];
          }
        }
      }
    }
  }

  memory::dims merged_output_dims = {output_dims[0], output_dims[1],
                                     output_dims[2] * split, output_dims[3]};
  save_to_npy(merged_pe, (layer_name + "merged"), merged_output_dims);
  std::free(merged_pe);
}
// function to run the resnet18
void run_resnet18(engine &eng, stream &eng_stream, void *pe_arrays[]) {
  // declare input and weight
  int pe_start_index = 1;
  // define the input, weight, bias and output shapes
  memory::dims input_dims;
  memory::dims weights_dims;
  memory::dims bias_dims;
  memory::dims output_dims;
  /* layer 1*/
  // dimensions for first layer
  int height = 224, width = 224, input_channels = 3, output_channels = 64,
      kernel_size = 7, stride = 2, padding = 6;
  int input_size = 1 * input_channels * height * width;
  int weight_size =
      output_channels * input_channels * kernel_size * kernel_size;
  int bias_size = output_channels;
  // load input and weights for first layer from npy files
  float *input = (float *)std::malloc(input_size * sizeof(float));
  float *weight = (float *)std::malloc(weight_size * sizeof(float));
  float *bias = (float *)std::malloc(bias_size * sizeof(float));
  read_from_npy("inputs/py_input.npy", input);
  read_from_npy("weights/conv1_wt.npy", weight);
  read_from_npy("weights/conv1_bs.npy", bias);
  // initialize the PE array
  int split = initialize_pe_array_for_convolution(
      pe_arrays, input, weight, bias, height, width, kernel_size, stride,
      padding, input_channels, output_channels);
  std::cout << "No of PE's to split into :  " << split << std::endl;
  input_dims = {1, input_channels, ((height / split) + padding),
                (width + padding)};
  weights_dims = {output_channels, input_channels, kernel_size, kernel_size};
  bias_dims = {output_channels};
  int new_height = (((height / split) + padding) - kernel_size) / stride + 1;
  int new_width = (((width) + padding) - kernel_size) / stride + 1;
  output_dims = {1, output_channels, new_height, new_width};
  // since input is padded kernel padding is 0
  convolution_dfa(eng, eng_stream, input_dims, weights_dims, bias_dims,
                  output_dims, pe_arrays, stride, 0, pe_start_index, split);
  int size =
      output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3] * split;
  input_size = input_dims[3] * input_dims[2] * input_dims[1] * input_dims[0];
  weight_size =
      weights_dims[3] * weights_dims[2] * weights_dims[1] * weights_dims[0];
  dump_merged(size, 1, "conv1_", pe_arrays, split, output_dims,
              (input_size + weight_size + bias_size), eng);
  /* layer 1 */
  /* layer 2 */
  input_dims = {1, output_channels, new_height, new_width};
  output_dims = {1, output_channels, new_height, new_width};
  relu_dfa(eng, eng_stream, input_dims, output_dims, pe_arrays, pe_start_index,
           split, (input_size + weight_size + bias_size));
  dump_merged(size, pe_start_index + split, "relu1_", pe_arrays, split,
              output_dims, (input_size + weight_size + bias_size), eng);
  /* layer 2*/
}

int main() {
  const int total_pe = (PE_ROWS * PE_COLUMNS);
  void *pe_arrays[total_pe];

  for (int i = 0; i < total_pe; i++) {
    pe_arrays[i] = std::malloc((SIZE_PER_PE * 4) * ONE_BYTE);
  }

  // Engine creation
  engine eng(engine::kind::cpu, 0);
  stream eng_stream(eng);

  run_resnet18(eng, eng_stream, pe_arrays);
  std::cout << "Dumped outputs to output/ folder" << std::endl;

  return 0;
}

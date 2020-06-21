#include "fpga_api.h"
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstring>

#define min(x, y) (((x) < (y)) ? (x) : (y))

FPGA::FPGA(off_t data_addr, off_t output_addr, int m_size, int v_size)
{
  m_size_ = 48;
  v_size_ = 64;
  data_size_ = (m_size_ + 1) * v_size_ * sizeof(int); // fpga bram data size

  fd_ = open("/dev/mem", O_RDWR);
  qdata_ = static_cast<int *>(mmap(NULL, data_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, data_addr));
  output_ = static_cast<unsigned int *>(mmap(NULL, sizeof(unsigned int), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, output_addr));

  num_block_call_ = 0;
}

FPGA::~FPGA()
{
  munmap(qdata_, data_size_);
  munmap(output_, sizeof(unsigned int));
  close(fd_);
}

int *FPGA::qmatrix(void)
{
  return qdata_ + v_size_;
}

int *FPGA::qvector(void)
{
  return qdata_;
}

void FPGA::reset(void)
{
  num_block_call_ = 0;
}

int FPGA::num_block_call(void)
{
  return num_block_call_;
}

void quantize(const float* input, char* quantized, int num_input, int bits_min, int bits_max, int offset, float scale)
{
  for(int i = 0; i < num_input; i++)
  {
    quantized[i] = 0; // TODO: convert floating point to quantized valuea
    quantized[i] = (input[i]/scale) + offset;
  }
}

void dequantize(int* quantized, float* output, int num_output, int offset, float scale)
{
  for(int i = 0; i < num_output; i++)
  {
    output[i] = 0; // TODO: convert quantized value to floating pointa
    output[i] = scale*(quantized[i]-offset);
  }
}

const int *__attribute__((optimize("O0"))) FPGA::qblockMV(Compute* comp)
{
  num_block_call_ += 1;

  // fpga version
  *output_ = 0x5555;
  while (*output_ == 0x5555)
    ;

  return qdata_;
}

void FPGA::largeMV(const float *large_mat, const float *input, float *output, int num_input, int num_output, Compute* comp)
{
  int *vec = this->qvector();
  int *mat = this->qmatrix();

  char *qlarge_mat = new char[num_input*num_output];
  char *qinput = new char[num_input];
  int *qoutput = new int[num_output];

  // quantize
  int act_bits_min = 0;
  int act_bits_max = (1<<(comp->act_bits-1))-1;

  float act_scale = 0; // TODO calculate the scale factor
  int act_offset = 0; // TODO calculate the zero-offseta
  act_scale = (comp->act_max-comp->act_min)/(float)act_bits_max;
  act_offset=-(comp->act_min)/act_scale;
  quantize(input, qinput, num_input, act_bits_min, act_bits_max, act_offset, act_scale);

  int weight_bits_min = 0;
  int weight_bits_max = (1<<(comp->weight_bits-1))-1;

  float weight_scale = 0; // TODO calculate the scale factor
  int weight_offset = 0; // TODO calculate the zero-offset
  weight_scale = (comp->weight_max-comp->weight_min)/(float)weight_bits_max;
  weight_offset=-(comp->weight_min)/weight_scale;
  quantize(large_mat, qlarge_mat, num_input*num_output, weight_bits_min, weight_bits_max, weight_offset, weight_scale);

  // 0) Initialize output vector
  for (int i = 0; i < num_output; ++i)
    qoutput[i] = 0;

  for (int i = 0; i < num_output; i += m_size_)
  {
    for (int j = 0; j < num_input; j += v_size_)
    {
      // 0) Initialize input vector
      int block_row = min(m_size_, num_output - i);
      int block_col = min(v_size_, num_input - j);
      memset(vec, 0, sizeof(char)*v_size_);
      memset(mat, 0, sizeof(char)*m_size_*v_size_);

      // 1) Assign a vector
      // IMPLEMENT THIS

      memcpy(vec, qinput+j, sizeof(char)*block_col);

      // 2) Assign a matrix
      // IMPLEMENT THIS

      for(int r = 0; r < block_row; r++){
        memcpy(mat+r*v_size_, qlarge_mat+(i+r)*num_input+j, sizeof(char)*block_col);
      }

      // 3) Call a function `qblockMV() to execute MV multiplication
      const int* ret = this->qblockMV(comp);

      // 4) Accumulate intermediate results
      for(int row = 0; row < block_row; ++row)
        qoutput[i + row] += ret[row];
    }
  }

  dequantize(qoutput, output, num_output, 0, act_scale*weight_scale);
}

void FPGA::convLowering(const std::vector<std::vector<std::vector<std::vector<float>>>> &cnn_weights,
                        std::vector<std::vector<float>> &new_weights,
                        const std::vector<std::vector<std::vector<float>>> &inputs,
                        std::vector<std::vector<float>> &new_inputs)
{
  /*
   * Arguments:
   *
   * conv_weights: [conv_channel, input_channel, conv_height, conv_width]
   * new_weights: [?, ?]
   * inputs: [input_channel, input_height, input_width]
   * new_inputs: [?, ?]
   *
   */

  int conv_channel = cnn_weights.size();
  int input_channel = cnn_weights[0].size();
  int conv_height = cnn_weights[0][0].size();
  int conv_width = cnn_weights[0][0][0].size();
  //int input_channel = inputs.size();
  int input_height = inputs[0].size();
  int input_width = inputs[0][0].size();

  // IMPLEMENT THIS
  // For example,
  // new_weights[0][0] = cnn_weights[0][0][0][0];
  // new_inputs[0][0] = inputs[0][0][0];

  int conv_size = conv_height*conv_width;

  for(int conch = 0; conch < conv_channel; conch++){
    for(int inch = 0; inch < input_channel; inch++){
      for(int i = 0; i < conv_height; i++){
        for(int j = 0; j < conv_width; j++){
          new_weights[conch][inch*conv_size + i*conv_width + j] =
            cnn_weights[conch][inch][i][j];
        }
      }
    }
  }

  int out_height = input_height - conv_height + 1;
  int out_width = input_width - conv_width + 1;

  for(int inch = 0; inch < input_channel; inch++){
    for(int conr = 0; conr < conv_height; conr++){
      for(int conc = 0; conc < conv_width; conc++){
        for(int outr = 0; outr < out_height; outr++){
          for(int outc = 0; outc < out_width; outc++){
            new_inputs[inch*conv_size + conr*conv_width+conc][outr*out_width+outc]
              = inputs[inch][conr+outr][conc+outc];
          }
        }
      }
    }
  }

}

#include "fpga_api.h"
#include <stdio.h>
#include <iostream>
#include <cstring>

using namespace std;

#define min(x, y) (((x) < (y)) ? (x) : (y))

FPGA::FPGA(off_t data_addr, off_t output_addr, int m_size, int v_size)
{
  m_size_ = m_size;
  v_size_ = v_size;
  data_size_ = (m_size_ + 1) * v_size_; // fpga bram data size

  qvec_ = new char[v_size_];
  qmat_ = new char[m_size_*v_size_];
  qout_ = new short[m_size_];

  output_ = new unsigned int[m_size_]; // use output_ as tempolar output
  data_ = new float[data_size_];

  num_block_call_ = 0;
}

FPGA::~FPGA()
{
  delete[] output_;
  delete[] data_;
  delete[] qvec_;
  delete[] qmat_;
  delete[] qout_;
}

float *FPGA::matrix(void)
{
  return data_ + v_size_;
}

float *FPGA::vector(void)
{
  return data_;
}

void FPGA::reset(void)
{
  num_block_call_ = 0;
}

int FPGA::num_block_call(void)
{
  return num_block_call_;
}

void quantize(float* input, char* quantized, int num_input, char bits_min, char bits_max, char offset, float scale)
{
  for(int i = 0; i < num_input; i++)
  {
    quantized[i] = 0; // TODO: convert floating point to quantized value
    quantized[i] = (input[i]/scale) + offset;
  }
}

void dequantize(short* quantized, float* output, int num_output, char offset, float scale)
{
  for(int i = 0; i < num_output; i++)
  {
    output[i] = 0; // TODO: convert quantized value to floating point
    output[i] = scale*(quantized[i]-offset);
  }
}

const float *FPGA::blockMV(Compute* comp)
{
  num_block_call_ += 1;

  // cpu version
  float *vec = this->vector();
  float *mat = this->matrix();
  float *out = reinterpret_cast<float *>(output_);

  if(comp->quantized)
  {
    char act_bits_min = 0;
    char act_bits_max = (1<<(comp->act_bits-1))-1;

    float act_scale = 0; // TODO calculate the scale factor
    char act_offset = 0; // TODO calculate the zero-offset
    act_scale = (comp->act_max - comp->act_min) / (float)act_bits_max;
    act_offset = -(comp->act_min)/act_scale;
    quantize(vec, qvec_, v_size_, act_bits_min, act_bits_max, act_offset, act_scale);

    char weight_bits_min = 0;
    char weight_bits_max = (1<<(comp->weight_bits-1))-1;

    float weight_scale = 0; // TODO calculate the scale factor
    char weight_offset = 0; // TODO calculate the zero-offset
    weight_scale = (comp->weight_max - comp->weight_min) / (float)weight_bits_max;
    weight_offset = -(comp->weight_min)/weight_scale;
    quantize(mat, qmat_, m_size_*v_size_, weight_bits_min, weight_bits_max, weight_offset, weight_scale);

    for (int i = 0; i < m_size_; ++i)
    {
      qout_[i] = 0;
      for (int j = 0; j < v_size_; ++j)
        qout_[i] += (qvec_[j]-act_offset) * (qmat_[v_size_ * i + j]-weight_offset);
    }

    dequantize(qout_, out, m_size_, 0, act_scale*weight_scale);
  }
  else
  {
    float vec_max = vec[0], vec_min = vec[0];
    for(int i = 1; i < v_size_; i++){
      if (vec_max < vec[i]) vec_max = vec[i];
      if (vec_min > vec[i]) vec_min = vec[i];
    }
    if(num_block_call_ == 1)
      printf("vec %f %f\n", vec_min, vec_max);

    for (int i = 0; i < m_size_; ++i)
    {
      out[i] = 0;
      for (int j = 0; j < v_size_; ++j)
        out[i] += vec[j] * mat[v_size_ * i + j];
    }
  }

  for (int i = 0; i < m_size_; ++i)
    data_[i] = out[i];

  return data_;
}

void FPGA::largeMV(const float *large_mat, const float *input, float *output, int num_input, int num_output, Compute* comp)
{
  float *vec = this->vector();
  float *mat = this->matrix();

  // 0) Initialize output vector
  for (int i = 0; i < num_output; ++i)
    output[i] = 0;

  for (int i = 0; i < num_output; i += m_size_)
  {
    for (int j = 0; j < num_input; j += v_size_)
    {
      // 0) Initialize input vector
      int block_row = min(m_size_, num_output - i);
      int block_col = min(v_size_, num_input - j);

      // 1) Assign a vector
      // IMPLEMENT THIS

      memset(vec, 0, sizeof(float)*(block_col+block_row*block_col));

      memcpy(vec, input+j, sizeof(float)*block_col);

      // 2) Assign a matrix
      // IMPLEMENT THIS

      for(int r = 0; r < block_row; r++){
        memcpy(mat+r*v_size_, large_mat+(i+r)*num_input+j, sizeof(float)*block_col);
      }

      // 3) Call a function `blockMV() to execute MV multiplication
      const float* ret = this->blockMV(comp);

      // 4) Accumulate intermediate results
      for (int row = 0; row < block_row; ++row)
        output[i + row] += ret[row];
    }
  }
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

  int conv_size = conv_height*conv_width;

  // IMPLEMENT THIS
  // For example,
  // new_weights[0][0] = cnn_weights[0][0][0][0];
  // new_inputs[0][0] = inputs[0][0][0];

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

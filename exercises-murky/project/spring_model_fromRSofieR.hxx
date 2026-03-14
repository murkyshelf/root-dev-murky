//Code generated automatically by TMVA for Inference of Model file [spring_model.onnx] at [Sat Mar 14 18:17:15 2026] 

#ifndef ROOT_TMVA_SOFIE_SPRING_MODEL
#define ROOT_TMVA_SOFIE_SPRING_MODEL

#include <algorithm>
#include <cmath>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_spring_model{
namespace BLAS{
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_net4bias = std::vector<float>(1);
float * tensor_net4bias = fTensor_net4bias.data();
std::vector<float> fTensor_net4weight = std::vector<float>(64);
float * tensor_net4weight = fTensor_net4weight.data();
std::vector<float> fTensor_net2bias = std::vector<float>(64);
float * tensor_net2bias = fTensor_net2bias.data();
std::vector<float> fTensor_net2weight = std::vector<float>(4096);
float * tensor_net2weight = fTensor_net2weight.data();
std::vector<float> fTensor_net0bias = std::vector<float>(64);
float * tensor_net0bias = fTensor_net0bias.data();
std::vector<float> fTensor_net0weight = std::vector<float>(64);
float * tensor_net0weight = fTensor_net0weight.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(512);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor netnet0Gemm_output_0 with size 256 bytes
float* tensor_netnet0Gemm_output_0 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor netnet1Tanh_output_0 with size 256 bytes
float* tensor_netnet1Tanh_output_0 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 256);

 // Allocating memory for intermediate tensor netnet2Gemm_output_0 with size 256 bytes
float* tensor_netnet2Gemm_output_0 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor netnet3Tanh_output_0 with size 256 bytes
float* tensor_netnet3Tanh_output_0 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 256);

 // Allocating memory for intermediate tensor output with size 4 bytes
float* tensor_output = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 252);


Session(std::string filename ="spring_model.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_net4bias, "tensor_net4bias", 1);
   ReadTensorFromStream(f, tensor_net4weight, "tensor_net4weight", 64);
   ReadTensorFromStream(f, tensor_net2bias, "tensor_net2bias", 64);
   ReadTensorFromStream(f, tensor_net2weight, "tensor_net2weight", 4096);
   ReadTensorFromStream(f, tensor_net0bias, "tensor_net0bias", 64);
   ReadTensorFromStream(f, tensor_net0weight, "tensor_net0weight", 64);
   f.close();

}

void doInfer(float const* tensor_input,  std::vector<float> &output_tensor_output ){


//--------- Gemm op_0 { 1 , 1 } * { 64 , 1 } -> { 1 , 64 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 64 * j;
      for (size_t k = 0; k < 64; k++) { 
         tensor_netnet0Gemm_output_0[y_index + k] = tensor_net0bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_netnet0Gemm_output_0, true, false, 64, 1, 1, 1, tensor_net0weight, tensor_input, 1,nullptr);

//------ TANH
   for (int id = 0; id < 64 ; id++){
      tensor_netnet1Tanh_output_0[id] = std::tanh(tensor_netnet0Gemm_output_0[id]);
   }

//--------- Gemm op_2 { 1 , 64 } * { 64 , 64 } -> { 1 , 64 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 64 * j;
      for (size_t k = 0; k < 64; k++) { 
         tensor_netnet2Gemm_output_0[y_index + k] = tensor_net2bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_netnet2Gemm_output_0, true, false, 64, 1, 64, 1, tensor_net2weight, tensor_netnet1Tanh_output_0, 1,nullptr);

//------ TANH
   for (int id = 0; id < 64 ; id++){
      tensor_netnet3Tanh_output_0[id] = std::tanh(tensor_netnet2Gemm_output_0[id]);
   }

//--------- Gemm op_4 { 1 , 64 } * { 1 , 64 } -> { 1 , 1 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = j;
      for (size_t k = 0; k < 1; k++) { 
         tensor_output[y_index + k] = tensor_net4bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_output, true, false, 1, 1, 64, 1, tensor_net4weight, tensor_netnet3Tanh_output_0, 1,nullptr);
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_output, output_tensor_output, 1);
}



std::vector<float> infer(float const* tensor_input){
   std::vector<float > output_tensor_output;
   doInfer(tensor_input, output_tensor_output );
   return {output_tensor_output};
}
};   // end of Session

} //TMVA_SOFIE_spring_model

#endif  // ROOT_TMVA_SOFIE_SPRING_MODEL

#include <iostream>
#include <memory>
//#include <stdio.h>
#include <unistd.h>

#include <torch/script.h>
#include <torch/torch.h>


int main() {
    // >>> yuv420 binary file 읽기.
    FILE* fp;
    fp = fopen("../python_kdw/resources/C01_BasketballDrill_832x480_50_QP37.yuv", "rb");
    if(fp == nullptr)  // c++ 에서 NULL 은 nullptr 이다.
    {
        std::cout << "buffer is NULL";
    }

    int width = 832;
    int height = 480;

    // 버퍼를 읽는다.
    auto* yuv_buffer = new unsigned char[width * height];
    fread(yuv_buffer, sizeof(unsigned char), width*height, fp);

    // torch::from_blob 는 unsigned char 가 들어가면 안되는 것 같아서 아래를 통해 형변환을 시켜주자.
    auto* yuv_buffer_float = new float[width * height];

    // 10진수=%d, 8진수=%o, 16진수=%x, # 을 추가하면 접두사도 출력됨)
    // 아래 코드를 통해 형변환 해줌. 동적 배열을 형변환 하는 최적의 코드는 무엇일까..
    for (int i = 0; i < height * width; i++){
            yuv_buffer_float[i] = yuv_buffer[i];
    }

    // 255 -> 1
    for (int i = 0; i < height * width; i++){
        yuv_buffer_float[i] = yuv_buffer_float[i] /= 255;
    }

    // 버퍼를 torch 의 tensor 로 옮겨주자.
    torch::DeviceType device_type;
    device_type = at::kCUDA;
    torch::Tensor input_tensor = torch::from_blob(yuv_buffer_float, {1, 1, height, width});

    // create a vector of inputs. (torch script 모델을 사용하려면 이 형식으로 바꿔줘야한다)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor.to(at::kCUDA));  // cuda 사용.

    //Execute the model and turn its output into a tensor.
    torch::jit::script::Module module;
    module = torch::jit::load("../python_kdw/resources/hevc009_RA_qp37/netG_traced.pt");

    // https://discuss.pytorch.org/t/difference-between-torch-tensor-and-at-tensor/35806
    // at::Tensor is not differentiable while torch::Tensor is.
    at::Tensor output = module.forward(inputs).toTensor();
    auto output_tensor = output.cpu().detach();

    std::memcpy(yuv_buffer_float, output_tensor.data_ptr(), sizeof(float) * width * height);

    inputs.clear();

    auto* yuv_buffer_recon = new unsigned char[width * height];
    float p = 0;
    // tensor to unsigned char
    for (int i = 0; i < height * width; i++){
        p = (unsigned char)(yuv_buffer_float[i] * 255);
        if(p > 255)
            p = 255;
        if(p < 0)
            p = 0;
        yuv_buffer_recon[i] = p;
    }

    FILE* pFile;
    pFile = fopen("../python_kdw/resources/C01_BasketballDrill_832x480_50_QP37_traced.yuv", "wb");
    fwrite(yuv_buffer_recon, sizeof(unsigned char), width*height, pFile);
    fclose(pFile);


    return 0;
}

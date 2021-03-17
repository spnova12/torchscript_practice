#include <iostream>
#include <string>
#include <torch/script.h>


int main() {

    // 읽을 비디오
    int width = 832;
    int height = 480;
    const char* yuv_dir = "../python_kdw/resources/C01_BasketballDrill_832x480_50_QP37.yuv";


    // 읽을 데이터를 저장할 버퍼를 만든다.
    int buffer_size = int(width * height *1.5);
    auto* yuv_buffer = new unsigned char[buffer_size];


    // yuv420 binary file 읽기.
    FILE* fp;
    fp = fopen(yuv_dir, "rb");
    if(fp == nullptr)  // c++ 에서 NULL 은 nullptr 이다.
        std::cout << "buffer is NULL";


    // buffer 에 읽은 데이터를 저장해준다.
    // 10진수=%d, 8진수=%o, 16진수=%x, # 을 추가하면 접두사도 출력됨)
    fread(yuv_buffer, sizeof(unsigned char), buffer_size, fp);


    // torch::from_blob 는 unsigned char 가 들어가면 안되는 듯. 때문에 float 로 형변환 필요.
    // 아래 코드를 통해 형변환 해줌. 동적 배열을 형변환 하는 최적의 코드는 무엇일까..
    auto* yuv_buffer_float = new float[width * height];

    for (int i = 0; i < height * width; i++)
        yuv_buffer_float[i] = yuv_buffer[i];


    // 255 -> 1 (pytorch 에서 0~1 의 값으로 사용했기 때문)
    for (int i = 0; i < height * width; i++)
        yuv_buffer_float[i] = yuv_buffer_float[i] /= 255;


    // buffer 를 torch::from_blob 을 통해서 torch 의 tensor 로 옮겨줌.
    // at::Tensor is not differentiable while torch::Tensor is ..
    // (https://discuss.pytorch.org/t/difference-between-torch-tensor-and-at-tensor/35806)
    at::Tensor input_tensor = torch::from_blob(yuv_buffer_float, {1, 1, height, width});


    // create a vector of inputs. (torch script 모델을 사용하려면 이 형식으로 바꿔줘야한다)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor.to(at::kCUDA));  // cuda 사용.


    // model 을 불러온다.
    torch::jit::script::Module module;
    module = torch::jit::load("../python_kdw/resources/hevc009_RA_qp37/netG_traced.pt");


    // 모델에 forward 를 해준다.
    at::Tensor output = module.forward(inputs).toTensor();
    inputs.clear();
    auto output_tensor = output.cpu().detach();


    // 결과를 yuv_buffer_float 버퍼에 저장해준다.
    std::memcpy(yuv_buffer_float, output_tensor.data_ptr(), sizeof(float) * width * height);


    // tensor to unsigned char
    auto* yuv_buffer_recon = new unsigned char[width * height];
    for (int i = 0; i < height * width; i++){
        float p = (unsigned char)(yuv_buffer_float[i] * 255);
        if(p > 255)
            p = 255;
        if(p < 0)
            p = 0;
        yuv_buffer_recon[i] = p;
    }


    // 복원된 영상을 저장해준다.
    FILE* pFile;
    pFile = fopen("../python_kdw/resources/C01_BasketballDrill_832x480_50_QP37_traced.yuv", "wb");
    fwrite(yuv_buffer_recon, sizeof(unsigned char), width*height, pFile);
    fclose(pFile);


    return 0;
}

torchscript on python and C++
=============================

먼저 python 에 대한 실험을 해본다. grdn_forward.py 를 통해 inference 해보자.  
python 에서 모델을 inference 할때 사용되는 모든 기능들이 c++ 에도 구현되어야 한다.   

일단 grdn_forward.py 를 통해서 python 안에서의 결과는 traced 된 모델과 기존 torch 모델과 같음을 확인하였다. 

또한 GPU 또한 잘 사용 됨을 확인 함. 추론 시간도 초기 한두번만 traced 모델이 더 오래걸리고,
traced 모델과 torch 모델이 속도가 거의 같음을 확인함.    

C++ 에서 traced model 을 돌려봄. python 의 결과와 다음의 차이가 있었음.  

input_c_psnr : 34.4710069960766  
input_python_psnr : 34.470999149919116    

두 값은 0.0000078 차이가 남. 
다른 언어기 때문에 발생하는 차이라고 생각할 수 있음??

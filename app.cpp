#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
  assert(module != nullptr);
  std::cout << "ok\n";
  int l;
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::randint(20, {10,1}, torch::dtype(torch::kInt64)));
  l = inputs.size();
  for(int i = 0; i < l; i++){
    std::cout << inputs[i] << '\n';
  }
  at::Tensor output = module->forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  int data[] = { 1, 2, 3,
                 4, 5, 6, 
                 17, 8, 12,
                 0 };
  std::vector<torch::jit::IValue> test_ip;
  torch::Tensor t = torch::zeros({10, 1}, torch::dtype(torch::kInt64));
  for(int i = 0; i < 10; i++){
    t[i] = data[i];
  }
  test_ip.push_back(t);
  l = test_ip.size() ;
  // auto foo_a = test_ip.accessor<int,1>();

  
  for(int i = 0; i < l; i++){
    std::cout << test_ip[i] << '\n';
  }
  at::Tensor ouut = module->forward(test_ip).toTensor();
  std::cout << ouut.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  // std::cout << test_ip << '\n';
}
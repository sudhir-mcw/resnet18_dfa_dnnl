# DNNL ResNet18 Inference with DFA implementation
The repo contains the Independent C++ inference with ARM compute Library for [ResNet 18 onnx model](https://huggingface.co/frgfm/resnet18/blob/main/model.onnx). The Python inference is used for validation purpose.
## Machine Requirements:
- Processor Architecture: ARM64
- RAM: Minimum 8GB
- OS: Ubuntu 20.04 
- Storage: Minimum 64GB
# Prequisites
* G++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
* cmake version 3.29.3
* GNU Make 4.2.1
* [cnpy](https://github.com/rogersce/cnpy) 
* [DNNL](https://github.com/oneapi-src/oneDNN)
* Python 3.8.10 
# Install Prequisites
1. Build the DNNL library by referring to the [documentation](https://oneapi-src.github.io/oneDNN/dev_guide_build.html)
2. Build the cnpy library by following the steps in [documentation](https://github.com/rogersce/cnpy?tab=readme-ov-file#installation)  
# Cloning the repo
Use the command below to clone the repo
```
    git clone https://github.com/sudhir-mcw/resnet18_dfa_dnnl.git
    cd resnet18_dfa_dnnl
```
Update CMake Configuration on successful prequisite installation
Open the CMakeLists.txt file in the root of the project directory and update the following
* PATH_TO_oneDNN  - Replace the <PATH_TO_oneDNN> with path to oneDNN source folder
* PATH_TO_CNPY     - Replace the <PATH_TO_CNPY> with the CNPY source folder's path  


# How to Run Python Inference (Used for Validation of C++ Inference Output)
1. Navigate to the project directory
```
    cd resnet18_dfa_dnnl
```
2. Download input image 
```
    mkdir input 
    wget -O input/chainsaw.png https://stihlusa-images.imgix.net/Category/41/Teaser.png
```
3. Download the ResNet18 onnx model from hugging face repo 
```
    wget https://huggingface.co/frgfm/resnet18/resolve/main/model.onnx
```
4. Install Required python packages
```
    pip install -r requirements.txt
```
5. Run the python script to dump weights from the onnx model 
```
    python model_dumper.py
```  
6. Run the python inference script to load image, preprocess and dump output files
```
    python inference.py <input_image_path>
```
Sample Usage
```
    python inference.py input/chainsaw.png
```
# How to Run C++ Inference 
Note: Execute the Python inference before running C++ Inference
1. Navigate to the project directory
```
    cd resnet18_dfa_dnnl
```
2. Build the cpp inference program
```
    cmake -B build -S .
    make -C build 
``` 
3. Run the program 
```
    ./build/inference
```

# Comparing Outputs
1. All the output files are stored in outputs/ folder, Manual comparison of files can be done using the compare.py file 
```
    python compare.py <file_1.npy> <file_2.npy>
```
Sample usage
```
    python compare.py outputs/cpp_conv1_merged.npy outputs/py_output_conv1.npy
```
Sample output 
```
    $  python compare.py outputs/cpp_conv1_merged.npy outputs/py_output_conv1.npy
    Files are identical upto 4 decimals
```

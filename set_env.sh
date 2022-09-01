source /home/ubw/dotfiles/scripts/wwswcuda 11.0
export CUDA_HOME="/usr/local/cuda-11.0"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/efa/lib:$LD_LIBRARY_PATH"
nvcc -V




LIB_NAME="libsum.so"
CUDA_FILE="Sum.cu"
WRAPPER_FILE=""

rm -f $LIB_NAME $GO_EXEC

echo "Compiling CUDA and wrapper files..."

nvcc -Xcompiler -fPIC -o $LIB_NAME --shared $CUDA_FILE $WRAPPER_FILE

if [ $? -ne 0 ]; then
    echo "Error: Failed to compile CUDA and wrapper files."
    exit 1
fi
echo "Successfully compiled CUDA and wrapper files."



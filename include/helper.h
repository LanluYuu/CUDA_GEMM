#pragma once
#include <stdexcept>
#include <random>
#include <cstring>
#include <cstdint>
#include <iostream>

#define BLOCK_SIZE 4096
#define ELE_IDX(x, y, col) (x * col + y)
#define DELTA 1e-1
#define WARMUPT 10
#define RUNTIMES 100

template<typename T>
struct Matrix {
    int32_t height;
    int32_t width;
    T* data; 

    Matrix(int32_t h, int32_t w) : height(h), width(w) {
        data = new T[h*w];
        if (!data) {
            throw std::runtime_error("\nhost data allocate fail!\n");
        }
    }

    ~Matrix() {
        delete[] data;
    }
};

//host function
template<typename T>
float Get_Matrix_Element(const Matrix<T>& X, int32_t row, int32_t col) {
    if (row < X.height && col < X.width) {
        return X.data[row * X.width + col];
    } else {
        std::cout<<"get matrix element unvalid index!"<<std::endl;
    }

    return 0.0f;
}

template<typename T>
void Set_Matrix_Element(Matrix<T>& X, int32_t row, int32_t col, float val) {
    size_t size = sizeof(T);
    if (size == 2) {
        if (row < X.height && col < X.width) {
        X.data[row * X.width + col] = __float2half(val);
        } else {
            std::cout<<"set matrix element unvalid index!"<<std::endl;
        }
    } else if(size == 4) {
        if (row < X.height && col < X.width) {
            X.data[row * X.width + col] = val;
        } else {
            std::cout<<"set matrix element unvalid index!"<<std::endl;
        }
    }

    return;
}

/*float GenRandomVal(float min, float max) {
    std::random_device rd; //random seed
    std::mt19937 gen(rd()); //Mersenne Twister random generator
    std::uniform_real_distribution<float> dis(min, max);

    return dis(gen); 
}*/
template<typename T>
bool GenRdVal4Mat(Matrix<T>& X) {
    if (!X.data) {
        throw std::runtime_error("matrix X is not allocated!\n");
        return false;
    }
    std::random_device rd; //random seed
    std::mt19937 gen(rd()); //Mersenne Twister random generator
    std::uniform_real_distribution<float> dis(0, 1);

    for (int32_t i = 0; i < X.height; ++i) {
        for(int32_t j = 0; j < X.width; ++j) {
            Set_Matrix_Element(X, i, j, dis(gen));
            //Set_Matrix_Element(X, i, j, 1.0); //for debug
        }
    }
    return true;
}

template<typename T>
void ComputeGolden(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    memset(C.data, 0, C.height * C.width * sizeof(T));
    for (int32_t i = 0; i < A.width; ++i) {
        for (int32_t row = 0; row < A.height; ++row) {
            for (int32_t col = 0; col < B.width; ++col) {
                C.data[ELE_IDX(row, col, C.width)] += A.data[ELE_IDX(row, i, A.width)] * B.data[ELE_IDX(i, col, B.width)];
            }
        }
    }

    return;
}

template<typename T>
bool CompareMat(const Matrix<T>& A, const Matrix<T>& B) {
    bool res = true;
    bool isHalf;
    if (sizeof(A.data[0]) == 2) {
        isHalf = true;
    } else if(sizeof(A.data[0]) == 4) {
        isHalf = false;
    } else {
        printf("Compare Mat DataType not allowed!\n");
        return false;
    }

    int32_t row = A.height;
    int32_t col = A.width;
    int32_t miss_num = 0;
    float sum_err = 0.0f;    
    for (int32_t i = 0; i < row; ++i) {
        for(int32_t j = 0; j < col; ++j) {
            float A_tmp;
            float B_tmp;
            if (isHalf) {
                A_tmp = __half2float(A.data[ELE_IDX(i, j, col)]);
                B_tmp = __half2float(B.data[ELE_IDX(i, j, col)]);
            } else {
                A_tmp = A.data[ELE_IDX(i, j, col)];
                B_tmp = B.data[ELE_IDX(i, j, col)];
            }
            float err = abs(A_tmp - B_tmp);
            sum_err  += err / A_tmp;
            if(err > DELTA) {
                res = false;
                miss_num += 1;
                //std::cout <<"\nMismatch, row:" << i << ", col: " << j << ", expected:" << B.data[ELE_IDX(i, j, col)] << ", got:" << A.data[ELE_IDX(i, j, col)];
            }
        }
    }
    //std::cout << "\ndelta:" << DELTA << std::endl;
    std::cout << "\ntotal mismatch:" << miss_num << std::endl;
    std::cout << "\naverage error:" << sum_err / (row * col) << std::endl;
    return res;
}

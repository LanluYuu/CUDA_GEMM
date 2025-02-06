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

struct Matrix {
    int32_t height;
    int32_t width;
    float* data; 

    Matrix(int32_t h, int32_t w) : height(h), width(w) {
        data = new float[h*w];
        if (!data) {
            throw std::runtime_error("\nhost data allocate fail!\n");
        }
    }

    ~Matrix() {
        delete[] data;
    }
};

//host function

float Get_Matrix_Element(const Matrix& X, int32_t row, int32_t col) {
    if (row < X.height && col < X.width) {
        return X.data[row * X.width + col];
    } else {
        std::cout<<"get matrix element unvalid index!"<<std::endl;
    }

    return 0.0f;
}

void Set_Matrix_Element(Matrix& X, int32_t row, int32_t col, float val) {
    if (row < X.height && col < X.width) {
        X.data[row * X.width + col] = val;
    } else {
        std::cout<<"set matrix element unvalid index!"<<std::endl;
    }

    return;
}

/*float GenRandomVal(float min, float max) {
    std::random_device rd; //random seed
    std::mt19937 gen(rd()); //Mersenne Twister random generator
    std::uniform_real_distribution<float> dis(min, max);

    return dis(gen); 
}*/

bool GenRdVal4Mat(Matrix& X) {
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

void ComputeGolden(const Matrix& A, const Matrix& B, Matrix& C) {
    memset(C.data, 0, C.height * C.width * sizeof(float));
    for (int32_t i = 0; i < A.width; ++i) {
        for (int32_t row = 0; row < A.height; ++row) {
            for (int32_t col = 0; col < B.width; ++col) {
                C.data[ELE_IDX(row, col, C.width)] += A.data[ELE_IDX(row, i, A.width)] * B.data[ELE_IDX(i, col, B.width)];
            }
        }
    }

    return;
}

bool CompareMat(const Matrix& A, const Matrix& B) {
    bool res = true;
    int32_t row = A.height;
    int32_t col = A.width;
    int32_t miss_num = 0;
    float sum_err = 0.0f;    
    for (int32_t i = 0; i < row; ++i) {
        for(int32_t j = 0; j < col; ++j) {
            float err = abs(A.data[ELE_IDX(i, j, col)] - B.data[ELE_IDX(i, j, col)]);
            sum_err  += err / A.data[ELE_IDX(i, j, col)];
            if((abs(A.data[ELE_IDX(i, j, col)] - B.data[ELE_IDX(i, j, col)]) > DELTA)) {
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

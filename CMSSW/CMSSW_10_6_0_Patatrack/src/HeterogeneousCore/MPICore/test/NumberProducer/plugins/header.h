
#pragma once


void LOG(std::string message, int t);

void cudaWrapper(double* buf_d, double* sum_h, int len);
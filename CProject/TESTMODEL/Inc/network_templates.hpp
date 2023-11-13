/*
 * network.hpp
 *
 *  Created on: Sep 20, 2023
 *      Author: Daniel Schnoell
 */

#ifndef SRC_NETWORK_TEMPLATES_HPP_
#define SRC_NETWORK_TEMPLATES_HPP_

#include "stm32f3xx_hal.h"

__STATIC_FORCEINLINE uint32_t __SXTB16_ROR8(uint32_t op1)
{
	uint32_t result;

	__ASM volatile("sxtb16 %0, %1, ror #8" : "=r"(result) : "r"(op1));
	return (result);
}

template <
	int16_t M_1,
	int16_t M_2,
	typename Type = float>
struct Matrix
{
	Type data[M_1][M_2];
};

union SIMD8
{
	struct
	{
		int8_t d1;
		int8_t d2;
		int8_t d3;
		int8_t d4;
	} true_data;
	uint32_t simd;
};
union SIMD16
{
	struct
	{
		int16_t d1;
		int16_t d2;
	} true_data;
	uint32_t simd;
};
union SIMD32
{
	int32_t true_data;
	uint32_t simd;
};

template <
	typename Type_Out,
	int16_t M1_1,
	int16_t M1_2,
	int16_t M2_1,
	int16_t M2_2,
	typename Lambda>
Matrix<M1_1, M2_1, Type_Out> Linear(const Matrix<M1_1, M1_2, int16_t> &A, const Matrix<M2_1, M2_2, int8_t> &B, const Matrix<M2_1, 1, int32_t> &bias, const Matrix<M2_1, 1, int32_t> &right_shift, Lambda act)
{
	Matrix<M1_1, M2_1, Type_Out> out;

	for (int i1_1 = 0; i1_1 < M1_1; i1_1++)
	{

		for (int i2_1 = 0; i2_1 < M2_1; i2_1++)
		{
			SIMD32 sum{{bias.data[i2_1][0]}};
			int i1_2 = 0;

			for (; i1_2 < M1_2 - 3; i1_2 += 4)
			{
				SIMD8 w{.true_data = {	//Weights
							B.data[i2_1][i1_2],
							B.data[i2_1][i1_2 + 1],
							B.data[i2_1][i1_2 + 2],
							B.data[i2_1][i1_2 + 3],
						}};
				SIMD16 in1{.true_data = { //First Input
							   A.data[i1_1][i1_2],
							   A.data[i1_1][i1_2 + 1]}};
				SIMD16 in2{.true_data = { //Second Input
							   A.data[i1_1][i1_2 + 2],
							   A.data[i1_1][i1_2 + 3]}};
				SIMD16 w1{.simd = __SXTB16_ROR8(w.simd)};
				SIMD16 w2{.simd = __SXTB16(w.simd)};

				sum.simd = __SMLAD(in1.simd, w1.simd, sum.simd);
				sum.simd = __SMLAD(in2.simd, w2.simd, sum.simd);
			}
			if (i1_2 == M1_2 - 3)
			{
				SIMD8 w{.true_data = {
							B.data[i2_1][i1_2],
							B.data[i2_1][i1_2 + 1],
							B.data[i2_1][i1_2 + 2],
							B.data[i2_1][i1_2 + 3], // should be 0
						}};
				SIMD16 in1{.true_data = {A.data[i1_1][i1_2], A.data[i1_1][i1_2 + 1]}};
				SIMD16 in2{.true_data = {A.data[i1_1][i1_2 + 2], 0}};
				SIMD16 w1{.simd = __SXTB16(w.simd)};
				SIMD16 w2{.simd = __SXTB16_ROR8(w.simd)};

				sum.simd = __SMLAD(in1.simd, w1.simd, sum.simd);
				sum.simd = __SMLAD(in2.simd, w2.simd, sum.simd);
			}
			else if (i1_2 == M1_2 - 2)
			{
				sum.true_data += ((int32_t)A.data[i1_1][i1_2]) * ((int32_t)B.data[i2_1][i1_2]);
				sum.true_data += ((int32_t)A.data[i1_1][i1_2 + 1]) * ((int32_t)B.data[i2_1][i1_2 + 1]);
			}
			else if (i1_2 == M1_2 - 1)
			{
				sum.true_data += ((int32_t)A.data[i1_1][i1_2]) * ((int32_t)B.data[i2_1][i1_2]);
			}

			out.data[i1_1][i2_1] = act(sum.true_data, right_shift.data[i2_1][0]);
		}
	}
	return out;
}

template <
	int16_t stride,
	int16_t M1_1,
	int16_t M1_2,
	int16_t M2_1,
	int16_t M2_2,
	typename Lambda>
Matrix<M1_1, ((M1_2 - M2_1) / stride + 1) * M2_2, int16_t> ConvT(const Matrix<M1_1, M1_2, float> &A, const Matrix<M2_1, M2_2, float> &B, const Matrix<M2_2, 1, float> &bias, Lambda act)
{
	Matrix<M1_1, ((M1_2 - M2_1) / stride + 1) * M2_2, int16_t> out;
	// int16_t k = 0;
	int16_t k = ((M1_2 - M2_1) / stride + 1) * M2_2 - M2_2;

	// for (int16_t i = 0; i < (M1_2-M2_1)+1 ; i += stride)
	for (int16_t i = ((M1_2 - M2_1) / stride) * stride; i >= 0; i -= stride)
	{
		// Preload the bias
		// for (int16_t t = M2_2-1 ; t >= 0; t--)
		float sum[M2_2];

#pragma unroll
		for (int16_t t = 0; t < M2_2; t++)
			sum[t] = bias.data[t][0];

		// for (int16_t j = 0; j <M2_1; j++)
		//  The convolution step
		// #pragma unroll 5
		for (int16_t j = M2_1 - 1; j >= 0; j--)
		{
// for (int16_t t = M2_2-1; t >= 0; t--)
#pragma unroll
			for (int16_t t = 0; t < M2_2; t++)
				sum[t] += A.data[0][i + j] * B.data[j][t];
		}
// for (int16_t t = M2_2; t >= 0; t--)
#pragma unroll
		for (int16_t t = 0; t < M2_2; t++)
			out.data[0][k + t] = act(sum[t]);
		// k += M2_2;
		k -= M2_2;
	}
	return out;
}

#endif /* SRC_NETWORK_TEMPLATES_HPP_ */

/*
 * network.cpp
 *
 *  Created on: Sep 22, 2023
 *      Author: Daniel Schnoell
 */

#include "network.hpp"
#include "network_templates.hpp"
#include "weights_Quant.hpp"

Matrix<85, 3> Conv_weight_ram;
Matrix<20, 112, int8_t> fc2_weight_ram;

void very_ugly_Linear(const int16_t *A, const int8_t *B, const int32_t *bias, const int32_t *right_shift, void (*act)(int32_t, int32_t, void *), uint32_t size_of_datatype, void *out, const int32_t M1_2, const int32_t M2_1, const int32_t M2_2);

void aiRun(const float input[1][1250][1], float result[2])
{

	const Matrix<1, 1250> *IN = (Matrix<1, 1250> *)(void *)input;

	const auto I1 = ConvT<32>(*IN, Conv_weight_ram, Conv_bias, onnxConv_ACT);

	// It could have been so nice, but alas lower points
	// const auto I2 = Linear<int16_t>(I1,fc2_weight_ram,fc2_bias,fc2_right_shift,fc2_ACT);
	// const auto I3 = Linear<int32_t>(I2,fc3_weight,fc3_bias,fc3_right_shift,fc3_ACT);

	Matrix<1, 20, int16_t> I2;
	very_ugly_Linear(&I1.data[0][0], &fc2_weight_ram.data[0][0], &fc2_bias.data[0][0], &fc2_right_shift.data[0][0], &fc2_ACT_garbage, sizeof(int16_t), &I2.data[0][0], 111, 20, 112);

	Matrix<1, 2, int32_t> I3;
	very_ugly_Linear(&I2.data[0][0], &fc3_weight.data[0][0], &fc3_bias.data[0][0], &fc3_right_shift.data[0][0], &fc3_ACT_garbage, sizeof(int32_t), &I3.data[0][0], 20, 2, 20);

	result[0] = float_conversion[0] * (float)I3.data[0][0];
	result[1] = float_conversion[1] * (float)I3.data[0][1];
}

void Model_Init()
{
	int8_t *addr = &fc2_weight_ram.data[0][0];
	int32_t p = 0;

#pragma nounroll
	for (int32_t q = 0; q < 20 * 112; q += 2)
	{
		uint8_t word = lower_half[p];
		addr[q] = (0xF0 & word) >> 4;
		addr[q + 1] = 0x0F & word;
		p++;
	}

	int32_t q = 0;
	uint32_t field = 0;
	uint32_t current_stream = 0;
	unsigned char bit = 0;

#pragma nounroll
	for (int32_t index = 0; index < 200 * 32; index++)
	{
		current_stream = current_stream << 1;
		if (!(index & 0x001F))
		{
			current_stream = upper_half[index >> 5];
		}

		field = (field << 1) | bit;
		bit = current_stream >> 31;

		if (bit == 0)
		{
			uint32_t key = 32 - __CLZ(field);
			field = 0;
			uint8_t value{decoder[key]};
			addr[q] |= value;
			q++;
			if (q >= 20 * 112)
				break;
		}
	}

#pragma nounroll
	for (int32_t i = 0; i < 85 * 3; i++)
	{
		*(&Conv_weight_ram.data[0][0] + i) = *(&Conv_weight.data[0][0] + i);
	}

	// for (int32_t i = 0; i <20*112; i++)
	// {
	// 	{
	// 		*(&fc2_weight_ram.data[0][0]+i) = *(&fc2_weight.data[0][0]+i);
	// 	}
	// }	
}

void very_ugly_Linear(const int16_t *A, const int8_t *B, const int32_t *bias, const int32_t *right_shift, void (*act)(int32_t, int32_t, void *), uint32_t size_of_datatype, void *out, const int32_t M1_2, const int32_t M2_1, const int32_t M2_2)
{
#pragma nounroll
	for (int i2_1 = 0; i2_1 < M2_1; i2_1++)
	{
		SIMD32 sum{bias[i2_1]};
		int i1_2 = 0;
#pragma nounroll
		for (; i1_2 < M1_2 - 3; i1_2 += 4)
		{
			SIMD8 w{.true_data = {
						B[i2_1 * M2_2 + i1_2],
						B[i2_1 * M2_2 + i1_2 + 1],
						B[i2_1 * M2_2 + i1_2 + 2],
						B[i2_1 * M2_2 + i1_2 + 3],
					}};
			SIMD16 in1{.true_data = {A[i1_2], A[i1_2 + 1]}};
			SIMD16 in2{.true_data = {A[i1_2 + 2], A[i1_2 + 3]}};
			SIMD16 w1{.simd = __SXTB16(w.simd)};
			SIMD16 w2{.simd = __SXTB16_ROR8(w.simd)};

			sum.simd = __SMLAD(in1.simd, w1.simd, sum.simd);
			sum.simd = __SMLAD(in2.simd, w2.simd, sum.simd);
		}
		if (i1_2 == M1_2 - 3)
		{
			SIMD8 w{.true_data = {
						B[i2_1 * M2_2 + i1_2],
						B[i2_1 * M2_2 + i1_2 + 1],
						B[i2_1 * M2_2 + i1_2 + 2],
						B[i2_1 * M2_2 + i1_2 + 3], // should be 0
					}};
			SIMD16 in1{.true_data = {A[i1_2], A[i1_2 + 1]}};
			SIMD16 in2{.true_data = {A[i1_2 + 2], 0}};
			SIMD16 w1{.simd = __SXTB16(w.simd)};
			SIMD16 w2{.simd = __SXTB16_ROR8(w.simd)};

			sum.simd = __SMLAD(in1.simd, w1.simd, sum.simd);
			sum.simd = __SMLAD(in2.simd, w2.simd, sum.simd);
		}
		// They are never reached, so they are thrown out. It's a very bad practice.
		/*else if (i1_2 == M1_2 - 2)
		{
			sum.true_data += ((int32_t)A[i1_2]) * ((int32_t)B[i2_1 * M2_2 + i1_2]);
			sum.true_data += ((int32_t)A[ i1_2 + 1]) * ((int32_t)B[i2_1 * M2_2 + i1_2 + 1]);
		}
		else if (i1_2 == M1_2 - 1)
		{
			sum.true_data += ((int32_t)A[i1_2]) * ((int32_t)B[i2_1 * M2_2 + i1_2]);
		}*/

		// Disregarding best practices, we're incrementing a void pointer and 
		// crossing our fingers it's big enough. After all, a little typo in 
		// the previous code couldn't possibly make any problems. 
		// At the end of the day compile-time array bound checking isn't useful
		// at all.
		act(sum.true_data, right_shift[i2_1], ((char *)out + size_of_datatype * (i2_1)));
	}
}
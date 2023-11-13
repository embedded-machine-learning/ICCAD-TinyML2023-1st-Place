/*
 * network.hpp
 *
 *  Created on: Sep 22, 2023
 *      Author: dschnoell
 */

#ifndef INC_NETWORK_HPP_
#define INC_NETWORK_HPP_



#ifdef __cplusplus
extern "C" {
#endif


void aiRun(const float input[1][1250][1], float result[2]);
void Model_Init();


#ifdef __cplusplus
}
#endif


#endif /* INC_NETWORK_HPP_ */

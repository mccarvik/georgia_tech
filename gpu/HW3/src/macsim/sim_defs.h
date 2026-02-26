#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include<cstdint>
#include<string>
#include<iostream>

typedef uint64_t sim_time_type;
typedef uint16_t stream_id_type;
typedef sim_time_type data_timestamp_type;

#define INVALID_TIME 18446744073709551615ULL
#define T0 0
#define INVALID_TIME_STAMP 18446744073709551615ULL
#define MAXIMUM_TIME 18446744073709551615ULL
#define ONE_SECOND 1000000000
typedef std::string sim_object_id_type;

#define CurrentTimeStamp Simulator->Time()
#define PRINT_ERROR(MSG) {\
							std::cerr << "ERROR:" ;\
							std::cerr << MSG << std::endl; \
							std::cin.get();\
							exit(1);\
						 }
#define PRINT_MESSAGE(M) std::cout << M << std::endl;
#define DEBUG(M) std::cout<<M<<std::endl;
#define DEBUG2(M) std::cout<<M<<std::endl;
#define MA_DEBUG(M) std::cout<<M<<std::endl;
#define MA_DEBUG2(M) std::cout<<M<<std::endl;
#define SIM_TIME_TO_MICROSECONDS_COEFF 1000
#define SIM_TIME_TO_SECONDS_COEFF 1000000000

// Uncomment to the following to enable logging of warp scheduling activity
// #define LOG_WARP_SCHEDULING
// #define LOG_CCWS_WARP_SCHEDULING

#ifdef LOG_WARP_SCHEDULING
  #define WSLOG(x) x
#else
  #define WSLOG(x)
#endif

#ifdef LOG_CCWS_WARP_SCHEDULING
  #define CCWSLOG(x) x
#else
  #define CCWSLOG(x)
#endif

// Renames
// Try to use these rather than built-in C types in order to preserve portability
typedef unsigned uns;
typedef unsigned char uns8;
typedef unsigned short uns16;
typedef unsigned uns32;
typedef unsigned long long uns64;
typedef char int8;
typedef short int16;
typedef int int32;
typedef int long long int64;
typedef int Generic_Enum;
typedef uns64 Counter;
typedef int64 Quad;
typedef uns64 UQuad;

/* power & hotleakage  */ /* please CHECKME Hyesoon 6-15-2009 */
typedef uns64 tick_t;
typedef uns64 counter_t;

/* Conventions */
typedef uns64 Addr;
typedef uns32 Binary;

#endif // !DEFINITIONS_H
cmake_minimum_required(VERSION 2.8.3)
project(pcnml)


find_package(PCL REQUIRED)
find_package(OpenCV 3.2.0 REQUIRED EXACT)
find_package(Eigen3 3.0 REQUIRED)

#find_package(glibc REQUIRED)




#find_package(blaze REQUIRED)
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)
#list(APPEND LINALG_DEFINITION USE_BLAZE)
#link_libraries(blaze::blaze ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

message('here erer :'${BLAZE_INCLUDE_DIRS})



set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mfma -msse2 -msse3 -msse4")
add_definitions(-std=c++17)
add_definitions(-mfma)
add_definitions(-mno-avx512f)
add_definitions(-mno-avx512vl)
add_definitions(-DUSE_SSE_AUTO)
add_definitions(-DSSE_MATHFUN_WITH_CODE)
set(CPU_ALL_OPTIMIZATIONS "SSE;SSE2;SSE3;SSSE3;SSE4_1;SSE4_2;POPCNT;AVX;FP16;AVX2;FMA3;AVX_512F;AVX512VL;AVX512F;AVX_512VL")
#SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq -march=knl -march=sandybridge -march=haswell")
message('0-0000000000000000000' ${AVX_FLAGS})

#-march=native
ADD_DEFINITIONS(-DEIGEN_ENABLE_AVX512=ON)
#ADD_DEFINITIONS(-DMEDIAN_METHOD)
#ADD_DEFINITIONS(-DMEAN_METHOD)
add_definitions(${PCL_DEFINITIONS})

# message( -----------${PCL_DEFINITIONS}-----------)

message(----${BLAZE_INCLUDE_DIRS})

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
        ${PROJECT_NAME}
        include
        include/ThirdParty
        include/ThirdParty/linasm
        ${PCL_INCLUDE_DIRS}
)

#file(GLOB all_source  *.cpp  )

#add_library()
#LINK_DIRECTORIES(/home/bohuan/code/linasm)

add_executable(test
        test/test.cpp
        include/ThirdParty/cvrgbd/src/depth_cleaner.cpp
        include/ThirdParty/cvrgbd/src/depth_registration.cpp
        include/ThirdParty/cvrgbd/src/depth_to_3d.cpp
        include/ThirdParty/cvrgbd/src/linemod.cpp
        include/ThirdParty/cvrgbd/src/normal.cpp
        include/ThirdParty/cvrgbd/src/odometry.cpp
        include/ThirdParty/cvrgbd/src/plane.cpp
        include/ThirdParty/cvrgbd/src/depth_cleaner.cpp
        include/ThirdParty/cvrgbd/src/utils.cpp
        include/pcnml/ThirdAlgorithm/ThirdAlgorithm.cpp
        )
#very important!!
#otherwise in some machine,it will be built with c++11, you will find some error!!!!
set_property(TARGET test PROPERTY CXX_STANDARD 17)


target_link_libraries(test
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
       # /home/bohuan/code/linasm/liblinasm.so
        )

add_executable(example
        test/example.cpp
        include/ThirdParty/cvrgbd/src/depth_cleaner.cpp
        include/ThirdParty/cvrgbd/src/depth_registration.cpp
        include/ThirdParty/cvrgbd/src/depth_to_3d.cpp
        include/ThirdParty/cvrgbd/src/linemod.cpp
        include/ThirdParty/cvrgbd/src/normal.cpp
        include/ThirdParty/cvrgbd/src/odometry.cpp
        include/ThirdParty/cvrgbd/src/plane.cpp
        include/ThirdParty/cvrgbd/src/depth_cleaner.cpp
        include/ThirdParty/cvrgbd/src/utils.cpp
        include/pcnml/ThirdAlgorithm/ThirdAlgorithm.cpp
        )
#very important!!
#otherwise in some machine,it will be built with c++11, you will find some error!!!!
set_property(TARGET test PROPERTY CXX_STANDARD 17)


target_link_libraries(example
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        # /home/bohuan/code/linasm/liblinasm.so
        )

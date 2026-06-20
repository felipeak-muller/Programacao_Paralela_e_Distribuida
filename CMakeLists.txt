cmake_minimum_required(VERSION 3.10)
project(Programacao_Paralela_e_Distribuida)

# Exigir C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Procurar pelos pacotes necessários
find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Criar o executável a partir do main.cpp
add_executable(main main.cpp)

# Linkar as bibliotecas do OpenCV, MPI e OpenMP ao executável
target_link_libraries(main PRIVATE 
    ${OpenCV_LIBS} 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
)
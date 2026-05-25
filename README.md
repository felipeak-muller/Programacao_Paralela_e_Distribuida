# Programacao_Paralela_e_Distribuida

Trabalho em Grupo utilizando OpenMP, MPI e OpenCV.

## Dependencias

```bash
sudo apt install libopencv-dev
```

Para autocomplete:

```bash
sudo apt install clangd
```

## Build

```bash
make
```

## Executar

```bash
make run
# ou
./main
```

O programa le `sample.mp4` e salva os frames em `frames/`.

## Limpar

```bash
make clean
```

## Build alternativo (CMake)

```bash
sudo apt install cmake
mkdir -p build && cd build
cmake ..
cmake --build .
./main
```

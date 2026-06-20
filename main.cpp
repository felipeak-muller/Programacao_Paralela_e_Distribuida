#include <algorithm> // Para usar o std::clamp
#include <cmath>     // Para usar o std::sqrt no Sobel
#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <omp.h>
#include <map>

#define MAX_THREADS 8


// 1. GAUSSIANO (Redução de ruído / Desfoque)

void apply_gaussian(const cv::Mat& src, cv::Mat& dst, int start_row, int end_row) {
    int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    int kernel_weight = 16; 
    int start = std::max(1, start_row);
    int end = std::min(src.rows - 1, end_row);
    for (int r = start; r < end; r++) {
        for (int c = 1; c < src.cols - 1; c++) {
            for (int ch = 0; ch < 3; ch++) { 
                int sum = 0;
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        int pixel_val = src.at<cv::Vec3b>(r + kr, c + kc)[ch];
                        sum += pixel_val * kernel[kr + 1][kc + 1];
                    }
                }
                dst.at<cv::Vec3b>(r, c)[ch] = std::clamp(sum / kernel_weight, 0, 255);
            }
        }
    }
}


// 2. SHARPEN (Nitidez)

void apply_sharpen(const cv::Mat& src, cv::Mat& dst, int start_row, int end_row) {
    int kernel[3][3] = {{ 0, -1,  0}, {-1,  5, -1}, { 0, -1,  0}};
    int start = std::max(1, start_row);
    int end = std::min(src.rows - 1, end_row);
    for (int r = start; r < end; r++) {
        for (int c = 1; c < src.cols - 1; c++) {
            for (int ch = 0; ch < 3; ch++) {
                int sum = 0;
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        int pixel_val = src.at<cv::Vec3b>(r + kr, c + kc)[ch];
                        sum += pixel_val * kernel[kr + 1][kc + 1];
                    }
                }
                dst.at<cv::Vec3b>(r, c)[ch] = std::clamp(sum, 0, 255); 
            }
        }
    }
}


// 3. SOBEL (Detecção de Bordas)

void apply_sobel(const cv::Mat& src, cv::Mat& dst, int start_row, int end_row) {
    int Kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Ky[3][3] = {{-1, -2, -1}, { 0, 0, 0}, { 1, 2, 1}};
    int start = std::max(1, start_row);
    int end = std::min(src.rows - 1, end_row);
    for (int r = start; r < end; r++) {
        for (int c = 1; c < src.cols - 1; c++) {
            for (int ch = 0; ch < 3; ch++) {
                int sum_x = 0;
                int sum_y = 0;
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        int pixel_val = src.at<cv::Vec3b>(r + kr, c + kc)[ch];
                        sum_x += pixel_val * Kx[kr + 1][kc + 1];
                        sum_y += pixel_val * Ky[kr + 1][kc + 1];
                    }
                }
                int magnitude = std::sqrt(sum_x * sum_x + sum_y * sum_y);
                dst.at<cv::Vec3b>(r, c)[ch] = std::clamp(magnitude, 0, 255);
            }
        }
    }
}

int main(int argc, char** argv) {
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (numprocs < 2) {
        if (myid == 0) std::cerr << "Execute com pelo menos 2 processos MPI.\n";
        MPI_Finalize();
        return 1;
    }

    // O programa inteiro agora roda dentro de um grande loop interativo
    while (true) {
        int opcao = -1;

        // Apenas o MASTER imprime o menu e lê o teclado
        if (myid == 0) {
            std::cout << "\n==================================\n";
            std::cout << "    PROCESSADOR DE VÍDEO MPI\n";
            std::cout << "==================================\n";
            std::cout << "1. Aplicar Filtro Gaussiano\n";
            std::cout << "2. Aplicar Filtro Sharpen\n";
            std::cout << "3. Aplicar Filtro Sobel\n";
            std::cout << "4. Excluir vídeo (resultado.mp4)\n";
            std::cout << "0. Sair do programa\n";
            std::cout << "Escolha uma opção: ";
            std::cin >> opcao;

            // Opção 4: Excluir o arquivo
            if (opcao == 4) {
                if (std::filesystem::remove("resultado.mp4")) {
                    std::cout << "[SUCESSO] Arquivo 'resultado.mp4' foi excluido.\n";
                } else {
                    std::cout << "[AVISO] Arquivo 'resultado.mp4' nao encontrado.\n";
                }
            } else if (opcao < 0 || opcao > 4) {
                std::cout << "[ERRO] Opcao invalida.\n";
            }
        }

        // O Master avisa (Bcast) a opção escolhida para todos os Trabalhadores
        MPI_Bcast(&opcao, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Se for 0, quebra o loop principal e encerra o MPI para todo mundo
        if (opcao == 0) break;
        
        // Se a opção foi excluir ou foi inválida, volta para o topo do menu
        if (opcao == 4 || opcao < 0 || opcao > 4) continue;


        // =================================
        //  INÍCIO DO PROCESSAMENTO DE VÍDEO
        // =================================
        if (myid == 0) {
            // PROCESSO MASTER
            const std::string video_path = "sample.mp4";
            cv::VideoCapture cap(video_path);
            
            if (!cap.isOpened()) {
                std::cerr << "Erro ao abrir o video: " << video_path << "\n";
                continue; // Volta pro menu ao invés de derrubar o programa
            }

            int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            int fps = cap.get(cv::CAP_PROP_FPS);
            
            cv::VideoWriter writer("resultado.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
            
            std::cout << "\nIniciando processamento...\n";

            std::map<int, cv::Mat> frame_buffer; 
            int next_frame_to_write = 0;
            int sent_frames = 0;
            bool video_done = false;
            int workers_alive = numprocs - 1; 

            // Loop principal de distribuição do Master
            while (workers_alive > 0) {
                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                if (status.MPI_TAG == 0) {
                    int id_livre;
                    MPI_Recv(&id_livre, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    cv::Mat current_frame;
                    if (!video_done && cap.read(current_frame)) {
                        int metadados[5] = {current_frame.rows, current_frame.cols, current_frame.type(), (int)(current_frame.total() * current_frame.elemSize()), sent_frames++};
                        MPI_Send(metadados, 5, MPI_INT, id_livre, 1, MPI_COMM_WORLD);
                        MPI_Send(current_frame.data, metadados[3], MPI_BYTE, id_livre, 2, MPI_COMM_WORLD);
                    } else {
                        video_done = true;
                        int end_meta[5] = {-1, 0, 0, 0, 0};
                        MPI_Send(end_meta, 5, MPI_INT, id_livre, 1, MPI_COMM_WORLD);
                        workers_alive--; 
                    }
                } 
                else if (status.MPI_TAG == 3) {
                    int meta[5];
                    MPI_Recv(meta, 5, MPI_INT, status.MPI_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    std::vector<uchar> buf(meta[3]);
                    MPI_Recv(buf.data(), meta[3], MPI_BYTE, status.MPI_SOURCE, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    cv::Mat p_frame(meta[0], meta[1], meta[2], buf.data());
                    frame_buffer[meta[4]] = p_frame.clone(); 
                    
                    while(frame_buffer.count(next_frame_to_write)) {
                        writer.write(frame_buffer[next_frame_to_write]);
                        frame_buffer.erase(next_frame_to_write); 
                        next_frame_to_write++;
                        
                        if (next_frame_to_write % 100 == 0) {
                            std::cout << "Gravados " << next_frame_to_write << " frames no video...\n";
                        }
                    }
                }
            }

            cap.release();
            writer.release();
            std::cout << "\nConcluido! Video salvo com sucesso ('resultado.mp4').\n";

        } else {
            // PROCESSO WORKER
            while (true) {
                // Avisa o Master que está livre (TAG 0)
                MPI_Send(&myid, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

                // Recebe os metadados do frame (TAG 1)
                int metadados[5];
                MPI_Recv(metadados, 5, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int linhas = metadados[0];
                if (linhas == -1) break; // O vídeo acabou! Quebra esse loop e volta pro Menu principal

                int colunas = metadados[1];
                int tipo = metadados[2];
                int total_bytes = metadados[3];
                int frame_idx = metadados[4];

                std::vector<uchar> buffer(total_bytes);
                MPI_Recv(buffer.data(), total_bytes, MPI_BYTE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                cv::Mat frame(linhas, colunas, tipo, buffer.data());
                cv::Mat frame_processado = cv::Mat::zeros(frame.size(), frame.type());

                int linhas_locais = linhas / MAX_THREADS;

                #pragma omp parallel num_threads(MAX_THREADS)
                {
                    int thread_id = omp_get_thread_num();
                    int start_row = thread_id * linhas_locais;
                    int end_row = (thread_id == MAX_THREADS - 1) ? linhas : start_row + linhas_locais;

                    // Aplica o filtro de acordo com a opção que o usuário digitou no menu!
                    if (opcao == 1) {
                        apply_gaussian(frame, frame_processado, start_row, end_row);
                    } else if (opcao == 2) {
                        apply_sharpen(frame, frame_processado, start_row, end_row);
                    } else if (opcao == 3) {
                        apply_sobel(frame, frame_processado, start_row, end_row);
                    }
                }

                // Devolve para o Master (TAG 3 e 4)
                MPI_Send(metadados, 5, MPI_INT, 0, 3, MPI_COMM_WORLD);
                MPI_Send(frame_processado.data, total_bytes, MPI_BYTE, 0, 4, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
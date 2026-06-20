#include <algorithm> // Para usar o std::clamp
#include <cmath>     // Para usar o std::sqrt no Sobel
#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <omp.h>
#include <map>       // Necessário para ordenar os frames que chegarem fora de ordem

#define MAX_THREADS 8


// GAUSSIANO

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


// SHARPEN

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

// SOBEL

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

    if (myid == 0) {
        // PROCESSO MASTER
        const std::string video_path = "sample.mp4";
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Erro ao abrir o video: " << video_path << "\n";
            MPI_Finalize();
            return 1;
        }

        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int fps = cap.get(cv::CAP_PROP_FPS);
        
        // Criador do Arquivo de Vídeo (Usando formato AVI que tem maior compatibilidade nativa no Windows/WSL)
        cv::VideoWriter writer("resultado.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
        
        std::cout << "Iniciando processamento. Gravando em 'resultado.avi'...\n";

        std::map<int, cv::Mat> frame_buffer; // Guarda os frames que chegarem fora de ordem
        int next_frame_to_write = 0;
        int sent_frames = 0;
        bool video_done = false;
        int workers_alive = numprocs - 1; // Controla quantos trabalhadores ainda estão operando

        // Loop principal do Master
        while (workers_alive > 0) {
            MPI_Status status;
            // MPI_Probe espreita a rede para ver quem está mandando mensagem
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == 0) {
                // TAG 0 significa: O trabalhador está livre pedindo um frame novo
                int id_livre;
                MPI_Recv(&id_livre, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                cv::Mat current_frame;
                if (!video_done && cap.read(current_frame)) {
                    // Manda metadados (TAG 1) e depois os bytes da imagem original (TAG 2)
                    int metadados[5] = {current_frame.rows, current_frame.cols, current_frame.type(), (int)(current_frame.total() * current_frame.elemSize()), sent_frames++};
                    MPI_Send(metadados, 5, MPI_INT, id_livre, 1, MPI_COMM_WORLD);
                    MPI_Send(current_frame.data, metadados[3], MPI_BYTE, id_livre, 2, MPI_COMM_WORLD);
                } else {
                    // O video acabou. Mandamos o sinal de encerramento para esse trabalhador
                    video_done = true;
                    int end_meta[5] = {-1, 0, 0, 0, 0};
                    MPI_Send(end_meta, 5, MPI_INT, id_livre, 1, MPI_COMM_WORLD);
                    workers_alive--; // Esse trabalhador vai desligar
                }
            } 
            else if (status.MPI_TAG == 3) {
                // TAG 3 significa: O trabalhador processou a imagem e está me devolvendo os metadados
                int meta[5];
                MPI_Recv(meta, 5, MPI_INT, status.MPI_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // TAG 4 significa: Recebendo os bytes da imagem final
                std::vector<uchar> buf(meta[3]);
                MPI_Recv(buf.data(), meta[3], MPI_BYTE, status.MPI_SOURCE, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                cv::Mat p_frame(meta[0], meta[1], meta[2], buf.data());
                frame_buffer[meta[4]] = p_frame.clone(); // Salva no mapa (dicionário)
                
                // Tenta gravar os frames na ordem correta
                while(frame_buffer.count(next_frame_to_write)) {
                    writer.write(frame_buffer[next_frame_to_write]);
                    frame_buffer.erase(next_frame_to_write); // Libera da memória RAM
                    next_frame_to_write++;
                    
                    if (next_frame_to_write % 100 == 0) {
                        std::cout << "Gravados " << next_frame_to_write << " frames no video...\n";
                    }
                }
            }
        }

        cap.release();
        writer.release();
        std::cout << "Concluido! Video salvo com sucesso. " << next_frame_to_write << " frames processados no total.\n";

    } else {
        // PROCESSO WORKER (Trabalhador)
        while (true) {
            // Avisa o Master que está pronto para trabalhar (TAG 0)
            MPI_Send(&myid, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Recebe os metadados do trabalho (TAG 1)
            int metadados[5];
            MPI_Recv(metadados, 5, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int linhas = metadados[0];
            if (linhas == -1) break; // Recebeu ordem de desligar, sai do loop

            int colunas = metadados[1];
            int tipo = metadados[2];
            int total_bytes = metadados[3];
            int frame_idx = metadados[4];

            // Recebe os bytes originais da imagem (TAG 2)
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

                
                // apply_gaussian(frame, frame_processado, start_row, end_row); 
                apply_sobel(frame, frame_processado, start_row, end_row);
                // apply_sharpen(frame, frame_processado, start_row, end_row);
            }

            MPI_Send(metadados, 5, MPI_INT, 0, 3, MPI_COMM_WORLD);
            MPI_Send(frame_processado.data, total_bytes, MPI_BYTE, 0, 4, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
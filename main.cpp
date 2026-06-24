// Processador de video distribuido: MPI divide frames entre processos,
// OpenMP paraleliza linhas dentro de cada worker.

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

// Quantas threads OpenMP cada worker usa ao processar um frame
#define MAX_THREADS 8

// Tags MPI — protocolo Master <-> Workers
#define TAG_WORKER_READY  0  // worker avisa que esta livre
#define TAG_FRAME_META    1  // metadados do frame (5 ints)
#define TAG_FRAME_DATA    2  // pixels brutos
#define TAG_RESULT_META   3  // metadados do frame processado
#define TAG_RESULT_DATA   4  // pixels processados
// Metadados = [rows, cols, type, total_bytes, frame_index]; rows=-1 encerra worker


// 1. GAUSSIANO (Redução de ruído / Desfoque)
// Convolucao 3x3 canal a canal; start_row/end_row permitem divisao entre threads OpenMP

void apply_gaussian(const cv::Mat& src, cv::Mat& dst, int start_row, int end_row) {
    // Matriz de pesos: centro vale mais (4) que cantos (1) — efeito de media suavizada
    int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    int kernel_weight = 16; // soma de todos os pesos; usada para normalizar o resultado
    // Evita acessar pixels fora da imagem (vizinhanca 3x3 precisa de borda interna)
    int start = std::max(1, start_row);
    int end = std::min(src.rows - 1, end_row);

    for (int r = start; r < end; r++) {           // cada thread processa seu intervalo de linhas
        for (int c = 1; c < src.cols - 1; c++) {  // pula colunas das bordas esquerda/direita
            for (int ch = 0; ch < 3; ch++) {      // B=0, G=1, R=2 — trata cada canal separado
                int sum = 0;
                // Percorre os 9 vizinhos do pixel (r,c) e acumula valor * peso
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        int pixel_val = src.at<cv::Vec3b>(r + kr, c + kc)[ch];
                        sum += pixel_val * kernel[kr + 1][kc + 1];
                    }
                }
                // Divide pela soma dos pesos e limita entre 0-255
                dst.at<cv::Vec3b>(r, c)[ch] = std::clamp(sum / kernel_weight, 0, 255);
            }
        }
    }
}


// 2. SHARPEN (Nitidez)
// Realca detalhes: pixel central * 5 menos a soma dos 4 vizinhos diretos (cima/baixo/esq/dir)

void apply_sharpen(const cv::Mat& src, cv::Mat& dst, int start_row, int end_row) {
    // Centro positivo (5), vizinhos negativos (-1) — amplifica diferenca em relacao ao entorno
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
                // Sharpen nao normaliza; clamp evita valores negativos ou acima de 255
                dst.at<cv::Vec3b>(r, c)[ch] = std::clamp(sum, 0, 255);
            }
        }
    }
}


// 3. SOBEL (Detecção de Bordas)
// Kx = gradiente horizontal, Ky = gradiente vertical; magnitude = sqrt(gx² + gy²)

void apply_sobel(const cv::Mat& src, cv::Mat& dst, int start_row, int end_row) {
    // Kx detecta variacao na horizontal (bordas verticais)
    int Kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    // Ky detecta variacao na vertical (bordas horizontais)
    int Ky[3][3] = {{-1, -2, -1}, { 0, 0, 0}, { 1, 2, 1}};
    int start = std::max(1, start_row);
    int end = std::min(src.rows - 1, end_row);

    for (int r = start; r < end; r++) {
        for (int c = 1; c < src.cols - 1; c++) {
            for (int ch = 0; ch < 3; ch++) {
                int sum_x = 0; // gradiente na direcao x
                int sum_y = 0; // gradiente na direcao y
                for (int kr = -1; kr <= 1; kr++) {
                    for (int kc = -1; kc <= 1; kc++) {
                        int pixel_val = src.at<cv::Vec3b>(r + kr, c + kc)[ch];
                        sum_x += pixel_val * Kx[kr + 1][kc + 1];
                        sum_y += pixel_val * Ky[kr + 1][kc + 1];
                    }
                }
                // Quanto maior a magnitude, mais forte a borda nesse pixel
                int magnitude = std::sqrt(sum_x * sum_x + sum_y * sum_y);
                dst.at<cv::Vec3b>(r, c)[ch] = std::clamp(magnitude, 0, 255);
            }
        }
    }
}

// Protocolo MPI — ver defines TAG_* acima

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

            // Workers podem devolver frames fora de ordem; buffer reordena antes de gravar
            std::map<int, cv::Mat> frame_buffer;
            int next_frame_to_write = 0;
            int sent_frames = 0;
            bool video_done = false;
            int workers_alive = numprocs - 1;

            // Loop principal de distribuição do Master
            // Roda enquanto ainda ha workers aguardando sinal de encerramento (linhas = -1)
            while (workers_alive > 0) {
                MPI_Status status;
                // Probe antes do Recv: descobre quem mandou mensagem e qual tag
                // MPI_ANY_SOURCE: aceita mensagem de qualquer worker (nao bloqueia em ordem fixa)
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                if (status.MPI_TAG == TAG_WORKER_READY) {
                    // Worker terminou um frame (ou acabou de iniciar) e pede mais trabalho
                    int id_livre;
                    MPI_Recv(&id_livre, 1, MPI_INT, status.MPI_SOURCE, TAG_WORKER_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    cv::Mat current_frame;
                    if (!video_done && cap.read(current_frame)) {
                        // Monta pacote: dimensoes, tipo OpenCV, tamanho em bytes, indice do frame
                        int metadados[5] = {
                            current_frame.rows, 
                            current_frame.cols, 
                            current_frame.type(), // tipo OpenCV (CV_8UC3, CV_8UC4, etc)
                            (int)(current_frame.total() * current_frame.elemSize()), // tamanho em bytes
                            sent_frames++ // indice do frame (0, 1, 2, ...)
                        };
                        MPI_Send(metadados, 5, MPI_INT, id_livre, TAG_FRAME_META, MPI_COMM_WORLD);
                        MPI_Send(current_frame.data, metadados[3], MPI_BYTE, id_livre, TAG_FRAME_DATA, MPI_COMM_WORLD); // envia os pixels brutos (bytes)
                    } else {
                        // Sem frames restantes: manda sinal de fim para este worker
                        video_done = true;
                        int end_meta[5] = {-1, 0, 0, 0, 0}; // rows=-1 e o codigo de encerramento no worker
                        MPI_Send(end_meta, 5, MPI_INT, id_livre, TAG_FRAME_META, MPI_COMM_WORLD);
                        workers_alive--; // um worker a menos aguardando trabalho
                    }
                }
                else if (status.MPI_TAG == TAG_RESULT_META) {
                    // Worker devolveu frame processado — recebe metadados e pixels
                    int meta[5];
                    MPI_Recv(meta, 5, MPI_INT, status.MPI_SOURCE, TAG_RESULT_META, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    std::vector<uchar> buf(meta[3]); // aloca buffer com o tamanho exato do frame
                    MPI_Recv(buf.data(), meta[3], MPI_BYTE, status.MPI_SOURCE, TAG_RESULT_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    cv::Mat p_frame(meta[0], meta[1], meta[2], buf.data()); // reconstrói Mat sem copiar ainda
                    // meta[4] = indice do frame no video (0, 1, 2...), NAO a ordem em que chegou
                    // O map usa esse indice como chave: frame_buffer[2] = "frame numero 2 pronto"
                    // Workers terminam fora de ordem, entao o 2 pode chegar antes do 1 — guardamos no slot certo e esperamos
                    frame_buffer[meta[4]] = p_frame.clone(); // clone copia pixels antes de buf ser destruido

                    // Grava em sequencia (0, 1, 2...) assim que o proximo indice chegar
                    // Ex: se chegou frame 2 mas falta o 1, guarda no map ate o 1 aparecer
                    // next_frame_to_write = proximo indice que o video de saida exige; so avanca sem buracos
                    while(frame_buffer.count(next_frame_to_write)) {
                        // Tem o frame que precisamos agora? Grava e tenta o proximo (1, 2, 3...)
                        writer.write(frame_buffer[next_frame_to_write]);
                        frame_buffer.erase(next_frame_to_write); // libera memoria do frame ja gravado
                        next_frame_to_write++; // so incrementa depois de gravar — mantem ordem no mp4

                        if (next_frame_to_write % 100 == 0) {
                            std::cout << "Gravados " << next_frame_to_write << " frames no video...\n";
                        }
                    }
                    // Se faltou algum indice (ex: tem 0 e 2 mas nao 1), o while para e espera o 1 chegar
                }
            }

            cap.release();
            writer.release();
            std::cout << "\nConcluido! Video salvo com sucesso ('resultado.mp4').\n";

        } else {
            // PROCESSO WORKER
            while (true) {
                // Avisa o Master que está livre (TAG_WORKER_READY)
                MPI_Send(&myid, 1, MPI_INT, 0, TAG_WORKER_READY, MPI_COMM_WORLD);

                // Recebe os metadados do frame (TAG_FRAME_META)
                int metadados[5];
                MPI_Recv(metadados, 5, MPI_INT, 0, TAG_FRAME_META, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int linhas = metadados[0];
                if (linhas == -1) break; // O vídeo acabou! Quebra esse loop e volta pro Menu principal

                // Recebe os metadados do frame (TAG_FRAME_META)
                int colunas = metadados[1];
                int tipo = metadados[2];
                int total_bytes = metadados[3];
                int frame_idx = metadados[4];

                // Recebe os pixels brutos (TAG_FRAME_DATA)
                std::vector<uchar> buffer(total_bytes);
                MPI_Recv(buffer.data(), total_bytes, MPI_BYTE, 0, TAG_FRAME_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Reconstrói Mat sem copiar ainda
                cv::Mat frame(linhas, colunas, tipo, buffer.data());
                cv::Mat frame_processado = cv::Mat::zeros(frame.size(), frame.type());

                int linhas_locais = linhas / MAX_THREADS; // Divide o numero de linhas entre as threads

                // Paraleliza o processamento das linhas do frame
                #pragma omp parallel num_threads(MAX_THREADS)
                {
                    int thread_id = omp_get_thread_num();
                    int start_row = thread_id * linhas_locais;
                    // Ultima thread cobre linhas restantes quando linhas % MAX_THREADS != 0
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

                // Devolve para o Master (TAG_RESULT_META e TAG_RESULT_DATA)
                MPI_Send(metadados, 5, MPI_INT, 0, TAG_RESULT_META, MPI_COMM_WORLD);
                MPI_Send(frame_processado.data, total_bytes, MPI_BYTE, 0, TAG_RESULT_DATA, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}

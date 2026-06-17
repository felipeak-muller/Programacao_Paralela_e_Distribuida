#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <omp.h>

#include MAX_THREADS 8

int apply_gaussian([][]int mat) {

}

int apply_sobel([][]int mat) {
    
}

int apply_sharpen([][] int mat) {

}


int main(int argc, char** argv) {
    int myid, numprocs;

    // Inicialização e identificação dos processos
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (numprocs < 2) {
        if (myid == 0)
            std::cerr << "Execute com pelo menos 2 processos MPI.\n";
        MPI_Finalize();
        return 1;
    }

    // Processo Master
    if (myid == 0) {
        const std::string video_path = "sample.mp4";
        const std::string output_dir = "frames";

        std::filesystem::create_directories(output_dir);

        // cap: arquivo de video aberto
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Erro ao abrir o video: " << video_path << "\n";
            MPI_Finalize();
            return 1;
        }

        std::cout << "Video aberto: " << video_path << "\n";
        std::cout << "FPS: " << cap.get(cv::CAP_PROP_FPS) << "\n";
        std::cout << "Total de frames: " << cap.get(cv::CAP_PROP_FRAME_COUNT) << "\n";

        cv::Mat frame;
        int frame_idx = 0;

        // cap.read() decodifica o proximo frame; retorna false no fim do video
        while (cap.read(frame)) {

            // Recebe sinal de um processo livre
            int processo_livre;
            MPI_Recv(&processo_livre, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Calcula tamanho total dos pixeis em bytes
            size_t total_bytes = frame.total() * frame.elemSize();

            // Enviar: [linhas, colunas, tipo, total_de_bytes, frame_idx]
            int metadados[5] = {frame.rows, frame.cols, frame.type(), (int)total_bytes, frame_idx++};
            MPI_Send(metadados, 5, MPI_INT, processo_livre, 0, MPI_COMM_WORLD);

            // Envia o frame para o processo livre
            MPI_Send(frame.data, total_bytes, MPI_BYTE, processo_livre, 1, MPI_COMM_WORLD);
        }

        cap.release();

        // Final, envia sinal de fim para todos os processos
        for (int i = 1; i < numprocs; i++) {
            // Após cada worker finalizar, ele envia que está disponível
            // se enviarmos direto o final, tanto a master envia que terminou quanto
            // worker envia que está pronto, entrando em deadlock
            // Para resolver primeiro recebemos o sinal que ele está pronto, e depois enviamos o final
            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Envia sinal de fim (linhas == -1 indica encerramento)
            int end[5] = {-1, 0, 0, 0, 0};
            MPI_Send(end, 5, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        std::cout << "Fim do video. " << frame_idx << " frames distribuidos.\n";

    } else {
        // Processos workers
        const std::string output_dir = "frames";

        while (true) {

            // Sinaliza que está disponível
            MPI_Send(&myid, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Recebe metadados
            int metadados[5];
            MPI_Recv(metadados, 5, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Extrai metadados
            int linhas      = metadados[0];
            int colunas     = metadados[1];
            int tipo        = metadados[2];
            int total_bytes = metadados[3];
            int frame_idx   = metadados[4];

            // Se linhas == -1 então acabou
            if (linhas == -1) break;

            // Aloca buffer e recebe do master o frame
            std::vector<uchar> buffer(total_bytes);
            MPI_Recv(buffer.data(), total_bytes, MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            cv::Mat frame(linhas, colunas, tipo, buffer.data());
            
            // Quantas linhas cada thread ficará responsável
            int linhas_locais = linhas / MAX_THREADS;

            // Aqui vamos spawnar  8 threads
            #pragma omp parallel max_threads(8)
            {
                // Puxa indice da linha

                // Percorre a matriz aplicando o filtro escolhido
            }

            std::cout << "Worker " << myid << " salvou " << filename << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}

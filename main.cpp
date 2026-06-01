#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    int myid, numprocs;

    // Inicialização e identificação dos processos
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    // Processo Master
    if (myid == 0) {
        const std::string video_path = "sample.mp4";
        const std::string output_dir = "frames";
        
        // cap: arquivo de video aberto
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Erro ao abrir o video: " << video_path << "\n";
            return 1;
        }

        std::cout << "Video aberto: " << video_path << "\n";
        std::cout << "FPS: " << cap.get(cv::CAP_PROP_FPS) << "\n";
        std::cout << "Total de frames: " << cap.get(cv::CAP_PROP_FRAME_COUNT) << "\n";
        
        cv::Mat frame;
        // cap.read() decodifica o proximo frame; retorna false no fim do video
        while (cap.read(frame)) {
            
            // Recebe sinal de um processo livre
            int processo_livre;
            MPI_Recv(&processo_livre, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_IGNORE_STATUS);

            // Calcula tamanho total dos pixeis em bytes
            size_t total_bytes = frame.total() * frame.elemSize();
            
            // Enviar: [linhas, colunas, tipo, total_de_bytes]
            int metadados[4] = {frame.rows, frame.cols, frame.type(), (int)total_bytes};
            MPI_Send(metadados, 4, MPI_INT, processo_livre, 0, MPI_COMM_WORLD);

            // Envia o frame para o processo livre
            MPI_Send(frame.data, total_bytes, MPI_BYTE, processo_livre, 1, MPI_COMM_WORLD);
        }

        // Final, envia sinal de fim para todos os processos 
        for (int i = 1; i < numprocs; i++) {
            // Após cada worker finalizar, ele envia que está disponível
            // se enviarmos direto o final, tanto a master envia que terminou quanto
            // worker envia que está pronto, entrando em deadlock
            // Para resolver primeiro recebemos o sinal que ele está pronto, e depois enviamos o final
            MPI_Recv(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Envia sinal de fim
            MPI_Send(-1, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        std::cout << "Fim do video.\n";

    } else {
        // Processos workers

        while (true) {

            // Sinaliza que está disponível
            MPI_Send(myid, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Recebe metadados
            int metadados[4];
            MPI_STATUS status;
            MPI_Recv(metadados, 4, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            
            // Extrai metadados
            int linhas = metadados[0];
            int colunas = metadados[1];
            int tipo = metadados[2];
            int total_bytes = metadados[3];
            
            // Se status for fim então acabou
            if (status.MPI_TAG == -1) {
                break;
            }


            cv::Mat frame;
            // Recebe do master o frame e aloca em tag
            MPI_Recv(&frame, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            
            // Processamento do frame

        }
        
    }
    


    cap.release();
    std::cout << "Extraidos " << frame_idx << " frames.\n";
    return 0;
}

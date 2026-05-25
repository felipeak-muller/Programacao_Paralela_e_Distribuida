#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main() {
    const std::string video_path = "sample.mp4";
    const std::string output_dir = "frames";

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir o video: " << video_path << "\n";
        return 1;
    }

    std::cout << "Video aberto: " << video_path << "\n";
    std::cout << "FPS: " << cap.get(cv::CAP_PROP_FPS) << "\n";
    std::cout << "Total de frames: " << cap.get(cv::CAP_PROP_FRAME_COUNT) << "\n";

    fs::create_directories(output_dir);
    std::cout << "Salvando frames em: " << output_dir << "/\n";

    cv::Mat frame;
    int frame_idx = 0;

    // cap.read() decodifica o proximo frame; retorna false no fim do video
    while (cap.read(frame)) {
        if (frame.empty()) {
            break;
        }

        // Aqui deveriamos distribuir os frames para o MPI e depois threads,
        // mas como exemplo, vamos salvar o frame em um arquivo.
        const std::string filename =
            output_dir + "/frame_" + std::to_string(frame_idx) + ".jpg";

        if (!cv::imwrite(filename, frame)) {
            std::cerr << "Erro ao salvar: " << filename << "\n";
            return 1;
        }

        frame_idx++;
    }

    cap.release();
    std::cout << "Extraidos " << frame_idx << " frames.\n";
    return 0;
}

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

int main(int argc, char** argv) {
	// List of tracker types in OpenCV 3.4.1
	string trackerTypes[8] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" };

	// Create a tracker
	string trackerType = trackerTypes[2];
	Ptr<Tracker> tracker;
	tracker = Tracker::create(trackerType);

	// Lendo o video
	VideoCapture video("teste3.mp4");

	// Se não encontrar o arquivo fecha da error
	if (!video.isOpened()) {
		cout << "Arquivo não encontrado" << endl;
		return 1;
	}

	// Abre uma janela para selecionar o que quer rastreiar
	Mat frame;
	bool ok = video.read(frame);
	Rect2d bbox = selectROI(frame, false);
	rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
	tracker->init(frame, bbox);

	//Recorta o pedaço que foi selecionado
	Mat imagemSelecionada = frame(bbox);

	//Salvar o pedaço capturado
	imwrite("captura.jpeg", imagemSelecionada);
	Mat imagemCap = imread("captura.jpeg");

	//Variaveis para manipular as imagens/video
	Mat image2, imageAux;

	while (video.read(frame)) {
		

		//coloca cada frame do video em image2, quando acabar encerra o loop
		video >> image2;
		if (image2.empty()) {
			break;
		}

		// Se apertar ESC encerra o programa.
		int k = waitKey(1);
		if (k == 27) {
			break;
		}

		//Encontrando os pontos de matches
		vector<KeyPoint> kp1, kp2;
		Mat descriptor1, descriptor2;
		Ptr<Feature2D> orb = xfeatures2d::SIFT::create(40);
		orb->detectAndCompute(imagemCap, Mat(), kp1, descriptor1);
		orb->detectAndCompute(image2, Mat(), kp2, descriptor2);

		//Desenha os orbs encontrados
		drawKeypoints(imagemCap, kp1, imageAux);
		drawKeypoints(image2, kp2, image2);
		vector<DMatch> matches;
		BFMatcher matcher;
		matcher.match(descriptor1, descriptor2, matches);
		/*Mostra os pontos chave tanto o vídeo quanto na imagem de apoio (figura que se quer rastrear)
			em novas janelas
		*/
		Mat img_Matches, H;
		drawMatches(imagemCap, kp1, image2, kp2, matches, img_Matches);
		imshow("Maches", img_Matches);

		// Contador para o FPS
		double timer = (double)getTickCount();

		//########Retangulo###########
		// Atualiza o retangulo rastreado
		bool ok = tracker->update(frame, bbox);

		// Calculate Frames per second (FPS)
		float fps = getTickFrequency() / ((double)getTickCount() - timer);
		string novoFPS = to_string((int)fps);

		if (ok) {
			// Se encontrar o ojeto rastreado no video, mostra um retango ao redor dele
			rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
		}
		else {
			// Se não encontrar, mostra uma mensagem de erro
			putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		}

		//Mostrar FPS
		putText(frame, "FPS: " + novoFPS, Point(50, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

		// Mostrar o retangulo rastreando.
		imshow("Tracking", frame);
	}
}

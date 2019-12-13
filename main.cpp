#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"


using namespace cv;
using namespace std;
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

int main(int argc, char** argv) {
	// Lista com todos os Tracks possiveis
	string trackerTypes[8] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" };

	// Criar o Tracker
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

		//Encontrando os pontos de matches, os Orbs
		vector<KeyPoint> kp1, kp2;
		Mat descriptor1, descriptor2;
		Ptr<Feature2D> orb = xfeatures2d::SIFT::create(400);
		orb->detectAndCompute(imagemCap, Mat(), kp1, descriptor1);
		orb->detectAndCompute(image2, Mat(), kp2, descriptor2);

		//Colocando em escala de cinza
		Mat image2Gray, imagemCapGray;
		cvtColor(image2, image2Gray, CV_BGR2GRAY);
		cvtColor(imagemCap, imagemCapGray, CV_BGR2GRAY);
		
		//Desenha os orbs encontrados
		drawKeypoints(imagemCap, kp1, imageAux);
		drawKeypoints(image2, kp2, image2);
		vector<DMatch> matches;
		BFMatcher matcher;
		matcher.match(descriptor1, descriptor2, matches);

		//Mostra os pontos chave tanto o vídeo quanto na imagem de apoio (figura que se quer rastrear)
		//em novas janelas além dos matches (sem o filtro dos good matches)
		Mat img_Matches;
		drawMatches(imagemCap, kp1, image2, kp2, matches, img_Matches);
		imshow("Maches", img_Matches);


		//############# - GOOD MATCHS MODELO 2 - ###################
		vector< DMatch > good_matches;
		double min_dist = 50;
		double max_dist = 0;

		// Percorre todas as linhas da matriz e verifica a menor distância entre
		// keypoints e a maior
		for (int i = 0; i < descriptor1.rows; i++){
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		for (int i = 0; i < descriptor1.rows; i++){
			if (matches[i].distance < 3 * min_dist)
				good_matches.push_back(matches[i]);
		}

		// Desenha os raios dos Good matches 2
		Mat img_matches;
		drawMatches(imagemCap, kp1, image2, kp2,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		vector<Point2f> obj;
		vector<Point2f> scene;

		for (int i = 0; i < good_matches.size(); i++){
			obj.push_back(kp1[good_matches[i].queryIdx].pt);
			scene.push_back(kp2[good_matches[i].trainIdx].pt);
		}
		
		// Faz a homografia entre a imagem e o video
		Mat H = findHomography(obj, scene);


		//############## - GOOD MATCHS MODELO 1 - ##################
		//Realiza o calculo para pegar os melhores pontos
		sort(matches.begin(), matches.end());
		const int numGoodMatches = matches.size() * 0.05f;
		matches.erase(matches.begin() + numGoodMatches, matches.end());

		// Desenha os raios dos Good matches 1
		Mat imMatches;
		drawMatches(imagemCap, kp1, image2, kp2, matches, imMatches);
		vector<Point2f> points1, points2;
		for (size_t i = 0; i < matches.size(); i++){
			points1.push_back(kp1[matches[i].queryIdx].pt);
			points2.push_back(kp2[matches[i].trainIdx].pt);
		}

		// Faz a homografia entre a imagem e o video
		Mat h = findHomography(points1, points2);

		// Pega as cordenadas da image e manda para o video, se detectar
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(imagemCap.cols, 0);
		obj_corners[2] = cvPoint(imagemCap.cols, imagemCap.rows); obj_corners[3] = cvPoint(0, imagemCap.rows);
		std::vector<Point2f> scene_corners(4);

		// Coloca a homografia com as coordenadas
		perspectiveTransform(obj_corners, scene_corners, h);

		// DESSENHAR AS LINHAS NA FORMA 1
		line(imMatches, scene_corners[0] + Point2f(imagemCap.cols, 0), scene_corners[1] + Point2f(imagemCap.cols, 0), Scalar(0, 255, 0), 4);
		line(imMatches, scene_corners[1] + Point2f(imagemCap.cols, 0), scene_corners[2] + Point2f(imagemCap.cols, 0), Scalar(0, 255, 0), 4);
		line(imMatches, scene_corners[2] + Point2f(imagemCap.cols, 0), scene_corners[3] + Point2f(imagemCap.cols, 0), Scalar(0, 255, 0), 4);
		line(imMatches, scene_corners[3] + Point2f(imagemCap.cols, 0), scene_corners[0] + Point2f(imagemCap.cols, 0), Scalar(0, 255, 0), 4);
		
		// DESSENHAR AS LINHAS NA FORMA 2
		perspectiveTransform(obj_corners, scene_corners, H);
		line(img_matches, scene_corners[0] + Point2f(imagemCap.cols, 0), scene_corners[1] + Point2f(imagemCap.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[1] + Point2f(imagemCap.cols, 0), scene_corners[2] + Point2f(imagemCap.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[2] + Point2f(imagemCap.cols, 0), scene_corners[3] + Point2f(imagemCap.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[3] + Point2f(imagemCap.cols, 0), scene_corners[0] + Point2f(imagemCap.cols, 0), Scalar(0, 255, 0), 4);

		// Imprime os dois modelos
		imshow("MODELO 1", imMatches);
		imshow("MODELO 2", img_matches);

		// Contador para o FPS
		double timer = (double)getTickCount();

		//######## - Retangulo sem homografia - ###########
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
		putText(imMatches, "FPS: " + novoFPS, Point(50, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

		// Mostrar o retangulo rastreando.
		imshow("Tracking", frame);
	}
}

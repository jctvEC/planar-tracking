#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\tracking.hpp"

using namespace cv;
using namespace std;

int main() {
	namedWindow("Teste", WINDOW_AUTOSIZE);
	Mat image1, image2, imageAux;
	VideoCapture cap("teste2.MP4");

	if (!cap.isOpened()) { //verifica se cap abriu como esperado
		cout << "camera ou arquivo em falta";
		std::cin.get();
		return 1;
	}
	image1 = imread("reis1.JPEG", IMREAD_GRAYSCALE);

	if (image1.empty()) { //verifica a imagem1
		cout << "imagem 1 vazia";
		std::cin.get();
		return 1;
	}

	int frameI = 0;

	//Variaveis para criar o retangulo.
	Mat frame;
	cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create("");
	cap.read(frame);
	cv::Rect2d trackingBox = cv::selectROI(frame, false);
//	tracker->init(frame, trackingBox);
	int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int frameHeigth = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	VideoWriter output("teste.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(frameWidth, frameHeigth));


	while (true) {
		cap >> image2;
		if (image2.empty()) {
			cout << "imagem 2 vazia";
			std::cin.get();
			return 1;
		}
		

		cvtColor(image2, image2, COLOR_BGR2GRAY);//coloca em grayscale

		vector<KeyPoint> kp1, kp2;
		Mat descriptor1, descriptor2;

		/*Aqui se tem 3 formas de encontrar os pontos de match.
			SIFT, SURF e ORB
		*/
		Ptr<Feature2D> orb = xfeatures2d::SIFT::create(40);
		//Ptr<Feature2D> orb = xfeatures2d::SURF::create(400);
		//Ptr<Feature2D> orb = ORB::create(400);
		orb->detectAndCompute(image1, Mat(), kp1, descriptor1);
		orb->detectAndCompute(image2, Mat(), kp2, descriptor2);

		


		drawKeypoints(image1, kp1, imageAux);
		drawKeypoints(image2, kp2, image2);

		vector<DMatch> matches;
		BFMatcher matcher;
		matcher.match(descriptor1, descriptor2, matches);

		/*Mostra os pontos chave tanto o vídeo quanto na imagem de apoio (figura que se quer rastrear)
			em novas janelas
		*/
		imshow("teste", image2);
		imshow("Teste", imageAux);

		//Desenha os matches em uma nova janela
		namedWindow("Teste", 0);
		Mat img_Matches,H,frame;
		//Stats stats;
		drawMatches(image1, kp1, image2, kp2, matches, img_Matches);
		imshow("Teste", img_Matches);

		//Desenhar o Retangulo
		rectangle(image2, trackingBox, cv::Scalar(255, 0, 0), 2, 8);
		imshow("Video feed", image2);
		output.write(image2);

		
		//vector<Point2f> points1, points2;
		//vector<Point2d> points1, points2;
		vector<KeyPoint> points1, points2;;

		//if(points1.size()>=4)
		//H = findHomography(points1, points2, RANSAC, 2.5f, img_Matches);

		frameI = frameI + 1;

		if (waitKey(1) == 27) {
			std::cin.get();
			break;
		}
	}

	std::cin.get();

	cv::destroyAllWindows();

	return 0;
}

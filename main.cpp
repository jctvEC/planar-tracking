
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
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

bool comparator(DMatch distancia1, DMatch distancia2);


int main() {
	namedWindow("TestePG", WINDOW_AUTOSIZE);
	Mat image1, image2, imageAux;
	VideoCapture cap("teste.MP4");

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
		Ptr<Feature2D> orb = xfeatures2d::SIFT::create(400);
		//Ptr<Feature2D> orb =3D xfeatures2d::SURF::create(400);
		//Ptr<Feature2D> orb =3D ORB::create(400);
		orb->detectAndCompute(image1, Mat(), kp1, descriptor1);
		orb->detectAndCompute(image2, Mat(), kp2, descriptor2);


		drawKeypoints(image1, kp1, imageAux);
		drawKeypoints(image2, kp2, image2);

		vector<DMatch> matches;
		BFMatcher matcher (cv::NORM_L2, true);
		matcher.match(descriptor1, descriptor2, matches);
		printf("%d ", matches.size());
		sort(matches.begin(), matches.end(), comparator);//ordena para pegar apenas os primeiros, ou seja os mais proximos.
		while (matches[matches.size() - 1].distance - matches[0].distance < 4) matches.pop_back();
		printf("%d ", matches.size());



		/*Mostra os pontos chave tanto o v=C3=ADdeo quanto na imagem de apoio (figura que se quer rastrear)		em novas janelas
		*/
		imshow("teste", image2);
		imshow("Teste", imageAux);

		//Desenha os matches em uma nova janela
		namedWindow("Teste", 0);
		Mat img_Matches, H, frame;
		//Stats stats;
		drawMatches(image1, kp1, image2, kp2, matches, img_Matches);
		imshow("Teste", img_Matches);

		//vector<Point2f> points1, points2;
		//vector<Point2d> points1, points2;
		vector<KeyPoint> points1, points2;


		if (points1.size() >= 4)
			//H = findHomography(points1, points2, RANSAC, 2.5f, img_Matches);

		//cout << "Frame No.: " << frameI + 1 << std::endl;
		//waitKey(0);
		frameI =frameI + 1;

		if (waitKey(1) == 27) {
			std::cin.get();
			break;
		}
	}

	//output.release();
	//cap.release();

	std::cin.get();

	cv::destroyAllWindows();

	return 0;
}

bool comparator(DMatch distancia1, DMatch distancia2) {
	return distancia1.distance > distancia1.distance;
};

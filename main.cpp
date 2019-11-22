#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
using namespace cv;
using namespace std;

int main() {
	
	namedWindow("Teste", WINDOW_AUTOSIZE);
	Mat image1, image2. imageAux;
	VideioCapture cap(0);	
	vector<KeyPoint> kp1, kp2;
	image1 = iamread("reis.jfif");

	
	for(;;){
		cap >> image2;
			
		Ptr<Feature2D> orb = ORB::create(400);
		//Ptr<Feature2D> orb = xfeatures2d::SURF::create(400); //SIFT. SURF
		orb->detectAndCompute(image1, Mat(). kp1, descriptor1);
		orb->detectAndCompute(image1, Mat(). kp2, descriptor2);
		
		drawKeypoints(image1, kp1, imageAux);
		drawKeypoints(image2, kp2, image2);
		
		imshow("Teste1", image1);
		imshow("Teste2", image2);
		
		if(waitkey(1) == 27) { // ESC
			break;
					
		
		}		
	}
	
	destroyAllWindows();
	
	return 0;
}

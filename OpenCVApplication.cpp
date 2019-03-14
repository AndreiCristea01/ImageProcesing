// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <string>


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}
void showHistogram2(const std::string& name, float* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	float max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

uchar test_value(uchar x) {

	if (x > 255)
		return 255;
	else
		return max(0, x);
}
void change_to_aditiv() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		Mat clone = src.clone();
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				src.at<uchar>(i, j) = test_value(src.at<uchar>(i, j) + 50);
			}
		}
		imshow("initial image", clone);
		imshow("new image", src);
		waitKey();
	}
}

void change_to_multiply() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat clone = src.clone();
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				src.at<uchar>(i, j) = test_value(src.at<uchar>(i, j)*1.5);
			}
		}
		imshow("initial image", clone);
		imshow("new image", src);
		waitKey();
	}
}
void image_color() {

	Mat img(256, 256, CV_8UC3);
	for(int i=0;i<256;i++)
		for(int j=0;j<256;j++)
			if (i < 128)
			{
				if (j < 128)
					img.at<Vec3b>(i, j) = { 255,255,255 };
				else
					img.at<Vec3b>(i, j) = { 0,0,255 };
			}
			else
			{
				if(j>128)
					img.at<Vec3b>(i, j) = { 0,255,0 };
				else
					img.at<Vec3b>(i, j) = { 0,255,255 };
			}
	imshow("image", img);
	waitKey(0);
}
void inverseMatrix()
{
	float values[9] = { 1, 3, 3, 3, 2, 8, 9, 6, 6 };
	Mat matrix(3, 3, CV_32FC1, values);
	Mat result = matrix.inv();
	std::cout << matrix << std::endl;
	std::cout << result << std::endl;
	getchar();
	getchar();
	waitKey(0);
}
void matrix_RGB() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

		int height = src.rows;
		int width = src.cols;
		Mat matrixB(height, width, CV_8UC1);
		Mat matrixG(height, width, CV_8UC1);
		Mat matrixR(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				matrixB.at<uchar>(i, j) = pixel[0];
				matrixG.at<uchar>(i, j) = pixel[1];
				matrixR.at<uchar>(i, j) = pixel[2];
			}
		imshow("image", src);
		imshow("B", matrixB);
		imshow("G", matrixG);
		imshow("R", matrixR);

		waitKey();
	}


}
void convert_grayscale() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

		int height = src.rows;
		int width = src.cols;
		Mat gray(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);

				gray.at<uchar>(i, j) = (pixel[0] + pixel[1] + pixel[2]) / 3;
			}
		imshow("image", src);
		imshow("gray", gray);


		waitKey();
	}


}
void grayscale_binar() {

	char fname[MAX_PATH];
	std::cout << "trash=";
	uchar t;
	std::cin >> t;
	while (openFileDlg(fname))
	{

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;
		Mat binar(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) > t)
					binar.at<uchar>(i, j) = 255;
				else
				{
					binar.at<uchar>(i, j) = 0;

				}


			}
		imshow("image", src);
		imshow("gray", binar);


		waitKey();
	}
}

Mat RGB_HSV(String fname) {



	Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

	int height = src.rows;
	int width = src.cols;
	Mat matrixH(height, width, CV_8UC1);
	Mat matrixS(height, width, CV_8UC1);
	Mat matrixV(height, width, CV_8UC1);
	Mat matrix_rez(height, width, CV_8UC3);
	std::vector<Mat> hsv;
	float r, g, b, M, m, C, V, S, H;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			Vec3b pixel = src.at<Vec3b>(i, j);
			b = (float)pixel[0] / 255;
			g = (float)pixel[1] / 255;
			r = (float)pixel[2] / 255;
			M = max(r, g, b);
			m = min(r, g, b);
			C = M - m;
			V = M;
			if (V != 0)
				S = C / V;
			else
			{
				S = 0;
			}
			if (C != 0) {
				if (M == r) H = 60 * (g - b) / C;
				if (M == g) H = 120 + 60 * (b - r) / C;
				if (M == b) H = 240 + 60 * (r - g) / C;
			}
			else {
				H = 0;
			}
			if (H < 0)
				H = H + 360;
			matrixH.at<uchar>(i, j) = H * 255 / 360;
			matrixS.at<uchar>(i, j) = S * 255;
			matrixV.at<uchar>(i, j) = V * 255;

		}

	
	hsv.push_back(matrixH);
	hsv.push_back(matrixS);
	hsv.push_back(matrixV);
	merge(hsv, matrix_rez);


	return matrix_rez;


}

bool isInside(int i, int j, Mat img) {

	if (i < 0 || j < 0)
		return 0;
	else
	{
		if (i < img.rows && j < img.cols)
			return 1;
	}

	return 0;
}


int * Histogram(String fname) {

	Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
	int rez[256];
	for (int i = 0; i < 256; i++) {
		rez[i] = 0;
	}

	int height = src.rows;
	int width = src.cols;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			rez[src.at<uchar>(i, j)]++;
		}
	

	return rez;
	

}
float *  FDP(String fname) {

	
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		
		float rez[256];
		int*  his = Histogram(fname);
		for (int i = 0; i < 256; i++) {
			rez[i] = 0.0;
			
		}
	
		
		for (int i = 0; i < 256; i++) {
			rez[i] = (float)his[i] /(float) (src.rows*src.cols);
		}
		
		return rez;
		

	

}
void acumulator(String fname, int nr) {

	Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
	int rez[256];
	for (int i = 0; i < 256; i++) {
		rez[i] = 0;
	}

	int height = src.rows;
	int width = src.cols;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			rez[src.at<uchar>(i, j)/nr]++;
		}


	showHistogram("histogram", rez, 256, 200);
	waitKey();

}

void praguri_multiple() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		float *fdp = FDP(fname);
		std::vector<int> vec;
		int wh = 5;
		int l_fer = 2 * wh + 1;
		float th = 0.0003;
		double m;
		bool maxim;
		vec.push_back(0);
		for (int i = wh; i < 256 - wh; i++) {
			m = 0.0;
			maxim = 1;
			for (int j = i - wh; j <= i + wh; j++) {
				if (fdp[j] > fdp[i]) 
					maxim = 0;
				m += fdp[j];
			}
			m = (double)m / l_fer;
			if (fdp[i] > m + th && maxim)
				vec.push_back(i);

		}
		vec.push_back(255);
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat clone = src.clone();
		int height = src.rows;
		int width = src.cols;

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				int g = src.at<uchar>(i, j);
				int dist = 1000;
				int pixel;
				for (std::vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {

					if (abs(*it - g) < dist) {
						dist = abs(*it - g);
						pixel = *it;
					}
				}
				src.at<uchar>(i, j) = pixel;
			}
		imshow("initial image", clone);
		imshow("nivele gri", src);

		waitKey();
	}

}
void praguri_multiple_erori() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		float *fdp = FDP(fname);
		std::vector<int> vec;
		int wh = 5;
		int l_fer = 2 * wh + 1;
		float th = 0.0003;
		double m;
		bool maxim;
		vec.push_back(0);
		for (int i = wh; i < 256 - wh; i++) {
			m = 0;
			maxim = 1;
			for (int j = i - wh; j <= i + wh; j++) {
				if (fdp[j] > fdp[i])
					maxim = 0;
				m += fdp[j];
			}
			m = m / l_fer;
			if (fdp[i] > m + th && maxim)
				vec.push_back(i);

		}
		vec.push_back(255);
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat clone = src.clone();
		int height = src.rows;
		int width = src.cols;

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				int g = src.at<uchar>(i, j);
				int dist = 1000;
				int pixel;
				for (std::vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {

					if (abs(*it - g) < dist) {
						dist = abs(*it - g);
						pixel = *it;
					}
				}
				
				int error = g - pixel;
				if (isInside(i, j + 1, src))
					src.at<uchar>(i, j + 1) = test_value(src.at<uchar>(i, j + 1) + 7 * error / 16);
				if (isInside(i+1, j - 1, src))
					src.at<uchar>(i+1, j - 1) = test_value(src.at<uchar>(i+1, j - 1) + 3 * error / 16);
				if (isInside(i+1, j , src))
					src.at<uchar>(i+1, j ) = test_value(src.at<uchar>(i+1, j ) + 5 * error / 16);
				if (isInside(i+1, j + 1, src))
					src.at<uchar>(i+1, j + 1) = test_value(src.at<uchar>(i+1, j + 1) +  error / 16);
				
			}
		imshow("initial image", clone);
		imshow("nivele gri", src);

		waitKey();
	}

}
void praguri_multiple_HSV() {

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		float *fdp = FDP(fname);
		std::vector<int> vec;
		int wh = 5;
		int l_fer = 2 * wh + 1;
		float th = 0.0003;
		double m;
		bool maxim;
		vec.push_back(0);
		for (int i = wh; i < 256 - wh; i++) {
			m = 0.0;
			maxim = 1;
			for (int j = i - wh; j <= i + wh; j++) {
				if (fdp[j] > fdp[i])
					maxim = 0;
				m += fdp[j];
			}
			m = (double)m / l_fer;
			if (fdp[i] > m + th && maxim)
				vec.push_back(i);

		}
		vec.push_back(255);
		Mat src =RGB_HSV(fname);
		Mat clone = src.clone();
		int height = src.rows;
		int width = src.cols;

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				Vec3b g = src.at<Vec3b>(i, j);
				int dist = 1000;
				Vec3b pixel = src.at<Vec3b>(i, j);
				for (std::vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {

					if (abs(*it - g[0]) < dist) {
						dist = abs(*it - g[0]);
						pixel[0] = *it;
					}
				}
				src.at<Vec3b>(i, j) = pixel;
			}
		Mat rgb;
		cv::cvtColor(src, rgb, CV_HSV2BGR);
		imshow("initial image", clone);
		imshow("nivele gri", rgb);

		waitKey();
	}

}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Change to aditive\n");
		printf(" 11 - Change to multiply\n");
		printf(" 12 - Build image\n");
		printf(" 13 - inverse of matrix\n");
		printf(" 14 - RGB\n");
		printf(" 15 - convert grayscale\n");
		printf("16 - grayscale_binar\n");
		printf("17 - rgb-hsv\n");
		printf("18 - Histograma\n");
		printf("19 - FDP\n");
		printf("20 - acumulator\n");
		printf("21 - praguri\n");
		printf("22 - praguri_erori\n");
		printf("23 - praguri_HSV\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		String path;
		int res[256];
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				change_to_aditiv();
				break;
			case 11:
				change_to_multiply();
				break;
			case 12:
				image_color();
				break;
			case 13:
				inverseMatrix();
				break;
			case 14:
				matrix_RGB();
				break;
			case 15:
				convert_grayscale();
				break;
			case 16:
				grayscale_binar();
				break;
			case 17:
				
				imshow("", RGB_HSV("Images/Lena_24bits.bmp"));
				waitKey(0);
				break;
			case 18:
				Histogram("Images/cell.bmp");
				break;
			case 19:
				FDP("Images/cell.bmp");
				break;
			case 20:
				acumulator("Images/cell.bmp", 4);
				break;
			case 21:
				praguri_multiple();
				break;
			case 22:
				praguri_multiple_erori();
				break;
			case 23:
				praguri_multiple_HSV();
				break;
				

		}
	}
	while (op!=0);
	return 0;
}
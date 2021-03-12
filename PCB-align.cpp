#include <iostream>
#include <math.h>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp> 
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/core/core.hpp>
typedef unsigned char BYTE;
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d; //只有加上这句命名空间，SiftFeatureDetector and SiftFeatureExtractor 才可以使用

Rect box;//矩形对象
int predefpoint_x, predefpoint_y;
bool drawing_point;//用于第一步，记录是否在标注关键点
bool movemouse;//记录是否拽动鼠标
bool drawing_box;//记录是否在画矩形对象stance(Point pt0, Point pt1)
bool last_button;
bool end_drawing;

int font_face = cv::FONT_HERSHEY_COMPLEX;
double font_scale = 1;
int thickness = 2;
int baseline;

void TemplateMatch(Mat* pTo, Mat pTemplate_raw, Mat* src, Point2f* image1_matchingpoint, Point2f* image2_matchingpoint)
{

	//循环变量
	int i, j, m, n;

	double dSumT; //模板元素的平方和
	double dSumS; //图像子区域元素的平方和
	double dSumST; //图像子区域和模板的点积    

				   //响应值
	double R;

	//记录当前的最大响应
	double MaxR;

	//最大响应出现位置
	int image1_nMaxX;
	int image1_nMaxY;
	int image2_nMaxX;
	int image2_nMaxY;

	int nTplHeight_raw = pTemplate_raw.rows;
	int nTplWidth_raw = pTemplate_raw.cols;
	MaxR = 0;
	for (int k = 2; k < 4; k++) {
		int TplROI_w = nTplWidth_raw / k;
		int x_range = nTplWidth_raw - TplROI_w;

		for (int l = 2; l < 4; l++) {
			int TplROI_h = nTplHeight_raw / l;
			int y_range = nTplHeight_raw - TplROI_h;

			int TplRand_X = rand()%x_range;
			int TplRand_Y = rand()%y_range;

			cout << "Rand-x : " << TplRand_X << "; Rand-y : " << TplRand_Y << endl;
			
			Mat pTemplate;

			pTemplate_raw(Rect(TplRand_X, TplRand_Y, TplROI_w, TplROI_h)).copyTo(pTemplate);
			int nHeight = src->rows;
			int nWidth = src->cols;
			//模板的高、宽
			int nTplHeight = pTemplate.rows;
			int nTplWidth = pTemplate.cols;

			//计算 dSumT
			dSumT = 0;
			for (m = 0; m < nTplHeight; m++)
			{
				for (n = 0; n < nTplWidth; n++)
				{
					// 模板图像第m行，第n个象素的灰度值
					int nGray = *pTemplate.ptr(m, n);

					dSumT += (double)nGray * nGray;
				}
			}

			//找到图像中最大响应的出现位置
			cout << nHeight - nTplHeight << ", " << nWidth - nTplWidth << endl;
			for (i = 0; i < nHeight - nTplHeight + 1; i++)
			{
				for (j = 0; j < nWidth - nTplWidth + 1; j++)
				{
					dSumST = 0;
					dSumS = 0;

					for (m = 0; m < nTplHeight; m++)
					{
						for (n = 0; n < nTplWidth; n++)
						{
							// 原图像第i+m行，第j+n列象素的灰度值
							int nGraySrc = *src->ptr(i + m, j + n);

							// 模板图像第m行，第n个象素的灰度值
							int nGrayTpl = *pTemplate.ptr(m, n);

							dSumS += (double)nGraySrc * nGraySrc;
							dSumST += (double)nGraySrc * nGrayTpl;
						}
					}

					R = dSumST / (sqrt(dSumS) * sqrt(dSumT));//计算相关响应

					//与最大相似性比较
					if (R > MaxR)
					{
						MaxR = R;
						image1_nMaxX = j;
						image1_nMaxY = i;
						image2_nMaxX = TplRand_X;
						image2_nMaxY = TplRand_Y;
					}
				}
			}


			cout << "nMaxX" << image1_nMaxX << ", " << image1_nMaxY << ", " << MaxR << endl;
		}
	}
		
	
	

	//将找到的最佳匹配区域复制到目标图像
	/*for (m = 0; m < nTplHeight; m++)
	{
		for (n = 0; n < nTplWidth; n++)
		{
			int nGray = *src->ptr(nMaxY + m, nMaxX + n);
			//pTo->setTo(nMaxX + n, nMaxY + m, RGB(nGray, nGray, nGray));
			pTo->at<BYTE>(nMaxY + m, nMaxX + n) = nGray;
		}
	}*/


	image1_matchingpoint->x = image1_nMaxX;
	image1_matchingpoint->y = image1_nMaxY;
	image2_matchingpoint->x = image2_nMaxX;
	image2_matchingpoint->y = image2_nMaxY;
	cout << "nMaxX" << image1_nMaxX << ", " << image1_nMaxY <<", "<<MaxR<< endl;
}



static double distance(Point pt0, Point pt1)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	return sqrt(pow(dx1, 2) + pow(dy1, 2));
}

static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

void findGoodmatches(vector<DMatch>& matches, vector<DMatch>& goodMatches, double threshold = 0.95) {
	double min_dist = matches[0].distance, max_dist = matches[0].distance;
	for (int m = 0; m < matches.size(); m++)
	{
		if (matches[m].distance < min_dist)
		{
			min_dist = matches[m].distance;
		}
		if (matches[m].distance > max_dist)
		{
			max_dist = matches[m].distance;
		}
	}
	cout << "min dist=" << min_dist << endl;
	cout << "max dist=" << max_dist << endl;
	//筛选出较好的匹配点
	//vector<DMatch> goodMatches;
	for (int m = 0; m < matches.size(); m++)
	{
		if (matches[m].distance < threshold * max_dist)
		{
			goodMatches.push_back(matches[m]);
		}
	}
	cout << "The number of good matches:" << goodMatches.size() << endl;
	cout << "The number of good matches:" << matches.size() << endl;
}

void onmouse(int event, int x, int y, int flag, void* img)//鼠标事件回调函数，鼠标点击后执行的内容应在此
{
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN://鼠标左键按下事件
		drawing_box = true;//标志在画框
		end_drawing = false;
		movemouse = false;
		last_button = false;
		box = Rect(x, y, 0, 0);//记录矩形的开始的点
		//predefpoint_x = x;
		//predefpoint_y = y;
		break;
	case CV_EVENT_MOUSEMOVE://鼠标移动事件
		if (last_button == false){
			end_drawing = false;
			movemouse = true;
		}
		if (drawing_box) {//如果左键一直按着，则表明在画矩形
			box.width = x - box.x;
			box.height = y - box.y;//更新长宽
		}
		break;
	case CV_EVENT_LBUTTONUP://鼠标左键松开事件
		drawing_box = false;//不在画矩形
		end_drawing = true;
		last_button = true;
		movemouse = false;
		/*if (movemouse == false) {
			end_drawing = true;

		}*/
		
		//这里好像没作用
		if (box.width < 0) {//排除宽为负的情况，在这里判断是为了优化计算，不用再移动时每次更新都要计算长宽的绝对值
			box.x = box.x + box.width;//更新原点位置，使之始终符合左上角为原点
			box.width = -1 * box.width;//宽度取正
		}
		if (box.height < 0) {//同上
			box.y = box.y + box.height;
			box.height = -1 * box.width;
		}
		
		break;
	default:
		break;
	}
}

void onmouse_drawpoint(int event, int x, int y, int flag, void* img) {
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN://鼠标左键按下事件
		//drawing_point = true;//标志在采点
		movemouse = false;
		//box = Rect(x, y, 0, 0);//记录矩形的开始的点
		
		
		break;
	case CV_EVENT_MOUSEMOVE://鼠标移动事件
		drawing_point = false;
		movemouse = true;
		break;
	case CV_EVENT_LBUTTONUP://鼠标左键松开事件
		if (movemouse == false) {
			predefpoint_x = x;
			predefpoint_y = y;
			//return_point = false;//不在画矩形
			drawing_point = true;
		}
		
		break;
	default:
		break;
	}


}



int main(int argc, char** argv) {


	//Create SIFT class pointer
	/*Mat img_1 = imread("test_data/gerber_origin.png", 0);
	Mat img_2 = imread("test_data/IMG_20201211_144430.jpg", 0);
	Mat img_3 = imread("test_data/IMG_20201211_144430.jpg");*/


	Mat img_1 = imread("camera_real_test_20210130/gerber-20210222.png");
	//Mat img_1 = imread("camera_real_test_20210130/sample.jpg", 0);

	//Mat img_2 = imread("camera_real_test_20210130/ori-resize.jpg", 0);
	Mat img_2 = imread("camera_real_test_20210130/ori-resize-20210222.jpg");
	//Mat img_3 = imread("camera_real_test_20210130/ori-resize.jpg");

	//Mat img_2 = imread("camera_real_test_20210130/results.jpg", 0);
	Mat img_3 = imread("camera_real_test_20210130/ori-resize-20210222.jpg");

	imwrite("gray.jpg", img_2);
	cv::resize(img_2, img_2, Size(img_1.cols, img_1.rows));
	cv::resize(img_3, img_3, Size(img_1.cols, img_1.rows));
	Mat temp, temp1;
	bool aritificial = false;
	vector<Point2f> image1Points, image2Points;
	//Mat temp1;
	if (aritificial) {
		//vector<Point2f> image1Points, image2Points;
		cout << "gerber:" << endl;
		namedWindow("Step 1. 鼠标取gerber图像中的匹配点 (最多标注六组点, 按退出键可退出当前界面)", 0);//窗口
		setMouseCallback("Step 1. 鼠标取gerber图像中的匹配点 (最多标注六组点, 按退出键可退出当前界面)", onmouse_drawpoint, &img_1);//注册鼠标事件到“鼠标画个框”窗口，即使在该窗口下出现鼠标事件就执行onmouse函数的内容,最后一个参数为传入的数据。这里其实没有用到
		imshow("Step 1. 鼠标取gerber图像中的匹配点 (最多标注六组点, 按退出键可退出当前界面)", img_1);
		//img.copyTo(temp);放在这里会出现拖影的现象，因为上一次画的矩形并没有被更新
		int num_point_in_gerber = 0;
		string point_descri = "MatchPoint";
		img_1.copyTo(temp1);
		while (1)
		{
			//img.copyTo(temp);//这句话放在if外，注释掉if里面那句.程序没有问题，但每次遍历循环时都会执行一次图像数据的复制传递操作，这是不必要。在高速的PC上没关系，但在嵌入式系统中时，可能会因为硬件性能而无法满足实时需求。因此不建议放这里咯
			//只要不再次按下鼠标左键触发事件,则程序显示的一直是if条件里面被矩形函数处理过的temp图像，如果再次按下鼠标左键就进入if，不断更新被画矩形函数处理过的temp，因为处理速度快所以看起来画矩形的过程是连续的没有卡顿，因为每次重新画都是在没有框的基础上画出新的框因为人眼的残影延迟所以不会有拖影现象。每次更新矩形框的传入数据是重新被img（没有框）的数据覆盖的temp（即img.data==temp.data）和通过回调函数更新了的Box记录的坐标点数据。
			//依照上面所述，则当画完一个矩形后，如果在单击一下鼠标左键(没有拖动),则drawing_box==true,因为Box记录到的坐标点数据计算出来的长宽为0（因为未进行拖动,box.width,box.height为0，则画矩形函数rectangle（）所传入第二第三个参数即对角点的参数两个是相等的，所以矩形的对角线是0就无法画出矩形），则显示的是没有框的原图，此时显示的temp的数据应是和img相等的
			if (drawing_point) {//不断更新正在画的矩形
				//这句放在这里是保证了每次更新矩形框都是在没有原图的基础上更新矩形框。
				//rectangle(temp1, Point(predefpoint_x, predefpoint_y), Point(box.x + box.width, box.y + box.height), Scalar(255, 255, 255));
				string text = point_descri + to_string(num_point_in_gerber) + " in gerber";
				circle(temp1, Point(predefpoint_x, predefpoint_y), 5, Scalar(0, 0, 255), 5);
				putText(temp1, text, Point(predefpoint_x - 10, predefpoint_y), font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 4, 0);
				imshow("Step 1. 鼠标取gerber图像中的匹配点 (最多标注六组点, 按退出键可退出当前界面)", temp1);//显示
				int x = predefpoint_x;
				int y = predefpoint_y;
				num_point_in_gerber++;
				cout << " num = " << num_point_in_gerber << ", (" << x << "," << y << ")" << endl;
				Point2f p_img1(x, y);
				image1Points.push_back(p_img1);
			}

			if (waitKey(30) == 27 || num_point_in_gerber >= 6) {//检测是否有按下退出键
				break;//退出程序
			}
		}
		imwrite("gerber_第一阶段.jpg", temp1);

		cout << "ori:" << endl;
		namedWindow("Step 1. 在ori图像中， 使用鼠标按顺序选取与gerber图像对应的匹配点 (按退出键可退出当前界面)", 0);//窗口
		setMouseCallback("Step 1. 在ori图像中， 使用鼠标按顺序选取与gerber图像对应的匹配点 (按退出键可退出当前界面)", onmouse_drawpoint, &img_3);//注册鼠标事件到“鼠标画个框”窗口，即使在该窗口下出现鼠标事件就执行onmouse函数的内容,最后一个参数为传入的数据。这里其实没有用到
		imshow("Step 1. 在ori图像中， 使用鼠标按顺序选取与gerber图像对应的匹配点 (按退出键可退出当前界面)", img_3);
		//img.copyTo(temp);放在这里会出现拖影的现象，因为上一次画的矩形并没有被更新
		int num_point_in_ori = 0;
		img_3.copyTo(temp1);
		while (1)
		{
			if (drawing_point) {//不断更新正在画的矩形

				string text = point_descri + to_string(num_point_in_ori) + " in ori";
				circle(temp1, Point(predefpoint_x, predefpoint_y), 5, Scalar(0, 0, 255), 5);
				putText(temp1, text, Point(predefpoint_x - 10, predefpoint_y), font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 4, 0);
				imshow("Step 1. 在ori图像中， 使用鼠标按顺序选取与gerber图像对应的匹配点 (按退出键可退出当前界面)", temp1);//显示
				int x = predefpoint_x;
				int y = predefpoint_y;
				num_point_in_ori++;
				cout << " num = " << num_point_in_ori << ", {" << x << "," << y << "}" << endl;

				//Point2f 
				Point2f p_img2(x, y);
				image2Points.push_back(p_img2);
			}

			if (waitKey(30) == 27 || num_point_in_ori >= 6 || num_point_in_ori == num_point_in_gerber) {//检测是否有按下退出键
				break;//退出程序
			}
		}
		imwrite("ori_第一阶段.jpg", temp1);
	}
	else {
		/*int PredefinedKeypoints_num = 12;
		
		int IMG1_PredefinedKeypoints[12][2] = { {172,109},{462,111},{464,1013},{594,1013},{891,1013},{1028,1018},
		{1102,331},{1241,331},{1526,1233},{1964,1238},{1840,241},{2249,239} };

		int IMG2_PredefinedKeypoints[12][2] = { {224,131},{505,129},{493,933},{633,937},{911,933},{1046,935},
		{1117,324},{1257,331},{1526,1132},{1943,1132},{1827,248},{2226,250} };*/
		int PredefinedKeypoints_num = 4;

		//int IMG1_PredefinedKeypoints[4][2] = { {140,91},{332,1256}, {2229,89}, {2422,1254} };

		//int IMG2_PredefinedKeypoints[4][2] = { {200,112},{374,1150},{2211,120},{2387,1155}};

		//int IMG1_PredefinedKeypoints[4][2] = { {137,86},{328,1259}, {2233,86}, {2426,1261} };

		//int IMG2_PredefinedKeypoints[4][2] = { {198,109},{372,1153},{2202,126},{2381,1152} };

		int IMG1_PredefinedKeypoints[4][2] = { {137,85},{336,1251}, {2373,42}, {2424,1264} };

		int IMG2_PredefinedKeypoints[4][2] = { {204,119},{372,1153},{2336,91},{2379,1149} };

		for (int i = 0; i < PredefinedKeypoints_num; i++) {
			Point2f p_img1(IMG1_PredefinedKeypoints[i][0], IMG1_PredefinedKeypoints[i][1]);
			Point2f p_img2(IMG2_PredefinedKeypoints[i][0], IMG2_PredefinedKeypoints[i][1]);

			image1Points.push_back(p_img1);
			image2Points.push_back(p_img2);

		}
	}
	

	//求转换矩阵
	Mat m_homography;
	vector<uchar> m;
	m_homography = findHomography(image1Points, image2Points, RANSAC, 3, m);//寻找匹配图像


	//等待任意按键按下
	//imwrite("match_img.jpg", img_RR_matches);


	Mat image_both = Mat::zeros(img_2.size(), CV_8UC3);
	vector<Point2f> points, points_trans;
	for (int i = 0; i < img_2.rows; i++) {
		for (int j = 0; j < img_2.cols; j++) {
			points.push_back(Point2f(j, i));
		}
	}
	perspectiveTransform(points, points_trans, m_homography);

	for (int i = 0; i < points.size(); i++) {
		int x = points_trans[i].x;
		int y = points_trans[i].y;

		int u = points[i].x;
		int v = points[i].y;

		if (x > 0 && y > 0 && x < img_2.cols && y < img_2.rows) {

			//image_both.at<Vec3b>(v, u)[2] = image_rgb.at<Vec3b>(v, u)[2] / 2 + (image_heat.at<uchar>(y, x)) / 2;
			//image_both.at<Vec3b>(v, u)[1] = image_rgb.at<Vec3b>(v, u)[1] / 2 + (image_heat.at<uchar>(y, x)) / 2;
			//image_both.at<Vec3b>(v, u)[0] = image_rgb.at<Vec3b>(v, u)[0] / 2 + (image_heat.at<uchar>(y, x)) / 2;

			image_both.at<Vec3b>(v, u)[2] = (img_3.at<Vec3b>(y, x)[2]);
			image_both.at<Vec3b>(v, u)[1] = (img_3.at<Vec3b>(y, x)[1]);
			image_both.at<Vec3b>(v, u)[0] = (img_3.at<Vec3b>(y, x)[0]);

		}
	}

	//imwrite("imgout_phase1.jpg", image_both);

	///////////////////////////////////////////////////////////
	//
	// 图像混合
	//
	double alpha = 0.9;
	Mat mixed_image;
	addWeighted(img_1, alpha, image_both, (1 - alpha), 0.0, mixed_image);
	imwrite("imgout_phase1.jpg", mixed_image);
	//namedWindow("mixed_image", 0);
	//imshow("mixed_image", mixed_image);



	/////////////////////////////////////////////////////////


	//vector<Point2f> step1_oripoints_trans;
	//perspectiveTransform(image2Points, step1_oripoints_trans, m_homography);
	namedWindow("Step 2. 选取需要重新匹配的矩形区域，至少选择三个不同的区域，最多选择六个不同的区域。选择完成后，按esc推出界面。", 0);//窗口
	setMouseCallback("Step 2. 选取需要重新匹配的矩形区域，至少选择三个不同的区域，最多选择六个不同的区域。选择完成后，按esc推出界面。", onmouse, &mixed_image);//注册鼠标事件到“鼠标画个框”窗口，即使在该窗口下出现鼠标事件就执行onmouse函数的内容,最后一个参数为传入的数据。这里其实没有用到
	imshow("Step 2. 选取需要重新匹配的矩形区域，至少选择三个不同的区域，最多选择六个不同的区域。选择完成后，按esc推出界面。", mixed_image);
	//img.copyTo(temp);放在这里会出现拖影的现象，因为上一次画的矩形并没有被更新
	//int num = 0;
	int num_rect = 0;
	bool grayAligned = true;
	bool contourAligned = true;
	string rect_descri = "Rectangle";
	mixed_image.copyTo(temp1);
	vector<Point2f> image1Points_phase2, image2Points_phase2;
	while (1)
	{
		//img.copyTo(temp);//这句话放在if外，注释掉if里面那句.程序没有问题，但每次遍历循环时都会执行一次图像数据的复制传递操作，这是不必要。在高速的PC上没关系，但在嵌入式系统中时，可能会因为硬件性能而无法满足实时需求。因此不建议放这里咯
		//只要不再次按下鼠标左键触发事件,则程序显示的一直是if条件里面被矩形函数处理过的temp图像，如果再次按下鼠标左键就进入if，不断更新被画矩形函数处理过的temp，因为处理速度快所以看起来画矩形的过程是连续的没有卡顿，因为每次重新画都是在没有框的基础上画出新的框因为人眼的残影延迟所以不会有拖影现象。每次更新矩形框的传入数据是重新被img（没有框）的数据覆盖的temp（即img.data==temp.data）和通过回调函数更新了的Box记录的坐标点数据。
		//依照上面所述，则当画完一个矩形后，如果在单击一下鼠标左键(没有拖动),则drawing_box==true,因为Box记录到的坐标点数据计算出来的长宽为0（因为未进行拖动,box.width,box.height为0，则画矩形函数rectangle（）所传入第二第三个参数即对角点的参数两个是相等的，所以矩形的对角线是0就无法画出矩形），则显示的是没有框的原图，此时显示的temp的数据应是和img相等的
		mixed_image.copyTo(temp);
		if (drawing_box) {//不断更新正在画的矩形
			//这句放在这里是保证了每次更新矩形框都是在没有原图的基础上更新矩形框。
			rectangle(temp, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 255, 255));
			imshow("Step 2. 选取需要重新匹配的矩形区域，至少选择三个不同的区域，最多选择六个不同的区域。选择完成后，按esc推出界面。", temp);//显示
			
			int x = box.x;
			int y = box.y;
			//num++;
			//cout << " num = " << num << ", (" << x << "," << y << ")" << endl;
		}
		//vector<Point2f> p01, p02;
		
		if (end_drawing == true & movemouse == false) {
			if (box.width >= 2 & box.height >= 2) {
				Mat image1ROI = img_1(Rect(box.x, box.y, box.width, box.height));
				Mat image_bothROI = image_both(Rect(box.x, box.y, box.width, box.height));
				Mat dst1, dst2;
				image1ROI.convertTo(dst1, dst1.type());
				image_bothROI.convertTo(dst2, dst2.type());
				imwrite("./ROI1.jpg", image1ROI);
				imwrite("./ROI2.jpg", image_bothROI);
				Mat image1ROI_gray, image_bothROI_gray;
				cvtColor(image1ROI, image1ROI_gray, COLOR_BGR2GRAY);
				image1ROI_gray = 255 - image1ROI_gray;
				cvtColor(image_bothROI, image_bothROI_gray, COLOR_BGR2GRAY);

				cout << box.x << ", " << box.y << endl;


				////////////////
				/// Step 3.0 灰度相关筛选
				///
				/// 
				/// 
				if ( grayAligned){
					Mat image1ROI_aligned_draw, image2ROI_aligned_draw;

					//threshold(image1ROI_gray, image1ROI_aligned_draw, 60, 255.0, CV_THRESH_BINARY);
					//threshold(image_bothROI_gray, image2ROI_aligned_draw, 60, 255.0, CV_THRESH_BINARY);
					image1ROI_gray.copyTo(image1ROI_aligned_draw);
					image_bothROI_gray.copyTo(image2ROI_aligned_draw);
					//image1ROI_aligned_draw = 255 - image1ROI_aligned_draw;
					Mat pt = image1ROI_aligned_draw;
					Point2f image1_matchingPoint, image2_matchingPoint;

					pt.data = new BYTE[image1ROI_aligned_draw.cols * image1ROI_aligned_draw.rows];
					memset(pt.data, 255, image1ROI_aligned_draw.cols * image1ROI_aligned_draw.rows);
					TemplateMatch(&pt, image2ROI_aligned_draw, &image1ROI_aligned_draw, &image1_matchingPoint, &image2_matchingPoint);
					//image1Points_phase2.push_back(image1_matchingPoint);
					//image2Points_phase2.push_back(image2_matchingPoint);
					image1Points_phase2.push_back(Point2f(image1_matchingPoint.x + box.x, image1_matchingPoint.y + box.y));
					image2Points_phase2.push_back(Point2f(image2_matchingPoint.x + box.x, image2_matchingPoint.y + box.y));
					circle(image1ROI_aligned_draw, Point(image1_matchingPoint.x, image1_matchingPoint.y), 2, Scalar(0, 0, 0), 2);
					circle(image2ROI_aligned_draw, Point(image2_matchingPoint.x, image2_matchingPoint.y), 2, Scalar(0, 0, 0), 2);
					Mat mixed_image_grayaligned;
					addWeighted(image1ROI_aligned_draw, alpha, image2ROI_aligned_draw, (1 - alpha), 0.0, mixed_image_grayaligned);
					string grayAligned_name1 = "./gray/" + to_string(num_rect) + "_1.jpg";
					string grayAligned_name2 = "./gray/" + to_string(num_rect) + "_2.jpg";
					string grayAligned_name3 = "./gray/" + to_string(num_rect) + "_3.jpg";
					imwrite(grayAligned_name1, image1ROI_aligned_draw);
					imwrite(grayAligned_name2, image2ROI_aligned_draw);
					imwrite(grayAligned_name3, mixed_image_grayaligned);
				}
				


				////////////

				if (contourAligned) {
					///////////////////////////////////////////////////
				/// Step3.1 检测出矩形区域中所含多边形的轮廓，然后使用外接最小圆的方法，获得多边形的中心，最后
				///			根据最小圆的圆心进行匹配对齐。
					blur(image1ROI_gray, image1ROI_gray, Size(3, 3));
					blur(image_bothROI_gray, image_bothROI_gray, Size(3, 3));
					int thresh = 100;
					int max_thresh = 255;
					RNG rng(12345);

					Mat threshold_output_gerber, threshold_output_ori;
					vector<vector<Point> > contours_gerber, contours_ori;
					vector<Vec4i> hierarchy_gerber, hierarchy_ori;

					/// 使用Threshold检测边缘
					threshold(image1ROI_gray, threshold_output_gerber, thresh, 255, THRESH_BINARY);
					threshold(image_bothROI_gray, threshold_output_ori, thresh, 255, THRESH_BINARY);
					/// 找到轮廓
					findContours(threshold_output_gerber, contours_gerber, hierarchy_gerber, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
					findContours(threshold_output_ori, contours_ori, hierarchy_ori, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

					vector<vector<Point> > contours_poly_gerber(contours_gerber.size()), contours_poly_ori(contours_ori.size());
					vector<Rect> boundRect(contours_gerber.size());
					//vector<Point2f>center_gerber, center_ori;
					//vector<float>radius_gerber, radius_ori;
					//vector<int> contours_index_gerber, contours_index_ori;

					Mat drawing_gerber = Mat::zeros(threshold_output_gerber.size(), CV_8UC3);
					Mat ROI1_colne = image1ROI_gray.clone();
					Mat ROI2_colne = image_bothROI_gray.clone();


					for (int i = 0; i < contours_gerber.size(); i++) {

						approxPolyDP(Mat(contours_gerber[i]), contours_poly_gerber[i], 3, true);
						Rect tmp_rect_gerber = boundingRect(Mat(contours_poly_gerber[i]));
						float max_similarity = 0.0;

						Point2f tmp_gerber_point;
						float tmp_radius;
						Point2f best_matching_point;
						minEnclosingCircle(contours_poly_gerber[i], tmp_gerber_point, tmp_radius);
						int index;
						float radius_ori, ther_similarity = 1;
						if (tmp_rect_gerber.size() != image1ROI_gray.size() & tmp_rect_gerber.width >= 3 & tmp_rect_gerber.height >= 3)
							//if ( tmp_rect_gerber.width >= 3 & tmp_rect_gerber.height >= 3)
						{
							for (int j = 0; j < contours_ori.size(); j++) {

								approxPolyDP(Mat(contours_ori[j]), contours_poly_ori[j], 3, true);
								Point2f tmp_ori_point;

								minEnclosingCircle(contours_poly_ori[j], tmp_ori_point, tmp_radius);

								double tmp_similarity = matchShapes(contours_gerber[i], contours_ori[j], CV_CONTOURS_MATCH_I1, 0);// +  distance(tmp_ori_point, tmp_gerber_point);
								cout << "similarity = " << tmp_similarity << " in " << num_rect << endl;
								if (tmp_similarity > max_similarity & tmp_similarity < 10000 & tmp_similarity > ther_similarity) {

									max_similarity = tmp_similarity;
									best_matching_point = tmp_ori_point;

									index = j;
									radius_ori = tmp_radius;

								}

							}
							if (max_similarity > ther_similarity) {
								cout << tmp_gerber_point.x + box.x << ", " << tmp_gerber_point.y + box.y << endl;
								image1Points_phase2.push_back(Point2f(tmp_gerber_point.x + box.x, tmp_gerber_point.y + box.y));
								image2Points_phase2.push_back(Point2f(best_matching_point.x + box.x, best_matching_point.y + box.y));
								Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
								drawContours(ROI1_colne, contours_poly_gerber, i, color, 1, 8, vector<Vec4i>(), 0, Point());
								drawContours(ROI2_colne, contours_poly_ori, index, color, 1, 8, vector<Vec4i>(), 0, Point());
								circle(ROI1_colne, tmp_gerber_point, (int)tmp_radius, color, 2, 8, 0);
								circle(ROI2_colne, best_matching_point, (int)radius_ori, color, 2, 8, 0);


							}

						}

					}
					string imageROI1_Contours = "./contours/ROI1_Contours_in_rect_" + to_string(num_rect) + ".jpg";
					string imageROI2_Contours = "./contours/ROI2_Contours_in_rect_" + to_string(num_rect) + ".jpg";
					imwrite(imageROI1_Contours, ROI1_colne);
					imwrite(imageROI2_Contours, ROI2_colne);

				}
				
				
				/// ////////////////////////////////////////////////////
				
				/// <param name="argc"></param>
				/// <param name="argv"></param>
				/// <returns></returns>
				/*
				int minHessian = 40;
				Ptr<Feature2D> f2d = xfeatures2d::SURF::create(minHessian);
				
				vector<KeyPoint> keypoints_1, keypoints_2;
				f2d->detect(image1ROI_gray, keypoints_1);
				f2d->detect(image_bothROI_gray, keypoints_2);

				Mat descriptors_1, descriptors_2;
				f2d->compute(image1ROI_gray, keypoints_1, descriptors_1);
				f2d->compute(image_bothROI_gray, keypoints_2, descriptors_2);

				BFMatcher matcher(NORM_L2, true);
				vector<DMatch> matches, goodMatches;
				matcher.match(descriptors_1, descriptors_2, matches);
				if (matches.size() >= 1) {
					findGoodmatches(matches, goodMatches, 0.8);
				}
				if (goodMatches.size() < 1) {
					goodMatches = matches;
				}
				
				Mat first_match;
				drawMatches(image1ROI_gray, keypoints_1, image_bothROI_gray, keypoints_2, goodMatches, first_match);
				string mathch_name = "matches_" + to_string(num_rect) + ".jpg";

				imwrite(mathch_name, first_match);
				cout << "size = " << keypoints_1.size() << ", " << keypoints_2.size()<<endl;
				
				if (goodMatches.size()>=1) {
					vector<DMatch> m_Matches;
					m_Matches = goodMatches;

					int ptCount = goodMatches.size();
					if (ptCount < 0)
					{
						cout << "Can't find enough match points" << endl;
						return 0;
					}

					vector <KeyPoint> RAN_KP1, RAN_KP2;
					//size_t是标准C库中定义的，应为unsigned int，在64位系统中为long unsigned int,在C++中为了适应不同的平台，增加可移植性。
					for (size_t i = 0; i < m_Matches.size(); i++)
					{
						RAN_KP1.push_back(keypoints_1[goodMatches[i].queryIdx]);
						RAN_KP2.push_back(keypoints_2[goodMatches[i].trainIdx]);
						//RAN_KP1是要存储img01中能与img02匹配的点
						//goodMatches存储了这些匹配点对的img01和img02的索引值

					}

					
					for (size_t i = 0; i < m_Matches.size(); i++)
					{
						//double minRelativeDistance = 10000.0;
						Point2f p01(RAN_KP1[i].pt.x + box.x, RAN_KP1[i].pt.y + box.y);
						Point2f p02(RAN_KP2[i].pt.x + box.x, RAN_KP2[i].pt.y + box.y);
						image1Points_phase2.push_back(p01);
						image2Points_phase2.push_back(p02);
					}
				}*/

				string text = rect_descri + to_string(num_rect);
				rectangle(temp1, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(0, 0, 255), 3);
				putText(temp1, text, Point(box.x - 20, box.y - 10), font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 4, 0);
				num_rect++;

				


			}
			imshow("Step 2. 选取需要重新匹配的矩形区域，至少选择三个不同的区域，最多选择六个不同的区域。选择完成后，按esc推出界面。", temp1);//显示
			end_drawing = false;
			movemouse = true;
		}
		
		if (waitKey(30) == 27 || num_rect>=15) {//检测是否有按下退出键
			break;//退出程序
		}
	}
	
	imwrite("rect_第二阶段.jpg", temp1);
	////////////////////////////////////////////////////////////////
	//
	// Phase 2
	//
	///////////////////////////////////////////////////////////////
	/*vector<Point2f> image1Points_phase2, image2Points_phase2;
	int IMG1_PredefinedKeypoints_phase2[12][2] = { {109,48},{141,88},{435,555},{401,791},{190,1296},
	{2370,44},{2235,88},{2348,326},{2128,784},{2348,903},{2419,1251},{2446,1296} };

	int IMG2_PredefinedKeypoints_phase2[12][2] = { {121,53},{152,93},{444,564},{413,796},{215,1301},
	{2399,30},{2251,75},{2374,317},{2141,789},{2361,915},{2433,1269},{2460,1312} };

	for (int i = 0; i < PredefinedKeypoints_num; i++) {
		Point2f p_img1(IMG1_PredefinedKeypoints_phase2[i][0], IMG1_PredefinedKeypoints_phase2[i][1]);
		Point2f p_img2(IMG2_PredefinedKeypoints_phase2[i][0], IMG2_PredefinedKeypoints_phase2[i][1]);

		image1Points_phase2.push_back(p_img1);
		image2Points_phase2.push_back(p_img2);

	}*/

	cout <<"After contours : "<< image1Points_phase2.size() << ", " << image2Points_phase2.size() << endl;
	for (int i = 0; i < image1Points.size(); i++) {
		image1Points_phase2.push_back(image1Points[i]);
		image2Points_phase2.push_back(image1Points[i]);
		//cout << image1Points[i] << step1_oripoints_trans[i] << endl;
	}
	
	cout << "After adding step1-points : " << image1Points_phase2.size() << ", " << image2Points_phase2.size() << endl;
	Mat m_homography_phase2;
	vector<uchar> m_phase2;
	m_homography_phase2 = findHomography(image1Points_phase2, image2Points_phase2, RANSAC, 3, m_phase2);




	Mat image_both_phase2 = Mat::zeros(img_2.size(), CV_8UC3);
	vector<Point2f> points_phase2, points_trans_phase2;
	for (int i = 0; i < img_2.rows; i++) {
		for (int j = 0; j < img_2.cols; j++) {
			points_phase2.push_back(Point2f(j, i));
		}
	}
	perspectiveTransform(points_phase2, points_trans_phase2, m_homography_phase2);

	for (int i = 0; i < points_phase2.size(); i++) {
		int x = points_trans_phase2[i].x;
		int y = points_trans_phase2[i].y;

		int u = points_phase2[i].x;
		int v = points_phase2[i].y;

		if (x > 0 && y > 0 && x < img_2.cols && y < img_2.rows) {



			image_both_phase2.at<Vec3b>(v, u)[2] = (image_both.at<Vec3b>(y, x)[2]);
			image_both_phase2.at<Vec3b>(v, u)[1] = (image_both.at<Vec3b>(y, x)[1]);
			image_both_phase2.at<Vec3b>(v, u)[0] = (image_both.at<Vec3b>(y, x)[0]);

		}
	}
	
	//double alpha = 0.9;
	//Mat mixed_image;

	addWeighted(img_1, alpha, image_both_phase2, (1 - alpha), 0.0, mixed_image);
	if (grayAligned) {
		imwrite("imgout_phase2_mixed_grayaligned.jpg", mixed_image);
	}
	else {
		imwrite("imgout_phase2_mixed.jpg", mixed_image);
	}
	

	imwrite("imgout_phase2_ori.jpg", image_both_phase2);


	namedWindow("Step 3.", 0);
	imshow("Step 3.", mixed_image);
	waitKey(0);
	cout << "Finished!" << endl;
	//namedWindow();




}
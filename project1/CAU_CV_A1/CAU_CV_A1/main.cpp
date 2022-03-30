#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>
#include <queue>
#include <vector>
#include <string>
#define GRAY 0
#define RGB 1
#define GRADIENT 2

using namespace cv;
using namespace std;

Point ptOld;
const int number_bins = 9;
void on_mouse(int event, int x, int y, int flags, void*);
void show_img(void*);
void show_img_final();
void plot_histogram(int x, int y, void*, int method);
void get_histograms(void*);
void get_histogramsRGB(void*);
void get_histogramsGRAD(void*);
void match_histogram(int method);

map<Mat*, Mat*> imgMap;
map<Mat*, int> recMap;
map<Mat*, string> winMap;
map<Mat*, vector<Point>> pntMap;
map<Mat*, bool> chkMap;
map<Mat*, vector<MatND> > histMap;
map<Mat*, vector<MatND> > histMapGrad;
map<Mat*, vector< vector<MatND> > > histMapRGB;
map<Mat*, vector< map<Mat*, int> > > matchMap;

map<Mat*, Scalar> colorMap;
map<string, int> methodMap =
{ { "GRAY", 0}, {"RGB", 1}, {"GRAD", 2} };

template<typename ... Args>

std::string string_format(const std::string& format, Args ... args)
{
    size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0' 
    if (size <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside }
}

int main(int ac, char** av) {

    Mat img1 = imread("1st.jpg");
	Mat img2 = imread("2nd.jpg");

    if (img1.empty() || img2.empty()) {
        cerr << "Image load failed!" << endl;
        return -1;
    }

    Mat img_resize_1, img_resize_2;

    cout << img1.rows<< " " << img1.cols<< " " << (float)img1.rows/(float)img1.cols << endl;

    resize(img1, img_resize_1, Size(), 0.1, 0.1, INTER_AREA);
    resize(img2, img_resize_2, Size(), 0.1, 0.1, INTER_AREA);


    namedWindow("img1");
    moveWindow("img1", 100, 100);
    winMap[&img_resize_1] = "img1";
    chkMap[&img_resize_1] = false;
    colorMap[&img_resize_1] = Scalar(255, 0, 0);
    namedWindow("img2");
    moveWindow("img2", 500, 100);
    winMap[&img_resize_2] = "img2";
    chkMap[&img_resize_2] = false;
    colorMap[&img_resize_2] = Scalar(0, 0, 255);

    setMouseCallback("img1", on_mouse, &img_resize_1);
    setMouseCallback("img2", on_mouse, &img_resize_2);
    show_img(&img_resize_1);
    show_img(&img_resize_2);

    //종료
    while (!chkMap[&img_resize_1] || !chkMap[&img_resize_2]) {
        waitKey(10);
    }
    // 히스토그램 분석 시작

    get_histograms(&img_resize_1);
    get_histograms(&img_resize_2);
    cout << "his" << endl;
    get_histogramsRGB(&img_resize_1);
    get_histogramsRGB(&img_resize_2);
    cout << "hisRGB" << endl;
    get_histogramsGRAD(&img_resize_1);
    get_histogramsGRAD(&img_resize_2);
    cout << "hisGRAD" << endl;

    cout << "Histogram method(GRAY, RGB, GRAD)" << endl;
    string method;
    cin >> method;
    for (int i = 0; i < method.size(); i++) {
        method[i] = toupper(method[i]);
    }

    plot_histogram(100, 100, &img_resize_1, methodMap[method]);
    plot_histogram(100, 430, &img_resize_2, methodMap[method]);
    cout << "plothis" << endl;
    match_histogram(methodMap[method]);
    show_img_final();
	imshow("img1", img_resize_1);
	imshow("img2", img_resize_2);


    Mat img3;
    hconcat(img_resize_1, img_resize_2, img3);

    namedWindow("img3");
    moveWindow("img3", 500, 100);
    imshow("img3", img3);
	waitKey(0);

	return 0;
}
float distance(MatND h1, MatND h2) {
    float l = 0;
    for (int i = 0; i < number_bins; i++) {
        l += h1.at<float>(i) * h1.at<float>(i) - h2.at<float>(i) * h2.at<float>(i);
    }
    l *= l < 0 ? -1 : 1;
    return l;
}

int match_his(void* his, void* param, int method=GRAY) {
    cv::Mat* img = (cv::Mat*) param;
    vector<cv::MatND> hists;
    cv::MatND hist(0, 0, CV_8SC1);

    if (method==RGB)
        hists = *(vector<cv::MatND>*) his;
    else
        hist = *(cv::MatND*) his;

    float min_dis = numeric_limits<float>::max();
    int m = -1;

    for (int i = 0; i < recMap[img]; i++) {
        MatND histogram = histMap[img][i];
        float l;
        switch (method) {
        case(GRAY):
            l = distance(histMap[img][i], hist);
            break;
        case(RGB):
            l = distance(histMapRGB[img][i][0], hists[0]) + distance(histMapRGB[img][i][1], hists[1]) + distance(histMapRGB[img][i][2], hists[2]);
            break;
        case(GRADIENT):
            l = distance(histMapGrad[img][i], hist);
            break;
        }
        printf("%d %f\n", i, l);
        if (l < min_dis) {
            min_dis = l;
            m = i;
        }

    }
    return m;
}
void match_histogram(int method) {
    map<Mat*, string>::iterator jt;
    map<Mat*, string>::iterator it;
    for (it = winMap.begin(); it != winMap.end(); it++){
        vector<map<Mat*, int> > t(4);

        for (jt = it; jt != winMap.end(); jt++) {
            if (it == jt) continue;
            for (int i = 0; i < recMap[it->first]; i++) {

                switch (method) {
                case GRAY:
                {
                    MatND histogram = histMap[it->first][i];
                    t[i][jt->first] = match_his(&histogram, jt->first, GRAY);
                    break;
                }
                case RGB:
                {
                    vector<MatND> histogram = histMapRGB[it->first][i];
                    t[i][jt->first] = match_his(&histogram, jt->first, RGB);
                    break;
                }
                case GRADIENT:
                {
                    MatND histogram = histMap[it->first][i];
                    t[i][jt->first] = match_his(&histogram, jt->first, GRADIENT);
                    break;
                }
                }
                
            }
        }

         matchMap[it->first] = t;
    }
}
void get_histograms(void* param) {
    cv::Mat* img = (cv::Mat*) param;

    Mat img_copy;
    img->copyTo(img_copy);

    cvtColor(img_copy, img_copy, COLOR_BGR2GRAY);
    copyMakeBorder(img_copy, img_copy, 4, 4, 4, 4, BORDER_CONSTANT, Scalar(0, 0, 0));
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;

    for (int i = 0; i < recMap[img]; i++) {
        MatND histogram;
        Point pnt = pntMap[img][i];
        int x = pnt.x;
        int y = pnt.y;

        Mat img2 = img_copy(Range(y - 4, y + 4), Range(x - 4, x + 4));
        
        calcHist(&img2, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);
        histMap[img].push_back(histogram);
    }

}

void get_histogramsGRAD(void* param) {
    cv::Mat* img = (cv::Mat*) param;

    Mat img_copy;
    img->copyTo(img_copy);

    cvtColor(img_copy, img_copy, COLOR_BGR2GRAY);
    copyMakeBorder(img_copy, img_copy, 4, 4, 4, 4, BORDER_CONSTANT, Scalar(0, 0, 0));
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;

    for (int i = 0; i < recMap[img]; i++) {
        MatND histogram = Mat(9, 1, CV_32F, Scalar::all(0));
        Point pnt = pntMap[img][i];
        int x = pnt.x;
        int y = pnt.y;

        Mat img2 = img_copy(Range(y - 4, y + 4), Range(x - 4, x + 4));
        Mat gx, gy;
        Mat mag, ang;
        Sobel(img2, gx, CV_32F, 1, 0);
        Sobel(img2, gy, CV_32F, 0, 1);
        cartToPolar(gx, gy, mag, ang, true);
        cout << mag << endl;
        cout << ang << endl;
        for (int x = 0; x<ang.cols; x++)
            for (int y = 0; y < ang.rows; y++) {
                float a = ang.at<float>(x, y);
                a = a >= 180 ? a - 180 : a;
                int index = (int)(a / 20);
                cout << index << endl;
                histogram.at<float>(index) += (a - index * 20) / 20 * mag.at<float>(x, y);
                histogram.at<float>((index + 1) % 9) += ((index + 1) * 20 - a) / 20 * mag.at<float>(x, y);
            }
        cout << histogram<<endl;
        histMapGrad[img].push_back(histogram);
    }

}

void get_histogramsRGB(void* param) {
    cv::Mat* img = (cv::Mat*) param;
    MatND histogram;

    Mat img_copy;
    img->copyTo(img_copy);
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;

    for (int i = 0; i < recMap[img]; i++) {
        Point pnt = pntMap[img][i];
        int x = pnt.x;
        int y = pnt.y;

        Mat img2 = img_copy(Range(y - 4, y + 4), Range(x - 4, x + 4));


        MatND histogramB, histogramG, histogramR;
        int channel_B[] = { 0 };  // Blue
        int channel_G[] = { 1 };  // Green
        int channel_R[] = { 2 };  // Red
        float channel_range[2] = { 0.0 , 255.0 };
        const float* channel_ranges[1] = { channel_range };
        int histSize[1] = { 256 };

        // R, G, B별로 각각 히스토그램을 계산한다.
        calcHist(&img2, 1, channel_B, Mat(), histogramB, 1, histSize, channel_ranges);
        calcHist(&img2, 1, channel_G, Mat(), histogramG, 1, histSize, channel_ranges);
        calcHist(&img2, 1, channel_R, Mat(), histogramR, 1, histSize, channel_ranges);
        vector<MatND> his = { histogramB, histogramG, histogramR };
        histMapRGB[img].push_back(his);
    }

}

void plot_histogram(int x, int y, void* param, int method) {
    cv::Mat* img = (cv::Mat*) param;

    int hist_w = 300;
    int hist_h = 300;
    int bin_w = cvRound((double)hist_w / number_bins);
    cout << bin_w << endl;
    for (int i = 0; i < recMap[img]; i++) {
        MatND histogram;
        vector<MatND> histogramRGB;
        switch (method) {
        case(GRAY):
            histogram = histMap[img][i];
        case(RGB):
            histogramRGB = histMapRGB[img][i];
        case(GRADIENT):
            histogram = histMapGrad[img][i];
        }
        Mat hist_img(hist_h, hist_w, CV_8UC3, Scalar::all(0));
        normalize(histogram, histogram, 0, hist_img.rows/2, NORM_MINMAX, -1, Mat());

        for (int j = 0; j < number_bins; j++)
        {
            //Point p1 = Point(bin_w * (j - 1), hist_h - cvRound(histogram.at<float>(j - 1)));
            //Point p2 = Point(bin_w * (j), hist_h - cvRound(histogram.at<float>(j)));

            if (method != RGB) {
                Point p1 = Point(bin_w * (j ), hist_h);
                Point p2 = Point(bin_w * (j +1 ), hist_h - cvRound(histogram.at<float>(j)));
                rectangle(hist_img, p1, p2, Scalar(255, 0, 0), 1, 8, 0);
            }
            else {
                for (int k = 0; k < 3; k++) {
                    Point p1 = Point(bin_w * (j + 1 / 3 * k), hist_h);
                    Point p2 = Point(bin_w * (j + 1 / 3 * (k + 1)), hist_h - cvRound(histogramRGB[k].at<float>(j)));
                    rectangle(hist_img, p1, p2, Scalar(k == 2 ? 255 : 0, k == 1 ? 255 : 0, k == 0 ? 255 : 0), 1, 8, 0);
                }
            }
        }
        putText(hist_img, string_format("%d", i + 1), Point(30, 30), FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, LINE_AA);
        namedWindow(string_format("%s (%d/4)", winMap[img].c_str(), i + 1));
        moveWindow(string_format("%s (%d/4)", winMap[img].c_str(), i + 1), x+ hist_w *i, y);
        imshow(string_format("%s (%d/4)", winMap[img].c_str(), i+1), hist_img);
    }

}

void show_img(void* param) {
    cv::Mat* img = (cv::Mat*) param;

    Mat img_copy;
    img->copyTo(img_copy);

    putText(img_copy, string_format("Left to rect Right to fix (%d/4)", recMap[img]), Point(0, img_copy.rows - 20), FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, LINE_AA);
    if (chkMap[img]) putText(img_copy, "Fixed", Point(0, img_copy.rows - 10), FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, LINE_AA);

    for (int i = 0; i < recMap[img]; i++) {
        Point pnt = pntMap[img][i];
        int x = pnt.x;
        int y = pnt.y;



        rectangle(img_copy, Rect(Point(x + 4, y + 4), Point(x - 4, y - 4)), colorMap[img], 1, 8, 0);
        if (x + 20 > img->cols) x -= 24;
        if (y - 20 < 0) y += 24;
        putText(img_copy, to_string(i + 1), Point(x, y), FONT_HERSHEY_PLAIN, 2, colorMap[img], 1, LINE_AA);
    }
    imshow(winMap[img], img_copy);
}

void show_img_final() {
    Mat img_copy, img_final;

    map<Mat*, string>::iterator it = winMap.begin();

    it->first->copyTo(img_final);
    it++;

    map<Mat*, Mat*>::iterator jt;
    for (it; it != winMap.end(); it++) {
        it->first->copyTo(img_copy);
        hconcat(img_final, img_copy, img_final);
    }

    map<Mat*, vector<Point> >::iterator pt = pntMap.begin();
    int cols = 0;
    for (pt; pt != pntMap.end(); pt++) {
        for (int i = 0; i < recMap[pt->first]; i++) {
            Point pnt = pt->second[i];
            int x = pnt.x;
            int y = pnt.y;

            rectangle(img_final, Rect(Point(x + cols + 4, y + 4), Point(x + cols - 4, y - 4)), colorMap[pt->first], 1, 8, 0);
            if (x + 20 > pt->first->cols + cols) x -= 24;
            if (y - 20 < 0) y += 24;
            putText(img_final, to_string(i + 1), Point(x+cols, y), FONT_HERSHEY_PLAIN, 2, colorMap[pt->first], 1, LINE_AA);

        }
        cols += pt->first->cols;
    }
    map<Mat*, vector<Point> >::iterator pt1 = pntMap.begin();
    map<Mat*, vector<Point> >::iterator pt2 = pntMap.begin();
    cols = 0;
    for (pt1; pt1 != pntMap.end(); pt1++) {
        for (int i = 0; i < recMap[pt1->first]; i++){
            int cols1 = 0;
            for (pt2 = pntMap.begin(); pt2 != pntMap.end(); pt2++) {
                if (pt1 != pt2) {
                    int j = matchMap[pt1->first][i][pt2->first];
                    line(img_final, Point(pt1->second[i].x + cols, pt1->second[i].y+4), Point(pt2->second[j].x + cols1, pt2->second[j].y-4), colorMap[pt1->first]);
                }
                cols1 += pt2->first->cols;
            }
        }
        cols += pt1->first->cols;
    }

    imshow("final", img_final);
}
void on_mouse(int event, int x, int y, int flags, void* param)
{
    static int n = 0;

    cv::Mat* img = (cv::Mat*) param;
    switch (event) {
    case EVENT_LBUTTONDOWN:
        if (!chkMap[img]) {
            ptOld = Point(x, y);
            if (recMap[img] < 4) {
                pntMap[img].push_back(ptOld);

                cout << "EVENT_LBUTTONDOWN" << n << ": " << x << ", " << y << endl;

                recMap[img]++;
            }
            else {
                pntMap[img].push_back(ptOld);
                pntMap[img].erase(pntMap[img].begin());
            }
            show_img(param);    
        }
        break;
    case EVENT_RBUTTONDOWN:
        if (recMap[img]==4) chkMap[img] = true;
        cout << "EVENT_RBUTTONUP: " << x << ", " << y << endl;
        show_img(param);

        break;
    default:
        break;
    }
}
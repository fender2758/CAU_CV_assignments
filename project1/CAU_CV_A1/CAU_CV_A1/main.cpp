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
#define HOG 3

using namespace cv;
using namespace std;

Point ptOld;
const int number_bins = 9;
const int patch_size = 4;

void on_mouse(int event, int x, int y, int flags, void*);   //마우스 클릭에 반응 함수
void show_img(void*);           //이미지 출력(패치, 숫자, 설명 등)
void show_img_final();          //붙여놓은 최종 이미지 출력
void plot_histogram(int x, int y, void*, int method);   //히스토그램 출력
void get_histograms(void*);     //히스토그램 (GRAY)
void get_histogramsRGB(void*);  //히스토그램 (RGB)
void get_histogramsGRAD(void*); //히스토그램 (SIFT)
void get_histogramsHOG(void*);  //히스토그램 (HOG)
void match_histogram(int method);//히스토그램 매칭(방법별)

map<Mat*, int> recMap;          //이미지에 현재 패치가 몇개 찍혔는지 (최대4)
map<Mat*, string> winMap;       //이미지가 어느 윈도우에 출력되는지
map<Mat*, vector<Point>> pntMap;//패치들의 포인트
map<Mat*, bool> chkMap;         //확정 지었는지
map<Mat*, vector<MatND> > histMap;      //히스토그램(GRAY)
map<Mat*, vector<MatND> > histMapGrad;  //히스토그램(SIFT)
map<Mat*, vector<MatND> > histMapHOG;   //히스토그램(HOG)
map<Mat*, vector< vector<MatND> > > histMapRGB; //히스토그램(RGB)
map<Mat*, vector< map<Mat*, int> > > matchMap;  //히스토그램 매칭 결과

map<Mat*, Scalar> colorMap;
map<string, int> methodMap =
{ { "GRAY", 0}, {"RGB", 1}, {"SIFT", 2}, {"HOG", 3} };

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

    //이미지 입력
    Mat img1 = imread("1st.jpg");
    Mat img2 = imread("2nd.jpg");
    //Mat img2 = imread("1st.jpg");

    //실패시 코드
    if (img1.empty() || img2.empty()) {
        cerr << "Image load failed!" << endl;
        return -1;
    }

    Mat img_resize_1, img_resize_2;
    //이미지 축소
    resize(img1, img_resize_1, Size(), 0.2, 0.2, INTER_AREA);
    resize(img2, img_resize_2, Size(), 0.2, 0.2, INTER_AREA);
    imwrite("img_resize_1.bmp", img_resize_1);
    imwrite("img_resize_2.bmp", img_resize_2);

    //이미지 표시
    namedWindow("img1");
    moveWindow("img1", 100, 100);
    winMap[&img_resize_1] = "img1";
    chkMap[&img_resize_1] = false;
    colorMap[&img_resize_1] = Scalar(255, 0, 0);
    namedWindow("img2");
    moveWindow("img2", 900, 100);
    winMap[&img_resize_2] = "img2";
    chkMap[&img_resize_2] = false;
    colorMap[&img_resize_2] = Scalar(0, 0, 255);

    //마우스 함수 설정
    setMouseCallback("img1", on_mouse, &img_resize_1);
    setMouseCallback("img2", on_mouse, &img_resize_2);
    show_img(&img_resize_1);
    show_img(&img_resize_2);

    //종료
    while (!chkMap[&img_resize_1] || !chkMap[&img_resize_2]) {
        waitKey(10);
    }
    // 히스토그램 분석 시작
    //실험을 위한 값들
    //pntMap[&img_resize_1] = { Point(27, 382), Point(274, 147), Point(597, 484), Point(345, 722) };
    //pntMap[&img_resize_2] = { Point(156, 222), Point(513, 221), Point(510, 729), Point(133, 712) };

    //pntMap[&img_resize_1] = { Point(184, 364), Point(370, 652), Point(452, 504), Point(274, 146) };
    //pntMap[&img_resize_2] = { Point(283, 319), Point(205, 676), Point(377, 630), Point(450, 222) };

    //pntMap[&img_resize_1] = { Point(93, 380), Point(252, 255), Point(471, 474), Point(286, 669) };
    //pntMap[&img_resize_2] = { Point(207, 268), Point(414, 287), Point(421, 621), Point(129, 488) };


    //pntMap[&img_resize_2] = { Point(27, 382), Point(273, 147), Point(597, 484), Point(345, 722) };

    //각 방법에 대한 히스토그램
    get_histograms(&img_resize_1);
    get_histograms(&img_resize_2);
    cout << "his" << endl;
    get_histogramsRGB(&img_resize_1);
    get_histogramsRGB(&img_resize_2);
    cout << "hisRGB" << endl;
    get_histogramsGRAD(&img_resize_1);
    get_histogramsGRAD(&img_resize_2);
    cout << "hisGRAD" << endl;
    get_histogramsHOG(&img_resize_1);
    get_histogramsHOG(&img_resize_2);
    cout << "hisHOG" << endl;

    //히스토그램 방법 정하기
    cout << "Histogram method(GRAY, RGB, SIFT, HOG)" << endl;
    string method;
    cin >> method;
    for (int i = 0; i < method.size(); i++) {
        method[i] = toupper(method[i]);
    }
    //방법에 따른 히스토그램 출력
    plot_histogram(100, 100, &img_resize_1, methodMap[method]);
    plot_histogram(100, 440, &img_resize_2, methodMap[method]);
    cout << "plothis" << endl;
    //각 패치 연결
    match_histogram(methodMap[method]);
    //최종 이미지 출력
    show_img_final();

	waitKey(0);

	return 0;
}
float distance(MatND h1, MatND h2) {
    MatND his1, his2;
    //히스토그램 정규화
    normalize(h1, his1, 100, 0, NORM_L2, -1, Mat());
    normalize(h2, his2, 100, 0, NORM_L2, -1, Mat());
    float l = 0;
    //L2 거리
    for (int i = 0; i < number_bins; i++) {
        //cout << l << " ";
        l += (h1.at<float>(i) - h2.at<float>(i)) * (h1.at<float>(i) - h2.at<float>(i));
        //l += (his1.at<float>(i) - his2.at<float>(i)) * (his1.at<float>(i) - his2.at<float>(i));
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
        case(HOG):
            //HOG의 경우 20도씩 회전하며 비교
            l = numeric_limits<float>::max();
            for (int j = 0; j < 9; j++) {
                Mat temp_his;
                temp_his = histMapHOG[img][i](Range(j, 9), Range(0, 1));
                temp_his.push_back(histMapHOG[img][i](Range(0, j), Range(0, 1)));

                float temp_l = distance(temp_his, hist);
                cout << i<<" "<<j<< " "<<temp_l << endl;
                if (l > temp_l) l = temp_l;
            }
            break;
        }
        printf("%d %f\n\n", i, l);
        //최소 거리 패치를 구한다
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
    //각 이미지별 다른 이미지에 가장 비슷한 패치 탐색(BRUTE FORCE)
    for (it = winMap.begin(); it != winMap.end(); it++){
        vector<map<Mat*, int> > t(4);

        for (jt = winMap.begin(); jt != winMap.end(); jt++) {
            if (it == jt) continue;
            for (int i = 0; i < recMap[it->first]; i++) {
                cout << i << endl;
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
                    MatND histogram = histMapGrad[it->first][i];

                    t[i][jt->first] = match_his(&histogram, jt->first, GRADIENT);
                    break;
                }
                case HOG:
                {
                    MatND histogram = histMapHOG[it->first][i];

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
    //패치에 의해 잘릴 수 있는 border 설정
    copyMakeBorder(img_copy, img_copy, patch_size, patch_size, patch_size, patch_size, BORDER_CONSTANT, Scalar(0, 0, 0));
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;

    for (int i = 0; i < recMap[img]; i++) {
        MatND histogram;
        Point pnt = pntMap[img][i];
        int x = pnt.x;
        int y = pnt.y;
        cout << x << " " << y << endl;
        Mat img2 = img_copy(Range(y, y + 2 * patch_size), Range(x, x + 2 * patch_size));
        //패치 크기에 따라 히스토그램 계산
        calcHist(&img2, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);
        //cout << histogram;
        histMap[img].push_back(histogram);
    }
    cout << endl;

}

void get_histogramsHOG(void* param) {
    cv::Mat* img = (cv::Mat*) param;

    Mat img_copy;
    img->copyTo(img_copy);

    cvtColor(img_copy, img_copy, COLOR_BGR2GRAY);
    //패치에 의해 잘릴 수 있는 border 설정
    copyMakeBorder(img_copy, img_copy, patch_size, patch_size, patch_size, patch_size, BORDER_CONSTANT, Scalar(0, 0, 0));
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;

    for (int i = 0; i < recMap[img]; i++) {
        cout << i << "HOG";
        MatND histogram = Mat(9, 1, CV_32F, Scalar::all(0));
        Point pnt = pntMap[img][i];
        int x = pnt.x;
        int y = pnt.y;

        Mat img2 = img_copy(Range(y, y + 2*patch_size), Range(x, x + 2*patch_size));

        Mat gx, gy;
        Mat mag, ang;
        Sobel(img2, gx, CV_32F, 1, 0);
        Sobel(img2, gy, CV_32F, 0, 1);
        cartToPolar(gx, gy, mag, ang, true);

        for (int x = 0; x < ang.cols; x++)
            for (int y = 0; y < ang.rows; y++) {
                float a = ang.at<float>(x, y);
                a = a >= 180 ? a - 180 : a;
                int index = (int)(a / 20);
                //각도에 따라 구성( 165 -> (165-160)*x, (180-165)*x)
                histogram.at<float>(index) += (a - index * 20) / 20 * mag.at<float>(x, y);
                histogram.at<float>((index + 1) % 9) += ((index + 1) * 20 - a) / 20 * mag.at<float>(x, y);
            }

        histMapHOG[img].push_back(histogram);
    }

}
void get_histogramsGRAD(void* param) {
    cv::Mat* img = (cv::Mat*) param;

    Mat img_copy;
    img->copyTo(img_copy);

    cvtColor(img_copy, img_copy, COLOR_BGR2GRAY);
    //패치에 의해 잘릴 수 있는 border 설정
    copyMakeBorder(img_copy, img_copy, patch_size * 2, patch_size * 2, patch_size * 2, patch_size * 2, BORDER_CONSTANT, Scalar(0, 0, 0));

    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;

    for (int i = 0; i < recMap[img]; i++) {
        MatND histogram = Mat(128, 1, CV_32F, Scalar::all(0));
        Point pnt = pntMap[img][i];
        int x = pnt.x;
        int y = pnt.y;

        vector<Mat> imgs;
        //4x4의 영역 설정
        for (int n = 0; n < 4; n++)
            for (int m = 0; m < 4; m++)
                imgs.push_back(img_copy(Range(y + patch_size * (n), y + patch_size * (n + 1)), Range(x + patch_size * (m), x + patch_size * (m + 1))));

        Mat gx, gy;
        Mat mag, ang;
        for (int j = 0; j < 16; j++) {
            Sobel(imgs[j], gx, CV_32F, 2, 0);
            Sobel(imgs[j], gy, CV_32F, 0, 2);
            cartToPolar(gx, gy, mag, ang, true);

            for (int x = 0; x < ang.cols; x++)
                for (int y = 0; y < ang.rows; y++) {
                    float a = ang.at<float>(x, y);
                    int index = (int)(a / 45);
                    //전부 concacenate 한 결과에 수정
                    histogram.at<float>(index + j * 8) += mag.at<float>(x, y) * (a - index * 45) / 45 ;
                    histogram.at<float>((index + 1) % 8 + j * 8) += mag.at<float>(x, y) * ((index + 1) * 45 - a) / 45;
                }
        }
        histMapGrad[img].push_back(histogram);
    }

}

void get_histogramsRGB(void* param) {
    cv::Mat* img = (cv::Mat*) param;
    MatND histogram;

    Mat img_copy;
    img->copyTo(img_copy);
    //패치에 의해 잘릴 수 있는 border 설정
    copyMakeBorder(img_copy, img_copy, patch_size, patch_size, patch_size, patch_size, BORDER_CONSTANT, Scalar(0, 0, 0));
    Mat img_channels[3];
    split(img_copy, img_channels);

    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;

    for (int i = 0; i < recMap[img]; i++) {
        Point pnt = pntMap[img][i];
        int x = pnt.x;
        int y = pnt.y;

        Mat img2 = img_copy(Range(y, y + 2 * patch_size), Range(x, x + 2 * patch_size));


        MatND histogramB, histogramG, histogramR;
        int channel_B[] = { 0 };  // Blue
        int channel_G[] = { 1 };  // Green
        int channel_R[] = { 2 };  // Red
        float channel_range[2] = { 0.0 , 255.0 };
        const float* channel_ranges[1] = { channel_range };

        // R, G, B별로 각각 히스토그램을 계산한다.
        calcHist(&img2, 1, channel_B, Mat(), histogramB, 1, &number_bins, channel_ranges);
        calcHist(&img2, 1, channel_G, Mat(), histogramG, 1, &number_bins, channel_ranges);
        calcHist(&img2, 1, channel_R, Mat(), histogramR, 1, &number_bins, channel_ranges);

        vector<MatND> his = { histogramB, histogramG, histogramR };
        histMapRGB[img].push_back(his);
    }

}

void plot_histogram(int x, int y, void* param, int method) {
    cv::Mat* img = (cv::Mat*) param;

    int hist_w = 300;
    int hist_h = 300;

    for (int i = 0; i < recMap[img]; i++) {
        MatND histogram;
        vector<MatND> histogramRGB;
        switch (method) {
        case(GRAY):
            histogram = histMap[img][i];
            break;
        case(RGB):
            histogramRGB = histMapRGB[img][i];
            break;
        case(GRADIENT):
            histogram = histMapGrad[img][i];
            break;
        case(HOG):
            histogram = histMapHOG[img][i];
            break;
        }

        //한 bin의 너비
        int bin_w = cvRound((double)hist_w / (method == RGB ? histogramRGB[0].rows : histogram.rows));
        Mat hist_img(hist_h, hist_w, CV_8UC3, Scalar::all(0));
        normalize(histogram, histogram, 0, hist_h/2, NORM_MINMAX);

        for (int j = 1; j < (method==RGB? histogramRGB[0].rows : histogram.rows); j++)
        {
            //Point p1 = Point(bin_w * (j - 1), hist_h - cvRound(histogram.at<float>(j - 1)));
            //Point p2 = Point(bin_w * (j), hist_h - cvRound(histogram.at<float>(j)));

            if (method != RGB) {
                Point p1 = Point(bin_w * (j - 1), hist_h);
                Point p2 = Point(bin_w * (j), hist_h - cvRound(histogram.at<float>(j-1)));
                //바 그래프로 구성
                rectangle(hist_img, p1, p2, Scalar(255, 0, 0), 1, 8, 0);
            }
            else {
                for (int k = 0; k < 3; k++) {
                    Point p1 = Point(cvRound(bin_w * (j + (float)k / 3)), hist_h);
                    Point p2 = Point(cvRound(bin_w * (j + ((float)k + 1) / 3)), hist_h - cvRound(histogramRGB[k].at<float>(j)));
                    //바 그래프로 구성
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
    //설명, 좌클릭으로 설정, 우클릭으로 확정
    putText(img_copy, string_format("Left to rect Right to fix (%d/4)", recMap[img]), Point(0, img_copy.rows - 20), FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, LINE_AA);
    if (chkMap[img]) putText(img_copy, "Fixed", Point(0, img_copy.rows - 10), FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, LINE_AA);

    for (int i = 0; i < recMap[img]; i++) {
        Point pnt = pntMap[img][i];
        int x = pnt.x;
        int y = pnt.y;

        rectangle(img_copy, Rect(Point(x + patch_size, y + patch_size), Point(x - patch_size, y - patch_size)), colorMap[img], 1, 8, 0);
        if (x + 20 > img->cols) x -= 20+ patch_size;
        if (y - 20 < 0) y += 20+ patch_size;
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
    //패치 출력
    map<Mat*, vector<Point> >::iterator pt = pntMap.begin();
    int cols = 0;
    for (pt; pt != pntMap.end(); pt++) {
        for (int i = 0; i < recMap[pt->first]; i++) {
            Point pnt = pt->second[i];
            int x = pnt.x;
            int y = pnt.y;

            rectangle(img_final, Rect(Point(x + cols + patch_size, y + patch_size), Point(x + cols - patch_size, y - patch_size)), colorMap[pt->first], 1, 8, 0);
            if (x + 20 > pt->first->cols + cols) x -= 20+ patch_size;
            if (y - 20 < 0) y += 20+ patch_size;
            putText(img_final, to_string(i + 1), Point(x+cols, y), FONT_HERSHEY_PLAIN, 2, colorMap[pt->first], 1, LINE_AA);

        }
        cols += pt->first->cols;
    }
    //연결선 출력
    map<Mat*, vector<Point> >::iterator pt1 = pntMap.begin();
    map<Mat*, vector<Point> >::iterator pt2 = pntMap.begin();
    cols = 0;
    for (pt1; pt1 != pntMap.end(); pt1++) {
        for (int i = 0; i < recMap[pt1->first]; i++){
            int cols1 = 0;
            for (pt2 = pntMap.begin(); pt2 != pntMap.end(); pt2++) {
                if (pt1 != pt2) {
                    int j = matchMap[pt1->first][i][pt2->first];
                    line(img_final, Point(pt1->second[i].x + cols, pt1->second[i].y+ patch_size), Point(pt2->second[j].x + cols1, pt2->second[j].y- patch_size), colorMap[pt1->first]);
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
            //현재 입력 패치수에 따라 추가 입력과 삭제를 반복
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
        //입력 패치수가 4개일때만 확정 가능
        if (recMap[img]==4) chkMap[img] = true;
        cout << "EVENT_RBUTTONUP: " << x << ", " << y << endl;
        show_img(param);

        break;
    default:
        break;
    }
}
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include "drawLandmarks.hpp"
#include "CvxText.h"

using namespace std;
using namespace cv;
using namespace cv::face;

void drawText(Mat & image, const String& text, Point position, double font_size, Scalar color);
float sideFaceDetection(vector<Point2f> &landmarks);
bool eyesOpenDetection(Mat imageFace);
static int ToWchar(char* &src, wchar_t* &dest, const char *locale = "zh_CN.utf8");

int main(int argc,char ** argv){
    VideoCapture IRCameraCapture;
    VideoCapture RGBCameraCapture;
    // set text 
    CvxText textSmall("./simhei.ttf");
    CvxText textBig("./simhei.ttf");
    // new Scalar to sizeFont 
    Scalar sizeFontSmall{15,0.5,0.1,0};
    Scalar sizeFontBig{30,0.5,0.1,0};
    // set fonts size and gap
    textSmall.setFont(nullptr,&sizeFontSmall,nullptr,0);
    textBig.setFont(nullptr,&sizeFontBig,nullptr,0);
    // Load Face Detector
    CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");

    //Load profile face Detector
    CascadeClassifier profileFaceDetector("haarcascade_profileface.xml");

    // Create an instance of Facemark
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel("lbfmodel.yaml");
    while(true){               
        RGBCameraCapture.open(0); //RGB Camera  
        if(RGBCameraCapture.isOpened()){
            cout << "RGB Cameras are opened!" << endl;
            for(;;){
                Mat imageRGB,imageGRAY,imageHist;
                RGBCameraCapture >> imageRGB;
                if(imageRGB.empty()){
                    cout << "rgb image read failed!" << endl;
                }
                else{
                    // Find face
                    vector<Rect> faces;
                    // Find upperbody
                    vector<Rect> profileFaces;
                    
                    // Convert frame to grayscale because
                    // faceDetector requires grayscale image.
                    //cout << "width:" << imageRGB.cols << "height:" << imageRGB.rows << endl;
                    //640*480 pixels
                    cvtColor(imageRGB, imageGRAY, COLOR_BGR2GRAY);

                    // Detect upperbody
                    profileFaceDetector.detectMultiScale(imageGRAY, profileFaces);
                    // Detect faces
                    faceDetector.detectMultiScale(imageGRAY, faces,1.1,3,0,Size(80,80));
                    
                    // Variable for landmarks.
                    // Landmarks for one face is a vector of points
                    // There can be more than one face in the image. Hence, we
                    // use a vector of vector of points.
                    vector< vector<Point2f> > landmarks;
                    //fresh image forever one image per 10 ms until keyboard pressed
                    // Run landmark detector
                    bool success = facemark->fit(imageRGB,faces,landmarks);
                    // Store former concentration
                    float ConcentrationArray[10];
                    if(success)
                    {
                        // If successful, render the landmarks on the face
                        // for(unsigned int i = 0; i < landmarks.size(); i++)
                        // {
                        //     //drawText(imageRGB);
                        //     drawLandmarks(imageRGB, landmarks[i]);
                        // }
                        // Detect side-degree
                        float faceProfile = sideFaceDetection(landmarks[0]);
                        // Detect eyes state
                        bool eyesState = eyesOpenDetection(imageGRAY(faces[0]));
                        // Calc current concentration
                        bool faceProfileSide = (profileFaces.size()>0);
                        float ConcentrationCurrent = (faceProfile*0.5) + (eyesState*0.5);
                        // Calc average concentration
                        static float ConcentrationAverage = 1.0;
                        static uint8_t C_time = 0;
                        ConcentrationArray[C_time] = ConcentrationCurrent;
                        if(C_time == 5){
                            C_time = 0;
                            ConcentrationAverage = 0.0;
                            for(int i = 0;i<6;i++){
                                ConcentrationAverage += ConcentrationCurrent;
                            }
                            ConcentrationAverage /= 6;
                        }
                        else{
                            C_time ++;
                        }

                        // Set four char array to store string
                        char* viewAngle = new char[40];
                        //char* eyesSplit = new char[20];
                        char* eyesSplit = new char[40];
                        char* realTime = new char[40];
                        char* average = new char[40];

                        // format float value to char array
                        sprintf(viewAngle,"正视角度:%.1f",faceProfile);
                        if(eyesState){
                            eyesSplit = (char*)"精神专注:open";
                        }
                        else{
                            eyesSplit = (char*)"精神专注:close";
                        }
                        //sprintf(eyesSplit,"Eyes Split:%s",eyesState);
                        sprintf(realTime,"实时专注:%.1f",ConcentrationCurrent);
                        sprintf(average,"平均专注:%.1f",ConcentrationAverage);
                        // convert char array to const string   
                        // const string viewAngleStr = viewAngle;
                        // const string eyesSplitStr = eyesSplit;
                        // const string realTimeStr = realTime;
                        // const string averageStr = average;
                        //drawLandmarks(imageRGB,landmarks[0]);
                        // write with chinese
                        wchar_t * viewAngleStrZH;
                        wchar_t * eyesSplitStrZH;
                        wchar_t * realTimeStrZH;
                        wchar_t * averageStrZH;

                        ToWchar(viewAngle,viewAngleStrZH);
                        ToWchar(eyesSplit,eyesSplitStrZH);
                        ToWchar(realTime,realTimeStrZH);
                        ToWchar(average,averageStrZH);

                        textSmall.putText(imageRGB, viewAngleStrZH, cv::Point(30,50), cv::Scalar(255, 255, 255));
                        textSmall.putText(imageRGB, eyesSplitStrZH, cv::Point(30,80), cv::Scalar(255, 255, 255));
                        textSmall.putText(imageRGB, realTimeStrZH, cv::Point(30,110), cv::Scalar(255, 255, 255));
                        textSmall.putText(imageRGB, averageStrZH, cv::Point(30,140), cv::Scalar(255, 255, 255));

                        // drawText(imageRGB,viewAngleStr,Point(60,30),0.5,Scalar(255,255,255));
                        // drawText(imageRGB,eyesSplitStr,Point(200,30),0.5,Scalar(255,255,255));
                        // drawText(imageRGB,realTimeStr,Point(340,30),0.5,Scalar(255,255,255));
                        // drawText(imageRGB,averageStr,Point(480,30),0.5,Scalar(255,255,255));
                    }
                    else if(profileFaces.size() > 0){
                        //drawText(imageRGB,"Please Concentrate!",Point(220,30),0.7,Scalar(255,255,255));
                        char * textWarning = (char *)"请集中注意力!";
                        wchar_t * textWarningStr;
                        ToWchar(textWarning,textWarningStr);
                        textBig.putText(imageRGB, textWarningStr, cv::Point(200,50), cv::Scalar(0, 0, 255));
                    }
                    else{
                        char * textLeave = (char *)"无人出席!";
                        wchar_t * textLeaveStr;
                        ToWchar(textLeave,textLeaveStr);
                        textBig.putText(imageRGB, textLeaveStr, cv::Point(300,50), cv::Scalar(0, 0, 255));
                        //drawText(imageRGB,"No One Present!",Point(250,30),0.7,Scalar(255,255,255));
                    }
                    imshow("RGB Camera",imageRGB);
                    if(waitKey(10)== 'q') return 0;
                }
            }            
        }
        else{
            cout << "RGB Cameras open failed!" << endl;
        }      
    }
}

void drawText(Mat & image, const String& text, Point position, double font_size, Scalar color)
{
    putText(image, 
            text,
            position,
            FONT_HERSHEY_COMPLEX, 
            font_size, // font face and scale
            color, // white
            1, LINE_AA); // line thickness and type
}

float sideFaceDetection(vector<Point2f> &landmarks){
    uint16_t noseTip,faceRight,faceLeft;
    float ratio;
    noseTip = landmarks[30].x;
    faceRight = landmarks[1].x;
    faceLeft = landmarks[15].x;
    ratio = (float)(noseTip-faceRight)/(faceLeft-noseTip);
    if(ratio >= 1) ratio = (float)(1.00/ratio);
    //cout << "ratio:" << ratio << endl;
    return ratio;
}

// 
bool eyesOpenDetection(Mat imageFace){
    // uint16_t leftEyeWidth,leftEyeHeight,rightEyeWidth,rightEyeHeight;
    // float degreeLeft,degreeRight,degree;
    // leftEyeHeight = landmarks[46].y - landmarks[44].y;
    // leftEyeWidth = landmarks[45].x - landmarks[42].x;

    // rightEyeHeight = landmarks[41].y - landmarks[37].y;
    // rightEyeWidth = landmarks[39].x - landmarks[36].x;

    // degreeLeft = (float) leftEyeHeight / leftEyeWidth;
    // degreeRight = (float) rightEyeHeight / rightEyeWidth;
    // degree = (float) (degreeLeft*0.5 + degree*0.5)/max(degreeLeft,degreeRight);
    // //cout << "degree:" << degree << endl;
    // return degree;

    //Load eyes Detector
    CascadeClassifier eyesDetector("haarcascade_eye.xml");

    // Find eyes
    vector<Rect> eyes;
    // Detect eyes
    eyesDetector.detectMultiScale(imageFace,eyes,1.1,3,0,Size(25,25),Size(50,50));
                        // 
    if(eyes.size() > 0){
        Mat imageBinary,imageEye,imageLris;
        imageEye = imageFace(eyes[0]);
        //imshow("eye",imageEye);
        //cout << imageEye << endl;
        adaptiveThreshold(imageEye,imageBinary,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY_INV,19,5);
        //imshow("binary",imageBinary); 

        Mat element1 = getStructuringElement(MORPH_RECT,Size(4,4),Size(-1,-1));
        morphologyEx(imageBinary,imageBinary,MORPH_CLOSE,element1,Size(-1,-1),1);
        //imshow("close",imageBinary);

        Mat element2 = getStructuringElement(MORPH_ELLIPSE,Size(9,9),Size(-1,-1));
        morphologyEx(imageBinary,imageLris,MORPH_OPEN,element2);
        //imshow("open",imageLris);

        // Detect the existence of lris
        // 
        return true;
    }
    else{
        return false;
    }
}

static int ToWchar(char* &src, wchar_t* &dest, const char *locale)
{
    if (src == NULL) {
        dest = NULL;
        return 0;
    }

    // 根据环境变量设置locale
    setlocale(LC_CTYPE, locale);

    // 得到转化为需要的宽字符大小
    int w_size = mbstowcs(NULL, src, 0) + 1;

    // w_size = 0 说明mbstowcs返回值为-1。即在运行过程中遇到了非法字符(很有可能使locale
    // 没有设置正确)
    if (w_size == 0) {
        dest = NULL;
        return -1;
    }

    //wcout << "w_size" << w_size << endl;
    dest = new wchar_t[w_size];
    if (!dest) {
        return -1;
    }

    int ret = mbstowcs(dest, src, strlen(src)+1);
    if (ret <= 0) {
        return -1;
    }
    return 0;
}
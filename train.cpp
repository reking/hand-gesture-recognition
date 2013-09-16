/*the hand*/


#include <iostream>
#include<string>
#include <fstream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;
using namespace std;
// Functions
static void read_imgList(const string& filename, vector<Mat>& images) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line;
    while (getline(file, line)) {
        images.push_back(imread(line, 0));
		
    }
}

static void read_labels(const string&filename,vector<int>& labels)
{
	std::ifstream file(filename.c_str(),ifstream::in);
	if(!file){
			string error_message = "No valid input svm labels was given, please check the given filename.";
            CV_Error(CV_StsBadArg, error_message);
	}
	string line;
	 while (getline(file,line)) {
		 labels.push_back(atoi(line.c_str()));
    }
}

static  Mat formatImages(const vector<Mat> &data)
{
    Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
    for(unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1,1);
        Mat row_i = dst.row(i);
        image_row.convertTo(row_i,CV_32F);
    }
    return dst;
}




int main(int argc, char** argv)
{
    if (argc != 3) {
        cout << "usage: " << argv[0] << " <image_list.txt> <labels_lists.txt>" << endl;
        exit(1);
    }

    // Get the path to your CSV.
    string imgList = string(argv[1]);
	string labelist=string(argv[2]);

    // vector to hold the images
    vector<Mat> images;
	vector<int> labels;

    // Read in the data. This can fail if not valid
    try {
        read_imgList(imgList, images);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << imgList << "\". Reason: " << e.msg << endl;
        exit(1);
    }
	try {
        read_labels(labelist, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << labelist << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

    // Reshape and stack images into a rowMatrix
    Mat data = formatImages(images);

    

   //------------------------ 2. Set up the support vector machines parameters --------------------
    CvSVMParams params;
    params.svm_type    = SVM::C_SVC;
    params.C 		   = 0.01;
    params.kernel_type = SVM::RBF;
	params.gamma=1;
	params.coef0=1;
	params.degree=0;
	params.nu=0;
	params.p=0;
	params.term_crit = TermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
    //params.term_crit = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);

    //------------------------ 3. Train the svm ----------------------------------------------------
	
    cout << "Starting training process" << endl;
    CvSVM handsvm=CvSVM();
	
	const CvMat traindata= data;

	const int len = labels.size();
	vector<int>::iterator iter = labels.begin();
	//--------------translate the label vector<int> fomr labels.tex to cvmat----------------------
    CvMat* label = cvCreateMat(1,len,CV_32S); 
	
	for(int i=0;i<len;i++)
	{
		*((int*)CV_MAT_ELEM_PTR(*label,0,i))=labels[i];
	}
	
	handsvm.train(&traindata,label, NULL, NULL,params);
	handsvm.save("svmdata.xml");
    cout << "Finished training process" << endl;
	Mat src=cvLoadImage("../handdata/3-7.jpg",0);
	namedWindow("the new",0);
	
	
	cout<<"cs:"<<src.channels()<<endl;
    Mat cs = src.reshape(1, 1) + 0; 
	
	cout<<"data.row:"<<data.row(1).size()<<endl;
	cs.convertTo(cs,CV_32FC1);
    CvMat d3 = cs;
    int ret = handsvm.predict(&d3);
    cout<<ret<<endl;
	imshow("the new",src);
	waitKey(0);
    return 0;
   
}
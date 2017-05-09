case @!utype!@: {
    double thresh = static_cast<double>(((@!type!@*) B_data)[0]);
    double maxval = static_cast<double>(((@!type!@*) B_data)[1]);

    cv::Mat src = cv::Mat(A->shape[0], A->shape[1], @!opencv_type!@, (@!type!@*) A_data);
    cv::Mat dst = cv::Mat(C->shape[0], C->shape[1], @!opencv_type!@, (@!type!@*) C_data);
    threshold(src, dst, thresh, maxval, cv::@!threshold_type!@);
    break;
}

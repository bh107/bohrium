/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#include <bh_extmethod.hpp>
#include <bh_main_memory.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace bohrium;
using namespace extmethod;
using namespace std;

namespace {

class ErodeImpl : public ExtmethodImpl {
public:
    void execute(bh_instruction *instr, void* arg) {
        // All matrices must be contigous
        assert(instr->isContiguous());

        // A is our image
        bh_view* A = &instr->operand[1];
        bh_data_malloc(A->base);
        void *A_data = A->base->data;

        // B is our kernel
        bh_view* B = &instr->operand[2];
        bh_data_malloc(B->base);
        void *B_data = B->base->data;

        // C is our output image
        bh_view* C = &instr->operand[0];
        bh_data_malloc(C->base);
        void *C_data = C->base->data;

        // We make sure that the data types are the same
        assert(A->base->type == B->base->type);
        assert(A->base->type == C->base->type);

        int B_size = B->shape[0] * B->shape[1];
        bh_int8* B_intdata = new bh_int8[B_size];

        cv::Mat src, dst;

        switch(A->base->type) {
            case bh_type::UINT8: {
                std::copy((bh_uint8*) B_data, ((bh_uint8*) B_data) + B_size, B_intdata);
                src = cv::Mat(A->shape[0], A->shape[1], CV_8UC1, (bh_uint8*) A_data);
                dst = cv::Mat(C->shape[0], C->shape[1], CV_8UC1, (bh_uint8*) C_data);
                break;
            }
            case bh_type::FLOAT32: {
                std::copy((bh_float32*) B_data, ((bh_float32*) B_data) + B_size, B_intdata);
                src = cv::Mat(A->shape[0], A->shape[1], CV_32FC1, (bh_float32*) A_data);
                dst = cv::Mat(C->shape[0], C->shape[1], CV_32FC1, (bh_float32*) C_data);
                break;
            }
            case bh_type::FLOAT64: {
                std::copy((bh_float64*) B_data, ((bh_float64*) B_data) + B_size, B_intdata);
                src = cv::Mat(A->shape[0], A->shape[1], CV_64FC1, (bh_float64*) A_data);
                dst = cv::Mat(C->shape[0], C->shape[1], CV_64FC1, (bh_float64*) C_data);
                break;
            }
            default: {
                std::stringstream ss;
                ss << bh_type_text(A->base->type) << " not supported by OpenCV for 'erode'.";
                throw std::runtime_error(ss.str());
            }
        }

        cv::Mat kernel = cv::Mat(B->shape[0], B->shape[1], CV_8UC1, B_intdata);

        // http://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb
        // src, dst, kernel[, anchor, iterations, borderType, borderValue]
        cv::erode(src, dst, kernel);
    }
};

class DilateImpl : public ExtmethodImpl {
public:
    void execute(bh_instruction *instr, void* arg) {
        // All matrices must be contigous
        assert(instr->isContiguous());

        // A is our image
        bh_view* A = &instr->operand[1];
        bh_data_malloc(A->base);
        void *A_data = A->base->data;

        // B is our kernel
        bh_view* B = &instr->operand[2];
        bh_data_malloc(B->base);
        void *B_data = B->base->data;

        // C is our output image
        bh_view* C = &instr->operand[0];
        bh_data_malloc(C->base);
        void *C_data = C->base->data;

        // We make sure that the data types are the same
        assert(A->base->type == B->base->type);
        assert(A->base->type == C->base->type);

        int B_size = B->shape[0] * B->shape[1];
        bh_int8* B_intdata = new bh_int8[B_size];

        cv::Mat src;
        cv::Mat dst;

        switch(A->base->type) {
            case bh_type::UINT8: {
                std::copy((bh_uint8*) B_data, ((bh_uint8*) B_data) + B_size, B_intdata);
                src = cv::Mat(A->shape[0], A->shape[1], CV_8UC1, (bh_uint8*) A_data);
                dst = cv::Mat(C->shape[0], C->shape[1], CV_8UC1, (bh_uint8*) C_data);
                break;
            }
            case bh_type::FLOAT32: {
                std::copy((bh_float32*) B_data, ((bh_float32*) B_data) + B_size, B_intdata);
                src = cv::Mat(A->shape[0], A->shape[1], CV_32FC1, (bh_float32*) A_data);
                dst = cv::Mat(C->shape[0], C->shape[1], CV_32FC1, (bh_float32*) C_data);
                break;
            }
            case bh_type::FLOAT64: {
                std::copy((bh_float64*) B_data, ((bh_float64*) B_data) + B_size, B_intdata);
                src = cv::Mat(A->shape[0], A->shape[1], CV_64FC1, (bh_float64*) A_data);
                dst = cv::Mat(C->shape[0], C->shape[1], CV_64FC1, (bh_float64*) C_data);
                break;
            }
            default: {
                std::stringstream ss;
                ss << bh_type_text(A->base->type) << " not supported by OpenCV for 'erode'.";
                throw std::runtime_error(ss.str());
            }
        }

        cv::Mat kernel = cv::Mat(B->shape[0], B->shape[1], CV_8UC1, B_intdata);

        // http://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c
        // src, dst, kernel[, anchor, iterations, borderType, borderValue]
        cv::dilate(src, dst, kernel);
    }
};

class ConnectedComponentsImpl : public ExtmethodImpl {
public:
    void execute(bh_instruction *instr, void* arg) {
        // All matrices must be contigous
        assert(instr->isContiguous());

        // A is our image
        bh_view* A = &instr->operand[1];
        bh_data_malloc(A->base);
        void *A_data = A->base->data;

        if(A->base->type != bh_type::UINT8) {
            throw std::runtime_error("Connected components by OpenCV only works for uint8 images.");
        }

        // B is a scalar of our connectivity
        bh_view* B = &instr->operand[2];
        bh_data_malloc(B->base);
        void* B_data = B->base->data;
        assert(B->base->nelem == 1);

        // C is our output labels
        bh_view* C = &instr->operand[0];
        bh_data_malloc(C->base);
        void *C_data = C->base->data;

        int connectivity = static_cast<int>(((bh_uint8*) B_data)[0]);

        cv::Mat img = cv::Mat(A->shape[0], A->shape[1], CV_8UC1, (bh_uint8*) A_data);
        cv::Mat labelImage = cv::Mat(C->shape[0], C->shape[1], CV_8UC1, (bh_uint8*) C_data);

        connectedComponents(img, labelImage, connectivity);

        for(int i = 0; i < img.rows * img.cols; ++i) {
            ((bh_uint8*) C_data)[i] = labelImage.at<int>(i);
        }
    }
};

} // Unnamed namespace

extern "C" ExtmethodImpl* opencv_erode_create() {
    return new ErodeImpl();
}

extern "C" void opencv_erode_destroy(ExtmethodImpl* self) {
    delete self;
}

extern "C" ExtmethodImpl* opencv_dilate_create() {
    return new DilateImpl();
}

extern "C" void opencv_dilate_destroy(ExtmethodImpl* self) {
    delete self;
}

extern "C" ExtmethodImpl* opencv_connected_components_create() {
    return new ConnectedComponentsImpl();
}

extern "C" void opencv_connected_components_destroy(ExtmethodImpl* self) {
    delete self;
}

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

#include <bh_base.hpp>
#include <bh_main_memory.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
#include <colors.hpp>
#include "compression.hpp"
#include "zlib.hpp"

using namespace std;

namespace bohrium {

namespace {
/// Help function that converts to OpenCV dtypes
int bh2cv_dtype(bh_type type) {
    switch (type) {
        case bh_type::INT8:
            return CV_8SC1;
        case bh_type::INT16:
            return CV_16SC1;
        case bh_type::INT32:
            return CV_32SC1;
        case bh_type::UINT8:
            return CV_8UC1;
        case bh_type::UINT16:
            return CV_16UC1;
        case bh_type::FLOAT32:
            return CV_32FC1;
        case bh_type::FLOAT64:
            return CV_64FC1;
        default:
            throw std::runtime_error("bh2cv_dtype: unsupported type UINT64");
    }
}
}

std::vector<unsigned char> Compression::compress(const bh_view &ary, const std::string &param) {
    std::vector<unsigned char> ret;
    if (not bh_is_contiguous(&ary) or bh_nelements(ary) != ary.base->nelem) {
        throw std::runtime_error("compress(): `ary` must be contiguous and represent the whole of its base");
    }
    if (ary.base->data == nullptr) {
        throw std::runtime_error("compress(): `ary` data is NULL");
    }
    vector<string> param_list;
    boost::split(param_list, param, boost::is_any_of(","));
    if (param.empty() or param_list.empty() or param_list[0] == "none") {
        ret.resize(ary.base->nbytes());
        memcpy(&ret[0], ary.base->data, ary.base->nbytes());
    } else if (param_list[0] == "zlib") {
        ret = zlib_compress(ary.base->data, ary.base->nbytes());
    } else if (param_list[0] == "jpg" or param_list[0] == "png" or param_list[0] == "jp2") {
        const int cv_type = bh2cv_dtype(ary.base->type);
        if (ary.base->type != bh_type::UINT8) {
            throw std::runtime_error("compress(): jpg and png only support uint8 arrays");
        }
        int sizes[BH_MAXDIM];
        for (int i = 0; i < ary.ndim; i++) {
            sizes[i] = static_cast<int>(ary.shape[i]);
        }

        // Convert the string `param` to the OpenCV `params`
        std::vector<int> params;
        if (param_list.size() > 1) {
            if (param_list[0] == "jpg") {
                params.push_back(CV_IMWRITE_JPEG_QUALITY);
                params.push_back(std::stoi(param_list[1]));
            } else if (param_list[0] == "png") {
                params.push_back(CV_IMWRITE_PNG_COMPRESSION);
                params.push_back(std::stoi(param_list[1]));
            }
        }

        cv::Mat mat(static_cast<int>(ary.ndim), sizes, cv_type, ary.base->data);
        cv::imencode("." + param_list[0], mat, ret, params);
    } else {
        throw std::runtime_error("compress(): unknown param");
    }
    stat_per_codex[param].push_back(Stat{ary.base->nbytes(), ret.size()});
    return ret;
}

std::vector<unsigned char> Compression::compress(const bh_base &ary, const std::string &param) {
    auto &a = const_cast<bh_base &>(ary);
    const bh_view view{a}; // View of the whole base
    return compress(view, param);
}

void Compression::uncompress(const std::vector<unsigned char> &data, bh_view &ary, const std::string &param) {
    if (not bh_is_contiguous(&ary) or bh_nelements(ary) != ary.base->nelem) {
        throw std::runtime_error("uncompress(): `ary` must be contiguous and represent the whole of its base");
    }
    if (data.empty()) {
        throw std::runtime_error("uncompress(): `data` is empty!");
    }
    bh_data_malloc(ary.base);

    vector<string> param_list;
    boost::split(param_list, param, boost::is_any_of(","));
    if (param.empty() or param_list.empty() or param_list[0] == "none") {
        assert(data.size() == ary.base->nbytes());
        memcpy(ary.base->data, &data[0], ary.base->nbytes());
    } else if (param_list[0] == "zlib") {
        zlib_uncompress(data, ary.base->data, ary.base->nbytes());
    } else if (param_list[0] == "jpg" or param_list[0] == "png" or param_list[0] == "jp2") {
        if (ary.base->type != bh_type::UINT8) {
            throw std::runtime_error("uncompress(): jpg and png only support uint8 arrays");
        }
        cv::Mat out = cv::imdecode(data, CV_LOAD_IMAGE_ANYDEPTH);
        if (out.data == nullptr) {
            throw std::runtime_error("imdecode(): failed!");
        }
        assert(ary.base->nbytes() == (size_t) (out.dataend - out.data));
        memcpy(ary.base->data, out.data, ary.base->nbytes());
    } else {
        throw std::runtime_error("compress(): unknown param");
    }
    stat_per_codex[param].push_back(Stat{ary.base->nbytes(), data.size()});
}

void Compression::uncompress(const std::vector<unsigned char> &data, bh_base &ary, const std::string &param) {
    bh_view view{ary}; // View of the whole base
    uncompress(data, view, param);
}

std::string Compression::pprintStats() const {
    stringstream ss;
    ss << BLU << "[PROXY-VEM] Profiling: \n" << RST;
    for (auto &param: stat_per_codex) {
        uint64_t total_raw = 0;
        uint64_t total_compressed = 0;
        for (const Stat &stat: param.second) {
            total_raw += stat.total_raw;
            total_compressed += stat.total_compressed;
        }
        ss << "Codex \"" << param.first << "\":\n";
        ss << "  Raw data: " << total_raw << "\n";
        ss << "  Zip data: " << total_compressed << "\n";
        ss << "  Ratio: " << total_raw / (double) total_compressed << "\n";
    }
    return ss.str();
}

std::string Compression::pprintStatsDetail() const {
    stringstream ss;
    for (auto &param: stat_per_codex) {
        ss << "Codex \"" << param.first << "\":\n";
        ss << "  Raw data: ";
        for (const Stat &stat: param.second) {
            ss << stat.total_raw << ", ";
        }
        ss << "\n";
        ss << "  Zip data: ";
        for (const Stat &stat: param.second) {
            ss << stat.total_compressed << ", ";
        }
        ss << "\n";
        ss << "  Ratio: ";
        for (const Stat &stat: param.second) {
            ss << stat.total_raw / (double) stat.total_compressed << ", ";
        }
        ss << "\n";
    }
    return ss.str();
}

}

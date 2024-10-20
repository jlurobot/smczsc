#include <vector>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <nanoflann.hpp>
#include <fftm/fftm.hpp>
#include <algorithm>
#include <xsearch.h>
#include <nanoflann.hpp>
#include <utils/KDTreeVectorOfVectorsAdaptor.h>
#include <utils/serializer.h>

using namespace std;
using namespace cv;


struct FeatureDesc
{
    cv::Mat1b img;
    cv::Mat1b T;
    cv::Mat1b M;
};


class SMZCSC
{
public:
    SMZCSC(int nscale, int minWaveLength, float mult, float sigmaOnf, int matchNum) : _nscale(nscale),
                                                                                         _minWaveLength(minWaveLength),
                                                                                         _mult(mult),
                                                                                         _sigmaOnf(sigmaOnf),
                                                                                         _matchNum(matchNum)
    {
    }
    SMZCSC(const SMZCSC &) = delete;
    SMZCSC &operator=(const SMZCSC &) = delete;

    static cv::Mat1b GetDesc(const pcl::PointCloud<point_type> &cloud);
    float Compare(const FeatureDesc &img1, const FeatureDesc &img2, int *bias = nullptr);

    FeatureDesc GetFeature(const cv::Mat1b &src);
    std::vector<cv::Mat2f> LogGaborFilter(const cv::Mat1f &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf);
    void GetHammingDistance(const cv::Mat1b &T1, const cv::Mat1b &M1, const cv::Mat1b &T2, const cv::Mat1b &M2, int scale, float &dis, int &bias);
    float GetL1Distance(const cv::Mat1b &img1, const cv::Mat1b &img2, int scale, int &bias);

    float GetDistance(const FeatureDesc &img1, const FeatureDesc &img2,unsigned int nscale, int *bias = nullptr){
        if (distance_type == 0)
            return GetL1Distance(img1.img, img2.img, 0, *bias);
        else {
            float dis = 0.0f;
            GetHammingDistance(img1.T, img1.M, img2.T, img2.M, nscale, dis,*bias);
            return dis;
        }
    }

    static inline cv::Mat circRowShift(const cv::Mat &src, int shift_m_rows);
    static inline cv::Mat circColShift(const cv::Mat &src, int shift_n_cols);
    static cv::Mat circShift(const cv::Mat &src, int shift_m_rows, int shift_n_cols);

    int distance_type = 1; // 0: L1, 1: Hamming
private:
    void LoGFeatureEncode(const cv::Mat1b &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf, cv::Mat1b &T, cv::Mat1b &M);

    int _nscale;
    int _minWaveLength;
    float _mult;
    float _sigmaOnf;
    int _matchNum;
};

cv::Mat1b SMZCSC::GetDesc(const pcl::PointCloud<point_type> &cloud)
{
    cv::Mat1b Map = cv::Mat1b::zeros(80, 360);

    for (auto p : cloud.points)
    {
        float dis = sqrt(p.data[0] * p.data[0] + p.data[1] * p.data[1]);
        float yaw = (atan2(p.data[1], p.data[0]) * 180.0f / M_PI) + 180;
        int Q_dis = std::min(std::max((int)floor(dis), 0), 79);
       
        int Q_yaw = std::min(std::max((int)floor(yaw + 0.5), 0), 359);
        if (Q_dis <= 19)
        {
            Q_yaw = Q_yaw / 8 * 8;
        }
        uint8_t thisz = Map.at<uint8_t>(Q_dis, Q_yaw);
        if (p.data[2] + 2 > thisz)
        {
                Map.at<uint8_t>(Q_dis, Q_yaw) = p.data[2]+2;
        }
    }
    return Map;
}

float SMZCSC::Compare(const FeatureDesc &img1, const FeatureDesc &img2, int *bias)
{
    if (_matchNum == 2) 
    {
        auto firstRect = FFTMatch(img2.img, img1.img);
        int firstShift = firstRect.center.x - img1.img.cols / 2;
        float dis1;
        int bias1;

        dis1 = GetDistance(img1, img2, firstShift, &bias1);

        auto T2x = circShift(img2.T, 0, 180);
        auto M2x = circShift(img2.M, 0, 180);
        auto img2x = circShift(img2.img, 0, 180);

        auto secondRect = FFTMatch(img2x, img1.img);
        int secondShift = secondRect.center.x - img1.img.cols / 2;
        float dis2 = 0;
        int bias2 = 0;

        dis2 = GetDistance(img1, img2, secondShift, &bias2);

        if (dis1 < dis2)
        {
            if (bias)
                *bias = bias1;
            return dis1;
        }
        else
        {
            if (bias)
                *bias = (bias2 + 180) % 360;
            return dis2;
        }
    }
    if (_matchNum == 1) 
    {
        auto T2x = circShift(img2.T, 0, 180);
        auto M2x = circShift(img2.M, 0, 180);
        auto img2x = circShift(img2.img, 0, 180);

        auto secondRect = FFTMatch(img2x, img1.img);
        int secondShift = secondRect.center.x - img1.img.cols / 2;
        float dis2 = 0;
        int bias2 = 0;

        dis2 = GetDistance(img1, img2, secondShift, &bias2);
        if (bias)
            *bias = (bias2 + 180) % 360;
        return dis2;
    }
    if (_matchNum == 0)
    {
        auto firstRect = FFTMatch(img2.img, img1.img);
        int firstShift = firstRect.center.x - img1.img.cols / 2;

        float dis1;
        int bias1;
        dis1 = GetDistance(img1, img2, firstShift, &bias1);
        if (bias)
            *bias = bias1;
        return dis1;
    }

    return 0.0f;
}

std::vector<cv::Mat2f> SMZCSC::LogGaborFilter(const cv::Mat1f &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf)
{
    int rows = src.rows;
    int cols = src.cols;
    cv::Mat2f filtersum = cv::Mat2f::zeros(1, cols);
    std::vector<cv::Mat2f> EO(nscale);
    int ndata = cols;
    if (ndata % 2 == 1)
        ndata--;
    cv::Mat1f logGabor = cv::Mat1f::zeros(1, ndata);
    cv::Mat2f result = cv::Mat2f::zeros(rows, ndata);
    cv::Mat1f radius = cv::Mat1f::zeros(1, ndata / 2 + 1);
    radius.at<float>(0, 0) = 1;
    for (int i = 1; i < ndata / 2 + 1; i++)
    {
        radius.at<float>(0, i) = i / (float)ndata;
    }
    double wavelength = minWaveLength;
    for (int s = 0; s < nscale; s++)
    {
        double fo = 1.0 / wavelength;
        double rfo = fo / 0.5;
        //
        cv::Mat1f temp; //(radius.size());
        cv::log(radius / fo, temp);
        cv::pow(temp, 2, temp);
        cv::exp((-temp) / (2 * log(sigmaOnf) * log(sigmaOnf)), temp);
        temp.copyTo(logGabor.colRange(0, ndata / 2 + 1));
        //
        logGabor.at<float>(0, 0) = 0;
        cv::Mat2f filter;
        cv::Mat1f filterArr[2] = {logGabor, cv::Mat1f::zeros(logGabor.size())};
        cv::merge(filterArr, 2, filter);
        filtersum = filtersum + filter;
        for (int r = 0; r < rows; r++)
        {
            cv::Mat2f src2f;
            cv::Mat1f srcArr[2] = {src.row(r).clone(), cv::Mat1f::zeros(1, src.cols)};
            cv::merge(srcArr, 2, src2f);
            cv::dft(src2f, src2f);
            cv::mulSpectrums(src2f, filter, src2f, 0);
            cv::idft(src2f, src2f);
            src2f.copyTo(result.row(r));
        }
        EO[s] = result.clone();
        wavelength *= mult;
    }
    filtersum = circShift(filtersum, 0, cols / 2);
    return EO;
}

void SMZCSC::LoGFeatureEncode(const cv::Mat1b &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf, cv::Mat1b &T, cv::Mat1b &M)
{
    cv::Mat1f srcFloat;
    src.convertTo(srcFloat, CV_32FC1);
    auto list = LogGaborFilter(srcFloat, nscale, minWaveLength, mult, sigmaOnf);
    std::vector<cv::Mat1b> Tlist(nscale * 2), Mlist(nscale * 2);
    for (int i = 0; i < list.size(); i++)
    {
        cv::Mat1f arr[2];
        cv::split(list[i], arr);
        Tlist[i] = arr[0] > 0;
        Tlist[i + nscale] = arr[1] > 0;
        cv::Mat1f m;
        cv::magnitude(arr[0], arr[1], m);
        Mlist[i] = m < 0.0001;
        Mlist[i + nscale] = m < 0.0001;
    }
    cv::vconcat(Tlist, T);
    cv::vconcat(Mlist, M);
}

FeatureDesc SMZCSC::GetFeature(const cv::Mat1b &src)
{
    FeatureDesc desc;
    desc.img = src;
    LoGFeatureEncode(src, _nscale, _minWaveLength, _mult, _sigmaOnf, desc.T, desc.M);
    return desc;
}

void SMZCSC::GetHammingDistance(const cv::Mat1b &T1, const cv::Mat1b &M1, const cv::Mat1b &T2, const cv::Mat1b &M2, int scale, float &dis, int &bias)
{
    dis = NAN;
    bias = -1;
    for (int shift = scale - 2; shift <= scale + 2; shift++)
    {
        cv::Mat1b T1s = circShift(T1, 0, shift);
        cv::Mat1b M1s = circShift(M1, 0, shift);
        cv::Mat1b mask = M1s | M2;
        int MaskBitsNum = cv::sum(mask / 255)[0];
        int totalBits = T1s.rows * T1s.cols - MaskBitsNum;
        cv::Mat1b C = T1s ^ T2;
        C = C & ~mask;
        int bitsDiff = cv::sum(C / 255)[0];
        if (totalBits == 0)
        {
            dis = NAN;
        }
        else
        {
            float currentDis = bitsDiff / (float)totalBits;
            if (currentDis < dis || isnan(dis))
            {
                dis = currentDis;
                bias = shift;
            }
        }
    }
    return;
}

float SMZCSC::GetL1Distance(const cv::Mat1b &img1, const cv::Mat1b &img2, int scale, int &bias)
{
    float dis = NAN;
    bias = -1;
    for (int shift = scale - 2; shift <= scale + 2; shift++)
    {
        cv::Mat1b T1s = circShift(img1, 0, shift);
        
        size_t total_count = T1s.rows * T1s.cols;
        float distance = 0.0f;
        for (size_t i = 0; i < total_count; i++)
        {
            uint8_t t1 = T1s.data[i];
            uint8_t t2 = img2.data[i];
            if(t1 > t2) distance += t1 - t2;
            else distance += t2 - t1;
        }

        distance /= total_count;

        
        if (distance < dis || isnan(dis))
        {
            dis = distance;
            bias = shift;
        }

    }
    return dis;
}

inline cv::Mat SMZCSC::circRowShift(const cv::Mat &src, int shift_m_rows)
{
    if (shift_m_rows % src.rows == 0)
        return src.clone();
    shift_m_rows %= src.rows;
    int m = shift_m_rows > 0 ? shift_m_rows : src.rows + shift_m_rows;
    cv::Mat dst(src.size(), src.type());
    src(cv::Range(src.rows - m, src.rows), cv::Range::all()).copyTo(dst(cv::Range(0, m), cv::Range::all()));
    src(cv::Range(0, src.rows - m), cv::Range::all()).copyTo(dst(cv::Range(m, src.rows), cv::Range::all()));
    return dst;
}

inline cv::Mat SMZCSC::circColShift(const cv::Mat &src, int shift_n_cols)
{
    if (shift_n_cols % src.cols == 0)
        return src.clone();
    shift_n_cols %= src.cols;
    int n = shift_n_cols > 0 ? shift_n_cols : src.cols + shift_n_cols;
    cv::Mat dst(src.size(), src.type());
    src(cv::Range::all(), cv::Range(src.cols - n, src.cols)).copyTo(dst(cv::Range::all(), cv::Range(0, n)));
    src(cv::Range::all(), cv::Range(0, src.cols - n)).copyTo(dst(cv::Range::all(), cv::Range(n, src.cols)));
    return dst;
}

cv::Mat SMZCSC::circShift(const cv::Mat &src, int shift_m_rows, int shift_n_cols)
{
    return circColShift(circRowShift(src, shift_m_rows), shift_n_cols);
}

using vtype = std::array<float, 80>;

struct __smczsc_device : search_device {
    KDTreeVectorOfVectorsAdaptor<std::vector<vtype>, float, 80, nanoflann::metric_L2> *tree = nullptr;
    SMZCSC sbsc;
    std::vector<FeatureDesc> features;
    std::vector<vtype> joined_vtype;
    std::vector<object_id> joined_id;

    __smczsc_device() : sbsc(4, 18, 1.6, 0.75, 0) {}
};

constexpr const char* __smczsc_module_name = "smczsc";

static vtype __smczsc_makev(const FeatureDesc &desc) {
    vtype v;
    assert(desc.img.rows == 80);

    for(int i = 0; i < 80; i++) {
        v[i] = 0;
        for(int j = 0; j < desc.img.cols; j++) {
            v[i] += (float)desc.img.at<uchar>(i, j);
        }
    }
    return v;
}

static object_id __smczsc_create_object(search_device* object, const pcl::PointCloud<point_type> &cloud) {
    if(object->module->name != __smczsc_module_name) {
        return obj_none;
    }

    auto sbsc = static_cast<__smczsc_device*>(object);
    auto key = sbsc->sbsc.GetDesc(cloud);
    auto desc = sbsc->sbsc.GetFeature(key);

    sbsc->features.push_back(std::move(desc));
    return sbsc->features.size() - 1;
}

static size_t __smczsc_search(search_device* object, object_id searched_target, search_result *results, size_t max_results) {
    if(object->module->name != __smczsc_module_name) {
        return 0;
    }

    auto sbsc = static_cast<__smczsc_device*>(object);
    auto &target = sbsc->features[searched_target];

    std::vector<std::pair<size_t, double>> ret_matches;
    vtype v = __smczsc_makev(target);
    size_t index[10];
    float dist[10];

    if(sbsc->tree == nullptr) {
        results[0].id = obj_none;
        results[0].score = 0.0f;
        return 1;
    }

    const size_t nMatches = sbsc->tree->index->knnSearch(v.data(), max_results, index, dist, 10);

    results->score = 1000000.0f;
    results->id = obj_none;

    for (size_t i = 0; i < nMatches; i++) {
        object_id cid = sbsc->joined_id[index[i]];
        int bias = 0;
        float distance = sbsc->sbsc.Compare(target, sbsc->features[cid], &bias);

        if(results->id == obj_none || distance < results->score) {
            results->id = cid;
            results->score = distance;
            results->yaw = bias / 180.0f * M_PI;
        }
    }

    return 1;
}

static bool __smczsc_config(search_device* object, const char *key, const char *value) {
     if(object->module->name != __smczsc_module_name) {
        return 0;
    }

    auto sbsc = static_cast<__smczsc_device*>(object);

    if(strcmp(key, "distance_type") == 0) {
        if(strcmp(value, "L1") == 0) {
            sbsc->sbsc.distance_type = 0;
            return true;
        }
        if(strcmp(value, "hamming") == 0) {
            sbsc->sbsc.distance_type = 1;
            return true;
        }
        printf("smczsc: unknown distance type: %s\r\n", value);
        return false;
    }
    return false;
}

static void __smczsc_join(search_device* object, object_id id) {
    if(object->module->name != __smczsc_module_name) {
        return;
    }

    auto sbsc = static_cast<__smczsc_device*>(object);
    sbsc->joined_id.push_back(id);
    sbsc->joined_vtype.push_back(__smczsc_makev(sbsc->features[id]));
}

static void __smczsc_join_flush(search_device* object) {
    if(object->module->name != __smczsc_module_name) {
        return;
    }

    auto sbsc = static_cast<__smczsc_device*>(object);

    if(sbsc->tree != nullptr) {
        delete sbsc->tree;
    }
    sbsc->tree = new KDTreeVectorOfVectorsAdaptor<std::vector<vtype>, float, 80, nanoflann::metric_L2>(80, sbsc->joined_vtype, 10);
    sbsc->tree->index->buildIndex();
}

static ssize_t __smczsc_serialize_featrue(FILE* fp, const FeatureDesc& f) {
    ssize_t s1 = serialize_opencv(fp, f.img);
    if(s1 < 0) {
        return -1;
    }

    ssize_t s2 = serialize_opencv(fp, f.M);
    if(s2 < 0) {
        return -1;
    }

    ssize_t s3 = serialize_opencv(fp, f.T);
    if(s3 < 0) {
        return -1;
    }

    return s1 + s2 + s3;
}

static ssize_t __smczsc_serialize(search_device* object, FILE* fp, object_id id) {
    if(object == nullptr || strcmp(object->module->name, __smczsc_module_name) != 0) {
        return 0;
    }

    auto sbsc_object = static_cast<__smczsc_device*>(object);
    if(id >= sbsc_object->features.size()) {
        return -1;
    }

    auto& value = sbsc_object->features[id];
    return __smczsc_serialize_featrue(fp, value);
}

static ssize_t __smczsc_deserialize_featrue(FILE* fp, FeatureDesc& f) {
    ssize_t s1 = deserialize_opencv(fp, f.img);
    if(s1 < 0) {
        return -1;
    }

    ssize_t s2 = deserialize_opencv(fp, f.M);
    if(s2 < 0) {
        return -1;
    }

    ssize_t s3 = deserialize_opencv(fp, f.T);
    if(s3 < 0) {
        return -1;
    }

    return s1 + s2 + s3;
}

static ssize_t __smczsc_deserialize(search_device* object, FILE* fp, object_id& id) {
    if(object == nullptr || strcmp(object->module->name, __smczsc_module_name) != 0) {
        return 0;
    }

    auto sbsc_object = static_cast<__smczsc_device*>(object);
    if(id >= sbsc_object->features.size()) {
        return -1;
    }

    FeatureDesc value;
    auto s = __smczsc_deserialize_featrue(fp, value);
    if(s < 0) {
        return -1;
    }

    sbsc_object->features.push_back(std::move(value));
    id = sbsc_object->features.size() - 1;
    return s;
}

static bool __smczsc_save(search_device* object, object_id id, const char* filename) {
    if(object == nullptr || strcmp(object->module->name, __smczsc_module_name) != 0) {
        return false;
    }

    auto sbsc_object = static_cast<__smczsc_device*>(object);
    if(id >= sbsc_object->features.size()) {
        return false;
    }

    auto& value = sbsc_object->features[id];
    FILE* fp = fopen(filename, "w");
    if(fp == nullptr) {
        return false;
    }

    for(int i = 0; i < value.img.rows; i++) {
        for(int j = 0; j < value.img.cols; j++) {
            if(j != value.img.cols - 1)
                fprintf(fp, "%u,", (unsigned int)value.img.at<uint8_t>(i, j));
            else
                fprintf(fp, "%u", (unsigned int)value.img.at<uint8_t>(i, j));
        }
        fprintf(fp, "\r\n");
    }

    fclose(fp);
    return true;
}
static search_device* __smczsc_create();

static void __smczsc_destroy(search_device* object) {
    if(object->module->name != __smczsc_module_name) {
        return;
    }

    auto sbsc = static_cast<__smczsc_device*>(object);
    if(sbsc->tree != nullptr) {
        delete sbsc->tree;
    }
    delete sbsc;
}

static search_module __smczsc_module = {
    .name = __smczsc_module_name,
   
    .create_object = __smczsc_create_object,
    .search = __smczsc_search,
    .config = __smczsc_config,
    .join = __smczsc_join,
    .join_flush = __smczsc_join_flush,

    .serialize = __smczsc_serialize,
    .deserialize = __smczsc_deserialize,
    .save = __smczsc_save,

    .create = __smczsc_create,
    .destroy = __smczsc_destroy
};

static search_device* __smczsc_create() {
    auto sbsc = new __smczsc_device();
    sbsc->module = &__smczsc_module;
    return sbsc;
}

register_search_module(__smczsc_module);

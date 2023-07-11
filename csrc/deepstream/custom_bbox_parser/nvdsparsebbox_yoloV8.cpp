#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <iostream>

/**
 * Function expected by DeepStream for decoding the TinyYOLOv2 output.
 *
 * C-linkage [extern "C"] was written to prevent name-mangling. This function must return true after
 * adding all bounding boxes to the objectList vector.
 *
 * @param [outputLayersInfo] std::vector of NvDsInferLayerInfo objects with information about the output layer.
 * @param [networkInfo] NvDsInferNetworkInfo object with information about the TinyYOLOv2 network.
 * @param [detectionParams] NvDsInferParseDetectionParams with information about some config params.
 * @param [objectList] std::vector of NvDsInferParseObjectInfo objects to which bounding box information must
 * be stored.
 *
 * @return true
 */

// This is just the function prototype. The definition is written at the end of the file.
extern "C" bool NvDsInferParseCustomYoloV8(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                           NvDsInferNetworkInfo const&            networkInfo,
                                           NvDsInferParseDetectionParams const&   detectionParams,
                                           std::vector<NvDsInferParseObjectInfo>& objectList);

static __inline__ float bbox_clip(const float& val, const float& minVal = 0.f, const float& maxVal = 1280.f)
{
    assert(minVal <= maxVal);
    return std::max(std::min(val, (maxVal - 1)), minVal);
}

static std::vector<NvDsInferParseObjectInfo> decodeYoloV8Tensor(const int*          num_dets,
                                                                const float*        bboxes,
                                                                const float*        scores,
                                                                const int*          labels,
                                                                const unsigned int& img_w,
                                                                const unsigned int& img_h)
{
    std::vector<NvDsInferParseObjectInfo> bboxInfo;
    size_t                                nums = num_dets[0];
    for (size_t i = 0; i < nums; i++) {
        float x0 = (bboxes[i * 4]);
        float y0 = (bboxes[i * 4 + 1]);
        float x1 = (bboxes[i * 4 + 2]);
        float y1 = (bboxes[i * 4 + 3]);
        x0       = bbox_clip(x0, 0.f, img_w);
        y0       = bbox_clip(y0, 0.f, img_h);
        x1       = bbox_clip(x1, 0.f, img_w);
        y1       = bbox_clip(y1, 0.f, img_h);
        NvDsInferParseObjectInfo obj;
        obj.left                = x0;
        obj.top                 = y0;
        obj.width               = x1 - x0;
        obj.height              = y1 - y0;
        obj.detectionConfidence = scores[i];
        obj.classId             = labels[i];
        bboxInfo.push_back(obj);
    }

    return bboxInfo;
}

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomYoloV8(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                           NvDsInferNetworkInfo const&            networkInfo,
                                           NvDsInferParseDetectionParams const&   detectionParams,
                                           std::vector<NvDsInferParseObjectInfo>& objectList)
{

    // Some assertions and error checking.

    if (outputLayersInfo.empty() || outputLayersInfo.size() != 4) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;
        return false;
    }

    // Obtaining the output layer.
    const NvDsInferLayerInfo& num_dets = outputLayersInfo[0];
    const NvDsInferLayerInfo& bboxes   = outputLayersInfo[1];
    const NvDsInferLayerInfo& scores   = outputLayersInfo[2];
    const NvDsInferLayerInfo& labels   = outputLayersInfo[3];

    // num_dets(int) bboxes(float) scores(float) labels(int)
    assert(num_dets.dims.numDims == 2);
    assert(bboxes.dims.numDims == 3);
    assert(scores.dims.numDims == 2);
    assert(labels.dims.numDims == 2);

    // Decoding the output tensor of YOLOv8 to the NvDsInferParseObjectInfo format.
    std::vector<NvDsInferParseObjectInfo> objects = decodeYoloV8Tensor((const int*)(num_dets.buffer),
                                                                       (const float*)(bboxes.buffer),
                                                                       (const float*)(scores.buffer),
                                                                       (const int*)(labels.buffer),
                                                                       networkInfo.width,
                                                                       networkInfo.height);

    objectList.clear();
    objectList = objects;
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8);

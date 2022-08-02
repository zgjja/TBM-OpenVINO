#ifndef MV_CAM_H
#define MV_CAM_H
#include <CameraApi.h>

class MVCam {
public:
    int                 iCameraCounts = 1;
    int                 iStatus = -1;
    tSdkCameraDevInfo   tCameraEnumList;
    int                 hCamera;
    tSdkCameraCapbility tCapability;
    tSdkFrameHead       sFrameHead;
    unsigned char       *pbyBuffer;
    int                 channel = 3;
    tSdkImageResolution sRoiResolution = {0};
    unsigned char       *g_pBgrBuffer;
    bool                shut_down = false;

    MVCam();
    ~MVCam();
    int init_mv_cam();
    void get();
    void get_information();
};
#endif

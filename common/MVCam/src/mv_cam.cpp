#include <iostream>
#include <memory>
#include <stdlib.h>
#include <mv_cam.h>

using namespace std;

MVCam::MVCam() {}

int MVCam::init_mv_cam() {
    CameraSdkInit(1);
    this->iStatus = CameraEnumerateDevice(&this->tCameraEnumList, &this->iCameraCounts);
#ifdef DEBUG_MVCAM
    cout << "state = " << this->iStatus << '\n'
         << "count = " << this->iCameraCounts << '\n';
    for (int i = 0; i < 20; ++i) {
        this->sRoiResolution.iIndex = i;
        CameraSetImageResolution(this->hCamera, &this->sRoiResolution);
        cout << "\ncam default set for idx " << '\n'
        this->get_information();
    }
#endif

    if (this->iCameraCounts == 0) return CAMERA_STATUS_FAILED;
    this->iStatus = CameraInit(&this->tCameraEnumList, -1, -1, &this->hCamera);
    if(this->iStatus != CAMERA_STATUS_SUCCESS) return CAMERA_STATUS_FAILED;

    CameraGetCapability(this->hCamera, &this->tCapability);

    // this->sRoiResolution.iIndex = 9;

    int roi = 256;
    this->sRoiResolution.iIndex = 0xff;
    // strcpy(this->sRoiResolution.acDescription, "My own ROI settings");
    this->sRoiResolution.uBinAverageMode = 0;
    this->sRoiResolution.uBinSumMode = 0;
    this->sRoiResolution.uResampleMask = 0;
    this->sRoiResolution.uSkipMode = 0;
    
    this->sRoiResolution.iHOffsetFOV = (this->tCapability.sResolutionRange.iWidthMax - roi) / 2;
    this->sRoiResolution.iVOffsetFOV = (this->tCapability.sResolutionRange.iHeightMax - roi) / 2;
    this->sRoiResolution.iWidth = roi;
    this->sRoiResolution.iHeight = roi;
    this->sRoiResolution.iWidthFOV = roi;
    this->sRoiResolution.iHeightFOV = roi;

    this->sRoiResolution.iWidthZoomHd = 0;
    this->sRoiResolution.iHeightZoomHd = 0;
    this->sRoiResolution.iWidthZoomSw = 0;
    this->sRoiResolution.iHeightZoomSw = 0;

    unsigned int buffer_size = this->tCapability.sResolutionRange.iWidthMax * this->tCapability.sResolutionRange.iHeightMax * 3;
    this->g_pBgrBuffer = (unsigned char*)CameraAlignMalloc(buffer_size, 16);

    CameraSetAeState(this->hCamera, true);
    CameraSetIspOutFormat(this->hCamera, CAMERA_MEDIA_TYPE_BGR8);  // CAMERA_MEDIA_TYPE_RGB8
    CameraSetAeState(this->hCamera, true);
    CameraSetAntiFlick(this->hCamera, true);
    if (CameraSetImageResolution(this->hCamera, &this->sRoiResolution) != CAMERA_STATUS_SUCCESS)
        return CAMERA_STATUS_FAILED;
    this->get_information();
    CameraPlay(this->hCamera);
    return CAMERA_STATUS_SUCCESS;
}

void MVCam::get() {
    if (!this->shut_down &&
        CameraGetImageBuffer(this->hCamera, &this->sFrameHead, &this->pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS)
    {
        CameraImageProcess(this->hCamera, this->pbyBuffer, this->g_pBgrBuffer, &this->sFrameHead);
        CameraImageOverlay(this->hCamera, this->g_pBgrBuffer, &this->sFrameHead);
        CameraReleaseImageBuffer(this->hCamera, this->pbyBuffer);
    }
}

void MVCam::get_information() {
    tSdkImageResolution tmp;
    if (CameraGetImageResolution(this->hCamera, &tmp) == CAMERA_STATUS_SUCCESS) {
        cout << "\nPrinting cam information:"
             << tmp.iHOffsetFOV << ' '    
             << tmp.iVOffsetFOV << ' '
             << tmp.iWidthFOV << ' '
             << tmp.iHeightFOV << ' '
             << tmp.iWidth << ' '
             << tmp.iHeight << ' '
             << "Description " << tmp.acDescription << '\n';
    } else cout << "Error occured when try acquiring cam ROI settings";
}

MVCam::~MVCam() {
    CameraUnInit(this->hCamera);
    CameraAlignFree(this->g_pBgrBuffer);
}
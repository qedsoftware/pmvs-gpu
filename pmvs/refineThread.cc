#include "refineThread.h"
#include "findMatch.h"
#include "optim.h"

using namespace PMVS3;

CrefineThread::CrefineThread(int numPostProcessThreads, CasyncQueue<RefineWorkItem> &postProcessQueue, CfindMatch &findMatch) : 
        m_workQueue(REFINE_QUEUE_LENGTH),
        m_postProcessQueue(postProcessQueue),
        m_numPostProcessThreads(numPostProcessThreads),
        m_optim(findMatch.m_optim),
        m_fm(findMatch),
        m_numTasks(0),
        m_initialized(false)
{
    m_idleVec = {0,0,0,-1};
    m_encodedVecs = (cl_double4 *)malloc(REFINE_MAX_TASKS*sizeof(cl_double4));
}

CrefineThread::~CrefineThread() {
    if(m_initialized) {
        printf("stopping refine thread\n");
        RefineWorkItem workItem;
        workItem.status = REFINE_ALL_TASKS_COMPLETE;
        enqueueWorkItem(workItem);
        pthread_join(m_refineThread, NULL);
        destroyCL();
        m_initialized = false;
    }
    free(m_encodedVecs);
}

void CrefineThread::init() {
    if(!m_initialized) {
        initCL();
        pthread_create(&m_refineThread, NULL, threadLoopTmp, (void*)this);
        m_initialized = true;
    }
}

void CrefineThread::initCL() {
    cl_uint numPlatforms, numDevices;
    cl_int cl_err;
    cl_platform_id platforms[1];
    cl_device_id devices[1];
    clGetPlatformIDs(1, platforms, &numPlatforms);
    const cl_context_properties cl_props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0};
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, devices, &numDevices);
    m_clCtx = clCreateContext(cl_props, 1, devices, NULL, NULL, &cl_err);
    if(cl_err == CL_SUCCESS) {
        printf("OpenCL context created successfully\n");
    }
    else {
        printf("OpenCL error creating context %d\n", cl_err);
    }
    m_clDevice = devices[0];

    std::ifstream t("/home/jkevin/src/OpenDroneMap/src/cmvs/program/base/pmvs/refinePatch.cl");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string pstr = buffer.str();
    strSubstitute(pstr, "<WSIZE>", m_fm.m_wsize);
    const char *pcstr = pstr.c_str();
    size_t pstrlen = pstr.length();
    cl_int clErr;
    m_clProgram = clCreateProgramWithSource(m_clCtx, 1, &pcstr, &pstrlen, &clErr);
    //printf("%s\n", pcstr);
    printf("created cl program %d\n", clErr);

    clBuildProgram(m_clProgram, 1, &m_clDevice, NULL, NULL, NULL);
    cl_build_status buildStatus;
    clGetProgramBuildInfo(m_clProgram, m_clDevice,
            CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), 
            &buildStatus, NULL);
    if(buildStatus != CL_BUILD_SUCCESS) {
        printf("error building program %d\n", buildStatus);
        char *buildLog = (char *)malloc(50*1024);
        clErr = clGetProgramBuildInfo(m_clProgram, m_clDevice,
                CL_PROGRAM_BUILD_LOG, 50*1024, buildLog, NULL);
        printf("got build info %d\n", clErr);
        printf("%s\n", buildLog);
        free(buildLog);
        std::exit(0);
    }
    else {
        printf("successfully built program\n");
    }

    m_clKernel = clCreateKernel(m_clProgram, "refinePatch", &clErr);
    if(clErr < 0) {
        printf("error creating opencl kernel %d\n", clErr);
    }
    else {
        printf("successfully created command queue\n");
    }

    m_clQueue = clCreateCommandQueue(m_clCtx, m_clDevice, 0, &clErr);
    if(clErr < 0) {
        printf("error creating command queue %d\n", clErr);
    }

    initCLImageArray();
    initCLImageParams();
    initCLPatchParams();

    // set kernel args
    clErr = clSetKernelArg(m_clKernel, 0, sizeof(cl_mem), &m_clImageArray);
    if(clErr < 0) {
        printf("error setKernelArg 0 %d\n", clErr);
    }
    clErr = clSetKernelArg(m_clKernel, 1, sizeof(cl_mem), &m_clImageParams);
    if(clErr < 0) {
        printf("error setKernelArg 1 %d\n", clErr);
    }
    clErr = clSetKernelArg(m_clKernel, 2, sizeof(cl_mem), &m_clPatchParams);
    if(clErr < 0) {
        printf("error setKernelArg 2 %d\n", clErr);
    }
    clErr = clSetKernelArg(m_clKernel, 3, sizeof(int), &m_fm.m_level);
    if(clErr < 0) {
        printf("error setKernelArg 3 %d\n", clErr);
    }
    clErr = clSetKernelArg(m_clKernel, 4, sizeof(cl_mem), &m_clEncodedVecs);
    if(clErr < 0) {
        printf("error setKernelArg 4 %d\n", clErr);
    }
}

void CrefineThread::rgbToRGBA(int width, int height, unsigned char *in, unsigned char *out) {
    unsigned char *cin = in;
    unsigned char *cout = out;
    for(int i=0; i<height; i++) {
        for(int j=0; j<width; j++) {
            for(int k=0; k<3; k++) {
                *cout = *cin;
                cin++;
                cout++;
            }
            *cout = 255;
            cout++;
        }
    }
}

void CrefineThread::initCLImageArray() {
    cl_int clErr;
    int maxWidth = 0;
    int maxHeight = 0;
    for(int i=0; i<m_fm.m_num; i++) {
        int cwidth = m_fm.m_pss.m_photos[i].getWidth(m_fm.m_level);
        int cheight = m_fm.m_pss.m_photos[i].getHeight(m_fm.m_level);
        maxWidth = std::max(maxWidth, cwidth);
        maxHeight = std::max(maxHeight, cheight);
    }
    cl_image_format imFormat = {CL_RGBA, CL_UNORM_INT8};
    unsigned char *rgbaBuffer = (unsigned char *)malloc(maxWidth*maxHeight*4);
    cl_image_desc imDesc = {
        CL_MEM_OBJECT_IMAGE2D_ARRAY,
        maxWidth, maxHeight, 1, m_fm.m_num,
        0, 0, 0, 0, NULL};

    m_clImageArray = clCreateImage(m_clCtx,
            CL_MEM_READ_ONLY,
            &imFormat,
            &imDesc, 
            NULL,
            &clErr);
    printf("created CL image array %x\n", clErr);

    for(int i=0; i<m_fm.m_num; i++) {
        int imWidth = m_fm.m_pss.m_photos[i].getWidth(m_fm.m_level);
        int imHeight = m_fm.m_pss.m_photos[i].getHeight(m_fm.m_level);
        // must convert to RGBA because nvidia doesn't support RGB
        rgbToRGBA(imWidth, imHeight, m_fm.m_pss.m_photos[i].imData(m_fm.m_level), rgbaBuffer);
        size_t origin[] = {0,0,i};
        size_t region[] = {imWidth, imHeight, 1};
        clEnqueueWriteImage(m_clQueue, m_clImageArray, CL_TRUE,
                origin, region, 0, 0,
                rgbaBuffer, NULL, 0, NULL);
    }
    free(rgbaBuffer);
}

void CrefineThread::initCLImageParams() {
    cl_int clErr;

    CLImageParams imParams[m_fm.m_num];

    for(int i=0; i<m_fm.m_num; i++) {
        m_optim.setImageParams(i, imParams[i]);
    }
    m_clImageParams = clCreateBuffer(m_clCtx, CL_MEM_READ_ONLY, m_fm.m_num*sizeof(CLImageParams), NULL, &clErr);
    if(clErr < 0) {
        printf("error creating ImageParams buffer %d\n", clErr);
    }
    clErr = clEnqueueWriteBuffer(m_clQueue, m_clImageParams, CL_TRUE, 0, m_fm.m_num*sizeof(CLImageParams), imParams, 0, NULL, NULL);
    if(clErr < 0) {
        printf("error writing ImageParams buffer %d\n", clErr);
    }
}

void CrefineThread::initCLPatchParams() {
  cl_int clErr;

  m_clPatchParams = clCreateBuffer(m_clCtx, CL_MEM_READ_ONLY, REFINE_MAX_TASKS*sizeof(CLPatchParams), NULL, &clErr);
  if(clErr < 0) {
      printf("error createBuffer PatchParams %d\n", clErr);
  }
  m_clEncodedVecs = clCreateBuffer(m_clCtx, CL_MEM_READ_WRITE, REFINE_MAX_TASKS*sizeof(cl_double4), NULL, &clErr);
  if(clErr < 0) {
      printf("error createBuffer EncodedVecs %d\n", clErr);
  }
}

void CrefineThread::destroyCL() {
    clReleaseCommandQueue(m_clQueue);
}

void CrefineThread::refinePatchesGPU() {
  int status;
  cl_int clErr;

  size_t globalWorkOffset[2] = {0,0};
  size_t globalWorkSize[2] = {REFINE_MAX_TASKS*m_fm.m_wsize,m_fm.m_wsize};
  size_t localWorkSize[2] = {m_fm.m_wsize, m_fm.m_wsize};
  //size_t globalWorkSize[2] = {1,1};
  //size_t localWorkSize[2] = {1,1};
  clErr = clEnqueueNDRangeKernel(m_clQueue, m_clKernel, 2, 
          globalWorkOffset, globalWorkSize, localWorkSize,
          0, NULL, NULL);
  if(clErr < 0) {
      printf("error launching kernel %d\n", clErr);
  }
  clErr = clEnqueueReadBuffer(m_clQueue, m_clEncodedVecs, CL_TRUE, 0, REFINE_MAX_TASKS*sizeof(cl_double4), m_encodedVecs, 0, NULL, NULL);
  if(clErr < 0) {
      printf("error reading encoded vecs %d\n", clErr);
  }

  //printf("patch cent %d %f\n", id, m_texsT[id][0][10]);
  
  
  /*
  status = GSL_SUCCESS;

  if (status == GSL_SUCCESS) {
    decode(patch.m_coord, patch.m_normal, p, id);
    
    patch.m_ncc = 1.0 -
      unrobustincc(computeINCC(patch.m_coord,
                               patch.m_normal, patch.m_images, id, 1));
  }
  else
    patch.m_images.clear();   
  
  ++m_status[status + 2];
  */
}

void CrefineThread::enqueueWorkItem(RefineWorkItem &workItem) {
    m_workQueue.enqueue(workItem);
}

void CrefineThread::clearWorkItems() {
    m_workQueue.clear();
}

void *CrefineThread::threadLoopTmp(void *args) {
    ((CrefineThread *)args)->threadLoop();
    return NULL;
}

void CrefineThread::threadLoop() {
    RefineWorkItem workItem;
    for(int i=0; i<REFINE_MAX_TASKS; i++) {
        m_idleTaskIds.push(i);
        writeEncodedVecToBuffer(i, m_idleVec);
    }
    int running = 1;
    while(running) {
        while(m_numTasks < REFINE_MAX_TASKS) {
            if(m_numTasks > 0 && m_workQueue.isEmpty()) break;
            workItem = m_workQueue.dequeue();
            if(workItem.status == REFINE_ALL_TASKS_COMPLETE) {
                running = 0;
                break;
            }
            else if(workItem.status == REFINE_TASK_IGNORE) {
            }
            else {
                addTask(workItem);
                m_numTasks++;
            }
        }
        if(running) {
            iterateRefineTasks();
            checkCompletedTasks();
        }
    }
    while(m_idleTaskIds.size() > 0) {
        m_idleTaskIds.pop();
    }
}

int CrefineThread::getTaskId() {
    int taskId = m_idleTaskIds.front();
    m_idleTaskIds.pop();
    return taskId;
}

void CrefineThread::addTask(RefineWorkItem &workItem) {
    workItem.status = REFINE_TASK_IN_PROGRESS;
    workItem.taskId = getTaskId();
    m_taskMap[workItem.taskId] = workItem;
    writeParamsToBuffer(workItem.taskId, *workItem.patchParams);
    writeEncodedVecToBuffer(workItem.taskId, workItem.encodedVec);
}

void CrefineThread::iterateRefineTasks() {
    std::map<int, RefineWorkItem>::iterator iter;
    // manage global OpenCL buffer here with info about each
    // patch being refined. Call OpenCL kernel to perform
    // several iterations of minimization on all tasks that
    // are in progress.
    //
    refinePatchesGPU();

    // For now just do original refine routine on CPU to test
    // batching code.
    for(iter = m_taskMap.begin(); iter != m_taskMap.end(); iter++) {
        if(iter->second.status == REFINE_TASK_IN_PROGRESS) {
            //refinePatchGPU(*(iter->second.patch), iter->second.id, 100);
            iter->second.numIterations++;
            cl_double4 buff = m_encodedVecs[iter->second.taskId];
            if(buff.w < .001 || iter->second.numIterations >= 200) {
                iter->second.encodedVec = buff;
                //m_optim.finishRefine(*(iter->second.patch), iter->second.id, buff, REFINE_SUCCESS);
                //printf("buffer val %lf %lf %lf %lf\n", buff.x, buff.y, buff.z, buff.w);
                //m_optim.refinePatch(*(iter->second.patch), iter->second.id, 100);
                iter->second.status = REFINE_TASK_COMPLETE;
            }
        }
    }
}

void CrefineThread::checkCompletedTasks() {
    std::map<int, RefineWorkItem>::iterator iter;
    for(iter = m_taskMap.begin(); iter != m_taskMap.end(); iter++) {
        if(iter->second.status == REFINE_TASK_COMPLETE) {
            m_idleTaskIds.push(iter->first);
            m_postProcessQueue.enqueue(iter->second);
            iter->second.status = REFINE_TASK_IGNORE;
            m_numTasks--;
            writeEncodedVecToBuffer(iter->second.taskId, m_idleVec);
        }
    }
}

bool CrefineThread::isWaiting() {
    return m_workQueue.numWaiting() > 0;
}

void CrefineThread::setTaskBufferIdle(int taskId) {
    writeEncodedVecToBuffer(taskId, m_idleVec);
}

void CrefineThread::writeParamsToBuffer(int taskId, CLPatchParams &patchParams) {
    cl_int clErr;
    clErr = clEnqueueWriteBuffer(m_clQueue, m_clPatchParams, false, taskId*sizeof(CLPatchParams), sizeof(CLPatchParams), &patchParams,
            0, NULL, NULL);
    if(clErr < 0) {
        printf("error writing patchParams to buffer %d\n", taskId);
    }
}

void CrefineThread::writeEncodedVecToBuffer(int taskId, cl_double4 &encodedVec) {
    cl_int clErr;
    clErr = clEnqueueWriteBuffer(m_clQueue, m_clEncodedVecs, false, taskId*sizeof(cl_double4), sizeof(cl_double4), &encodedVec,
            0, NULL, NULL);
    if(clErr < 0) {
        printf("error writing encodedVec to buffer %d\n", taskId);
    }
}


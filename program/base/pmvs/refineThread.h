#ifndef PMVS3_REFINE_THREAD_H
#define PMVS3_REFINE_THREAD_H

#include "asyncQueue.h"
#include <pthread.h>
#include <map>
#include "patch.h"
#include <CL/cl.h>
#include <sstream>
#include <boost/shared_ptr.hpp>

namespace PMVS3 {
    enum {REFINE_TASK_INCOMPLETE,
        REFINE_TASK_IN_PROGRESS,
        REFINE_TASK_COMPLETE,
        REFINE_ALL_TASKS_COMPLETE,
        REFINE_TASK_IGNORE,
        REFINE_SUCCESS
    };

#define REFINE_MAX_TASKS 1024
#define REFINE_QUEUE_LENGTH 1400

    typedef struct _CLImageParams {
        cl_float4 projection[3];
        cl_float3 xaxis;
        cl_float3 yaxis;
        cl_float3 zaxis;
        cl_float4 center;
        cl_float ipscale;
    } CLImageParams;

    typedef struct _CLPatchParams {
        cl_float4 center;
        cl_float4 ray;
        cl_float dscale;
        cl_float ascale;
        cl_int nIndexes;
        cl_int indexes[M_TAU_MAX];
    } CLPatchParams;

    typedef boost::shared_ptr<CLPatchParams> PCLPatchParams;

    class RefineWorkItem {
        public:
            RefineWorkItem() : id(-1) {};
            Patch::Ppatch patch;
            PCLPatchParams patchParams;
            cl_float4 encodedVec;
            int id;
            int taskId;
            int status;
            int numIterations;
    };

    class CfindMatch;
    class Coptim;

    class CrefineThread {
        public:
            CrefineThread(int numPostProcessThreads, CasyncQueue<RefineWorkItem> &postProcessQueue, CfindMatch &findMatch);
            ~CrefineThread();
            void init();
            void enqueueWorkItem(RefineWorkItem &workItem);
            void clearWorkItems();
            bool isWaiting();
            int totalIterations() {return m_totalIterations;};

        protected:
            CasyncQueue<RefineWorkItem> m_workQueue;
            std::queue<int> m_idleTaskIds;
            std::map<int, RefineWorkItem> m_taskMap;
            pthread_t m_refineThread;
            pthread_mutex_t m_workQueueLock;
            pthread_mutex_t m_workTaskIdLock;
            CasyncQueue<RefineWorkItem> &m_postProcessQueue;
            Coptim &m_optim;
            CfindMatch& m_fm;
            int m_numPostProcessThreads;
            int m_numTasks;
            bool m_initialized;
            int m_totalIterations;

            //-----------------------------------------------------------------
            // OpenCL
            cl_context m_clCtx;
            cl_device_id m_clDevice;
            cl_program m_clProgram;
            cl_kernel m_clKernel;
            cl_command_queue m_clQueue;

            // CL image array
            cl_mem m_clImageArray;

            // CL params
            cl_mem m_clImageParams;
            cl_mem m_clPatchParams;
            cl_mem m_clEncodedVecs;
            cl_mem m_clSimplexVecs;
            cl_mem m_clSimplexStates;
            cl_float4 m_idleVec;
            cl_float4 *m_encodedVecs;
            cl_int *m_simplexStates;
            cl_int m_simplexStateInitAll;

            void initCL();
            static void rgbToRGBA(int width, int height, unsigned char *in, unsigned char *out);
            void initCLImageArray();
            void initCLImageParams();
            void initCLPatchParams();
            void destroyCL();
            void refinePatchesGPU();

            static void *threadLoopTmp(void *args);
            void threadLoop();
            int getTaskId();
            void addTask(RefineWorkItem &workItem);
            void iterateRefineTasks();
            void checkCompletedTasks();
            void stopPostProcessThreads();
            template <typename T1, typename T2>
            static void strSubstitute(std::string &str, T1 searchStrIn, T2 replaceIn, bool replaceAll = false);
            void setTaskBufferIdle(int taskId);
            void writeParamsToBuffer(int taskId, CLPatchParams &patchParams);
            void writeEncodedVecToBuffer(int taskId, cl_float4 &encodedVec);
            void writeSimplexStatesToBuffer();
            void initializeSimplexState(int taskId);
            void initializeAllSimplexStates();
    };

    template <typename T1, typename T2>
    void CrefineThread::strSubstitute(std::string &str, T1 searchStrIn, T2 replaceIn, bool replaceAll) {
        size_t startPos;
        std::string searchStr(searchStrIn);
        std::stringstream ss;

        ss << replaceIn;

        startPos = str.find(searchStr);
        while(startPos != std::string::npos) {
            str.replace(startPos, searchStr.length(), ss.str());
            if(replaceAll) {
                startPos = str.find(searchStr);
            }
            else {
                startPos = std::string::npos;
            }
        }
    }
  
}
#endif

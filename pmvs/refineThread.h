#ifndef PMVS3_REFINE_THREAD_H
#define PMVS3_REFINE_THREAD_H

#include <pthread.h>
#include <map>
#include "patch.h"
#include "optim.h"
#include "asyncQueue.h"

namespace PMVS3 {
    enum {REFINE_TASK_INCOMPLETE,
        REFINE_TASK_IN_PROGRESS,
        REFINE_TASK_COMPLETE,
        REFINE_ALL_TASKS_COMPLETE,
        REFINE_TASK_IGNORE
    };

    class RefineWorkItem {
        public:
            RefineWorkItem() : id(-1) {};
            Patch::Ppatch patch;
            int id;
            int status;
    };

    class CrefineThread {
        public:
            CrefineThread(int numPostProcessThreads, CasyncQueue<RefineWorkItem> &postProcessQueue, Coptim &optim);
            ~CrefineThread();
            void enqueueWorkItem(RefineWorkItem &workItem);
            void clearWorkItems();
            int getTaskId();
            bool isWaiting();

        private:
            CasyncQueue<RefineWorkItem> m_workQueue;
            CasyncQueue<int> m_idleTaskIds;
            std::map<int, RefineWorkItem> m_taskMap;
            int m_maxTasks;
            pthread_t m_refineThread;
            pthread_mutex_t m_workQueueLock;
            pthread_mutex_t m_workTaskIdLock;
            CasyncQueue<RefineWorkItem> &m_postProcessQueue;
            Coptim &m_optim;
            int m_numPostProcessThreads;
            int m_numTasks;

            static void *threadLoopTmp(void *args);
            void threadLoop();
            void addTask(RefineWorkItem &workItem);
            void iterateRefineTasks();
            void checkCompletedTasks();
            void stopPostProcessThreads();
    };
}
#endif

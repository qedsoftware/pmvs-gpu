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

#define REFINE_MAX_TASKS 1024
#define REFINE_QUEUE_LENGTH 2048

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
            bool isWaiting();

        private:
            CasyncQueue<RefineWorkItem> m_workQueue;
            std::queue<int> m_idleTaskIds;
            std::map<int, RefineWorkItem> m_taskMap;
            pthread_t m_refineThread;
            pthread_mutex_t m_workQueueLock;
            pthread_mutex_t m_workTaskIdLock;
            CasyncQueue<RefineWorkItem> &m_postProcessQueue;
            Coptim &m_optim;
            int m_numPostProcessThreads;
            int m_numTasks;

            static void *threadLoopTmp(void *args);
            void threadLoop();
            int getTaskId();
            void addTask(RefineWorkItem &workItem);
            void iterateRefineTasks();
            void checkCompletedTasks();
            void stopPostProcessThreads();
    };
}
#endif

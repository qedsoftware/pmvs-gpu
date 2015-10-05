#include "refineThread.h"

using namespace PMVS3;

CrefineThread::CrefineThread(int numPostProcessThreads, CasyncQueue<RefineWorkItem> &postProcessQueue, Coptim &optim) : 
        m_workQueue(REFINE_QUEUE_LENGTH),
        m_postProcessQueue(postProcessQueue),
        m_numPostProcessThreads(numPostProcessThreads),
        m_optim(optim),
        m_numTasks(0)
{
    pthread_create(&m_refineThread, NULL, threadLoopTmp, (void*)this);
}

CrefineThread::~CrefineThread() {
    printf("stopping refine thread\n");
    RefineWorkItem workItem;
    workItem.status = REFINE_ALL_TASKS_COMPLETE;
    enqueueWorkItem(workItem);
    pthread_join(m_refineThread, NULL);
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
    int taskId = getTaskId();
    m_taskMap[taskId] = workItem;
}

void CrefineThread::iterateRefineTasks() {
    std::map<int, RefineWorkItem>::iterator iter;
    // manage global OpenCL buffer here with info about each
    // patch being refined. Call OpenCL kernel to perform
    // several iterations of minimization on all tasks that
    // are in progress.
    //
    // For now just do original refine routine on CPU to test
    // batching code.
    for(iter = m_taskMap.begin(); iter != m_taskMap.end(); iter++) {
        if(iter->second.status == REFINE_TASK_IN_PROGRESS) {
            //m_optim.refinePatchGPU(*(iter->second.patch), iter->second.id, 100);
            m_optim.refinePatch(*(iter->second.patch), iter->second.id, 100);
            iter->second.status = REFINE_TASK_COMPLETE;
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
        }
    }
}

bool CrefineThread::isWaiting() {
    return m_workQueue.numWaiting() > 0;
}

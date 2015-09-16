#include "refineThread.h"

PMVS3::CrefineThread(int numExpandThreads, int numRefineThreads) : 
        m_queue(numExpandThreads),
        m_numRefineThreads(numRefineThreads),
        m_isRunning(false)
{
    m_refineThreads = (pthread_t *)malloc(m_numRefineThreads * sizeof(pthread_t));
}

PMVS3::~CrefineThread() {
    stopThreads();
    free(m_refineThreads);
}

void PMVS3::startThreads() {
    for(int i=0; i<m_numRefineThreads; i++) {
        pthread_create(&m_refineThreads[i], NULL, refineThreadTmp, (void*)this);
    }
    m_isRunning = true;
}

void PMVS3::stopThreads() {
    RefineWorkItem workItem;
    if(m_isRunning) {
        workItem.eventType = THREAD_EVENT_STOP;
        for(int i=0; i<m_numRefineThreads; i++) {
            m_queue.enqueue(workItem);
            pthread_join(m_refineThreads[i], NULL);
        }
        m_isRunning = false;
    }
}

void CrefineThread::threadLoopTmp(void *args) {
    ((CrefineThread *)args)->threadLoop();
}

void CrefineThread::threadLoop() {
    refineWorkItem workItem;
    int running = 1;
    while(running) {
        switch(workItem.eventType) {
            case THREAD_EVENT_REFINE_PATCH:
                if(refineQueueFull()) {
                    flushRefineQueue();
                }
                break;
            case THREAD_EVENT_STOP:
                flushRefineQueue();
            default:
                running = 0;
        }
    }
}

void CrefineThread::enqueueWorkItem(refineWorkItem &workItem) {
}

#include <pthread.h>
#include <gsl/gsl_multimin.h>

namespace PMVS3 {
    typedef struct _refineWorkItem {
    } RefineWorkItem;

    class CrefineThread {
        public:
            CrefineThread(int numExpandThreads, int numRefineThreads);
            ~CrefineThread();
            static void threadLoopTmp(void *args);
            void threadLoop();
            void enqueueWorkItem(refineWorkItem &workItem);

        private:
            AsyncQueue m_workQueue;
            bool m_isRunning;
            pthread_t *m_refineThreads;

            void flushRefineQueue();
    };
}

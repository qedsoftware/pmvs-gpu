#ifndef PMVS3_ASYNC_QUEUE_H
#define PMVS3_ASYNC_QUEUE_H
#include <queue>
#include <pthread.h>

namespace PMVS3 {
    template <class T>
    class CasyncQueue
    {
        public:
            CasyncQueue(int maxLength) :
                _maxLength(maxLength)
            {
                pthread_mutex_init(&_queueLock, NULL);
                pthread_cond_init(&_emptyCondition, NULL);
                pthread_cond_init(&_fullCondition, NULL);
                _numWaiting = 0;
            }

            ~CasyncQueue()
            {
                pthread_mutex_destroy(&_queueLock);
                pthread_cond_destroy(&_emptyCondition);
                pthread_cond_destroy(&_fullCondition);
            }

            void enqueue(T t)
            {
                pthread_mutex_lock(&_queueLock);
                if(_maxLength > 0) {
                    while(_queue.size() >= _maxLength) {
                        pthread_cond_wait(&_fullCondition, &_queueLock);
                    }
                }
                _queue.push(t);
                pthread_cond_signal(&_emptyCondition);
                pthread_mutex_unlock(&_queueLock);
            }

            T dequeue()
            {
                pthread_mutex_lock(&_queueLock);
                _numWaiting++;
                while(_queue.empty()) {
                    pthread_cond_wait(&_emptyCondition, &_queueLock);
                }
                T val = _queue.front();
                _queue.pop();
                pthread_cond_signal(&_fullCondition);
                _numWaiting--;
                pthread_mutex_unlock(&_queueLock);
                return val;
            }

            bool isEmpty() {
                pthread_mutex_lock(&_queueLock);
                bool isEmpty = _queue.empty();
                pthread_mutex_unlock(&_queueLock);
                return isEmpty;
            }

            int numWaiting() {
                int rval;
                pthread_mutex_lock(&_queueLock);
                rval = _numWaiting;
                pthread_mutex_unlock(&_queueLock);
                return rval;
            }

            void clear() {
                pthread_mutex_lock(&_queueLock);
                while(!_queue.empty()) {
                    _queue.pop();
                }
                pthread_mutex_unlock(&_queueLock);
            }

        private:
            std::queue<T> _queue;
            pthread_mutex_t _queueLock;
            pthread_cond_t _emptyCondition;
            pthread_cond_t _fullCondition;
            int _maxLength;
            int _numWaiting;
    };
}
#endif

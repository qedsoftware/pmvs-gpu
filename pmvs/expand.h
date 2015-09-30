#ifndef PMVS3_EXPAND_H
#define PMVS3_EXPAND_H

#include <vector>
#include <queue>
#include <list>
#include "patchOrganizerS.h"
#include "refineThread.h"

namespace PMVS3 {
class CfindMatch;
  
class Cexpand {
 public:
  Cexpand(CfindMatch& findMatch);
  ~Cexpand();

  void init(void);
  void run(void);

  float computeRadius(const Patch::Cpatch& patch);
  
 protected:
  int expandSub(const Patch::Ppatch& orgppatch, const int id,
                const Vec4f& canCoord);
  int postProcessSub(const Patch::Ppatch& newppatch, const int id);
  
  int updateCounts(const Patch::Cpatch& patch);
  
  int checkCounts(Patch::Cpatch& patch);
  
  void findEmptyBlocks(const Patch::Ppatch& ppatch,
                       std::vector<std::vector<Vec4f> >& canCoords);
 protected:

  std::priority_queue<Patch::Ppatch, std::vector<Patch::Ppatch>, P_compare>
    m_queue;
  pthread_cond_t m_emptyCondition;
  pthread_mutex_t m_queueLock;
  
  CfindMatch& m_fm;
  
  //-----------------------------------------------------------------
  // thread related
  //-----------------------------------------------------------------  
  void expandThread(void);
  static void* expandThreadTmp(void* arg);

  void postProcessThread(void);
  static void* postProcessThreadTmp(void *arg);

  CasyncQueue<int> m_idQueue;
  CasyncQueue<RefineWorkItem> m_postProcessQueue;
  CrefineThread m_refineThread;
  int m_numPatchesInFlight;

  // Number of trials
  std::vector<int> m_ecounts;
  // Number of failures in the prep
  std::vector<int> m_fcounts0;
  // Number of failures in the post processing
  std::vector<int> m_fcounts1;
  // Number passes
  std::vector<int> m_pcounts;

};
};

#endif // PMVS3_EXPAND_H

#ifndef CORE_H
#define CORE_H

#include <queue>
#include <vector>
#include <unordered_map>
#include "trace.h"
#include "macsim.h"
#include "cache.h"

class macsim;
class cache_c;

struct ExecutionData {
    int warp_id;
    int dest_reg;
    int timestamp;
};

class core_c {
public:
  int ld_req_cnt = 0;                   // Number of load requests
  int st_req_cnt = 0;                   // Number of store requests
  int tensor_instr_cnt = 0;             // Number of tensor instructions
  int c_running_block_num = 0;
  int c_fetching_block_id = -1;

  // Memory responses recieved
  std::queue<int> c_memory_responses;

  // Active warp pool
  std::vector<warp_s*> c_dispatched_warps;

  // Suspended warps pool: warps waiting for response from memory
  std::unordered_map<int, warp_s*> c_suspended_warps;

  // Structure holding the current executing instructions
  std::vector<ExecutionData> c_exec_buffer;

  //////////////////////////////////////////////////////////////////////////////
  
  // Create a new core object
  core_c(macsim* gpusim, int core_id, sim_time_type cur_cycle);
  
  // Destroy core object
  ~core_c();

  // Attach L2 cache
  void attach_l2_cache(cache_c * cache_ptr);

  // Is the core retired?
  bool is_retired();

  // Get number of cycles elapsed for core
  sim_time_type get_cycle();

  // Get number of instructions retired by core
  int get_insts();

  // Get number of stalled cycles
  sim_time_type get_stall_cycles();

  // Get number of warps currently on the core (running/active/suspended)
  int get_running_warp_num();

  // Get maximum number of warps that the core can hold
  int get_max_running_warp_num();

  // Run one cycle
  void run_a_cycle();


private:
  friend class macsim;

  int core_id = -1;             // Core ID
  macsim* gpusim;               // Pointer to Macsim instance
  cache_c* c_l2cache;           // Pointer to L2 $
  
  cache_c* c_l1cache;           // Ptr to L1 cache
  bool ENABLE_CACHE;
  bool ENABLE_CACHE_LOG;
  int l1cache_size;
  int l1cache_assoc;
  int l1cache_line_size; 
  int l1cache_banks;

  uint64_t num_vta_hits=0;                // Counter to keep track of VTA hits

  bool c_retire = false;                  // Has the core retired?
  sim_time_type c_cycle = 0;              // Number of cycles elapsed
  sim_time_type stall_cycles = 0;         // Counts number of stalled cycles
  uint64_t inst_count_total = 0;          // Total number of instructions executed by core 
  const int c_max_running_warp_num = 4;   // Maximum number of warps that can run on a core.

  // Pointer to currently running warp
  warp_s* c_running_warp = NULL;

  // Warp scheduler
  bool schedule_warps(Warp_Scheduling_Policy_Types policy);

  // Round Robin (RR) warp scheduler
  bool schedule_warps_rr();

  // Greedy Then Oldest (GTO) warp scheduler
  bool schedule_warps_gto();

  // Greedy Then Oldest (CCWS) warp scheduler
  bool schedule_warps_ccws();

  // Send a memory request
  bool send_mem_req(int wid, trace_info_nvbit_small_s* trace_info, bool enable_cache);

  // add the completion cycle timestamp of compute instructions to the execution buffer
  bool add_insts_to_exec_buffer(int completion_cycle, int warp_id, int dest_reg);

  // remove an entry if its completion timestamp is less than or equal to current cycle
  void remove_insts_in_exec_buffer(int current_cycle);

  // check for register dependencies
  bool check_dependency(warp_s* warp);
};

#endif
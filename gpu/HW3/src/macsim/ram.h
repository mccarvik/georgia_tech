#ifndef RAM_H
#define RAM_H

#include <stdint.h>
#include <queue>
#include "sim_defs.h"

using namespace std;

struct RAM_request
{
    uint64_t addr;
    bool is_store;
    int access_sz;
    sim_time_type req_time;
    int core_id;
    int warp_id;
    uint64_t request_id;
};

struct RAM_response
{
    uint64_t request_id;
    int core_id;
    int warp_id;
};

class RAM {
    public:
    RAM(uint64_t latency=200):
        latency(latency)
    {}
    ~RAM(){}

    // Run a cycle
    void run_a_cycle();

    // setup request and response queues
    void set_queues(queue<RAM_request>* req_queue_ptr, queue<RAM_response>* resp_queue_ptr);

    private:
    queue<RAM_request>* request_queue_ptr;
    queue<RAM_response>* response_queue_ptr;
    uint64_t ncycles = 0;
    uint64_t latency;
};

#endif // RAM_H
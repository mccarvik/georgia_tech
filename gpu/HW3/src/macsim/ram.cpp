#include "sim_defs.h"
#include "ram.h"

void RAM::run_a_cycle(){
    if(request_queue_ptr->size() > 0) {
        RAM_request req = request_queue_ptr->front();

        // return responses if t_request + latency is reached
        if (ncycles > req.req_time + latency){
            request_queue_ptr->pop();
            RAM_response resp = {
                .request_id = req.request_id,
                .core_id = req.core_id,
                .warp_id = req.warp_id
            };
            response_queue_ptr->push(resp);
        }
    }
    ncycles++;
}

void RAM::set_queues(queue<RAM_request>* req_queue_ptr, queue<RAM_response>* resp_queue_ptr) {
    request_queue_ptr = req_queue_ptr;
    response_queue_ptr = resp_queue_ptr;
}

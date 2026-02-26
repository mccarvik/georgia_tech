#include "ccws_vta.h"

ccws_vta::ccws_vta(int num_entries): 
    num_entries(num_entries)
{
    // Allocate space for VTA entries for warp
    entries.resize(num_entries);
}

ccws_vta::~ccws_vta(){}

bool ccws_vta::access(uint64_t tag, bool update){
    time++;
    for(auto it=entries.begin(); it!=entries.end(); it++) {
        if(it->valid && it->tag == tag){
            if(update) { 
                it->lru_timestamp = time;
            }
            return true; // Hit
        }
    }

    // Miss
    return false;
}

void ccws_vta::insert(uint64_t tag) {
    time++;
    // Find lru element
    int oldest=0;
    for(int i=0; i<num_entries; i++){
        if(entries[i].lru_timestamp < entries[oldest].lru_timestamp)
            oldest = i;
    }
    
    // insert at oldest
    entries[oldest].valid=true;
    entries[oldest].tag=tag;
    entries[oldest].lru_timestamp=time;
}

void ccws_vta::print(){
    for(auto x: entries){
        printf("(%d:%lx:%ld) ", x.valid, x.tag, x.lru_timestamp);
    }
}
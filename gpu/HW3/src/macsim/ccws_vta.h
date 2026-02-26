#pragma once

#include <stdint.h>
#include <vector>
#include "sim_defs.h"

struct vta_entry{
    bool valid =false;
    uint64_t tag = 0;
    sim_time_type lru_timestamp=0;
};

// CCWS Victim Tag Array
class ccws_vta{
public:
    int num_entries = 0;
    sim_time_type time = 10;

    ccws_vta(int num_entries);
    ~ccws_vta();

    // Access VTA: returns true if hit
    bool access(uint64_t tag, bool update=true);

    // Insert a tag in VTA
    void insert(uint64_t tag);

    // Print VTA entries
    void print();
private:
    std::vector<vta_entry> entries;
};
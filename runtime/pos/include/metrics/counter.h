#pragma once

#include <iostream>
#include <unordered_map>
#include <string>

#include "pos/include/common.h"


template<typename K>
class POSMetrics_CounterList {
 public:
    POSMetrics_CounterList(){}
    ~POSMetrics_CounterList() = default;


    inline void add_counter(K index){
        auto it = this->_map.find(index);
        if(unlikely(it == this->_map.end())){
            this->_map[index] = 1;
        } else {
            this->_map[index] += 1;
        }
    }


    inline uint64_t get_counter(K index){
        auto it = this->_map.find(index);
        if(unlikely(it == this->_map.end())){
            return 0;
        } else {
            return it->second;
        }
    }


    inline void reset_counter(K index){
        auto it = this->_map.find(index);
        if(likely(it != this->_map.end())){
            it->second = 0;
        }
    }


    inline void reset_counters(){ this->_map.clear(); }


    inline std::string str(std::unordered_map<K,std::string> counter_names){
        std::string print_string("");
        typename std::unordered_map<K, std::string>::iterator name_map_iter;
        typename std::unordered_map<K, uint64_t>::iterator map_iter;

        for(map_iter = this->_map.begin(); map_iter != this->_map.end(); map_iter++){
            POS_ASSERT(counter_names.count(map_iter->first) > 0);
            print_string += std::string("[Counter Metric Report] ")
                            + counter_names[map_iter->first] + std::string(": ");
            print_string += std::to_string(map_iter->second);
            print_string += std::string("\n");
            counter_names.erase(map_iter->first);
        }

        for(name_map_iter=counter_names.begin(); name_map_iter!=counter_names.end(); name_map_iter++){
            print_string += std::string("[Reducer Metric Report] ") + counter_names[name_map_iter->first] + std::string(": N/A\n");
        }

        return print_string;
    }


 private:
    std::unordered_map<K, uint64_t> _map;
};

#pragma once

#include <iostream>
#include <unordered_map>
#include <limits>

#include "pos/include/common.h"


enum pos_metrics_reducer_op_t : uint8_t {
    kPOSMetricReducerOp_Max = 0,
    kPOSMetricReducerOp_Min,
    kPOSMetricReducerOp_Avg
};


template<typename K, typename V>
class POSMetrics_ReducerList {
 public:
    POSMetrics_ReducerList() {}
    ~POSMetrics_ReducerList() = default;


    inline void reduce(K index, V value){
        auto max_it = this->_max_map.find(index);
        if(likely(max_it != this->_max_map.end())){
            auto min_it = this->_min_map.find(index);
            auto avg_it = this->_sum_map.find(index);
            auto avg_counter_it = this->_avg_counter_map.find(index);

            POS_ASSERT(min_it != this->_min_map.end());
            POS_ASSERT(avg_it != this->_sum_map.end());
            POS_ASSERT(avg_counter_it != this->_avg_counter_map.end());
            POS_ASSERT(avg_counter_it->second > 0);

            if(max_it->second < value) max_it->second = value;
            if(min_it->second > value) min_it->second = value;
            avg_it->second += value;
            avg_counter_it->second += 1;
        } else {
            this->_max_map[index] = value;
            this->_min_map[index] = value;
            this->_sum_map[index] = value;
            this->_avg_counter_map[index] = 1;
        }
    }


    template<pos_metrics_reducer_op_t op>
    inline V get_reduce(K index){
        POS_STATIC_ASSERT(
            op == kPOSMetricReducerOp_Max || op == kPOSMetricReducerOp_Min
        );

        if constexpr (op == kPOSMetricReducerOp_Max){
            auto it = this->_max_map.find(index);
            if(likely(it != this->_max_map.end())){
                return it->second;
            } else {
                return std::numeric_limits<V>::min();
            }
        } else if constexpr (op == kPOSMetricReducerOp_Min){
            auto it = this->_min_map.find(index);
            if(likely(it != this->_min_map.end())){
                return it->second;
            } else {
                return std::numeric_limits<V>::max();
            }
        }
    }


    inline double get_reduce_avg(K index){
        auto it = this->_sum_map.find(index);
        if(likely(it != this->_sum_map.end())){
            auto counter_it = this->_avg_counter_map.find(index);
            POS_ASSERT(counter_it != this->_avg_counter_map.end());
            POS_ASSERT(counter_it->second > 0);
            return (double)(it->second) / (double)(counter_it->second);
        } else {
            return (double)(0);
        }
    }


    template<pos_metrics_reducer_op_t op>
    inline void reset_reducer(K index){
       auto max_it = this->_max_map.find(index);
        if(likely(max_it != this->_max_map.end())){
            auto min_it = this->_min_map.find(index);
            auto avg_it = this->_sum_map.find(index);
            auto avg_counter_it = this->_avg_counter_map.find(index);

            POS_ASSERT(min_it != this->_min_map.end());
            POS_ASSERT(avg_it != this->_sum_map.end());
            POS_ASSERT(avg_counter_it != this->_avg_counter_map.end());
            POS_ASSERT(avg_counter_it->second > 0);

            this->_max_map.erase(max_it);
            this->_min_map.erase(min_it);
            this->_sum_map.erase(avg_it);
            this->_avg_counter_map.erase(avg_counter_it);
        }
    }


    inline void reset_reducers(){
        this->_max_map.clear();
        this->_min_map.clear();
        this->_sum_map.clear();
        this->_avg_counter_map.clear();
    }


    inline std::string str(std::unordered_map<K,std::string> reducer_names){
        std::string print_string("");
        typename std::unordered_map<K, std::string>::iterator name_map_iter;
        typename std::unordered_map<K, uint64_t>::iterator max_map_iter, min_map_iter, sum_map_iter;
        typename std::unordered_map<K, V>::iterator avg_counter_map_iter;

        POS_ASSERT(this->_max_map.size() == this->_min_map.size());
        POS_ASSERT(this->_max_map.size() == this->_sum_map.size());
        POS_ASSERT(this->_max_map.size() == this->_avg_counter_map.size());

        for(
            max_map_iter = this->_max_map.begin(),
            min_map_iter = this->_min_map.begin(),
            sum_map_iter = this->_sum_map.begin(),
            avg_counter_map_iter = this->_avg_counter_map.begin()
            ;
            max_map_iter != this->_max_map.end()
            ;
            max_map_iter++,
            min_map_iter++,
            sum_map_iter++,
            avg_counter_map_iter++
        ){
            POS_ASSERT(max_map_iter != this->_max_map.end());
            POS_ASSERT(min_map_iter != this->_min_map.end());
            POS_ASSERT(sum_map_iter != this->_sum_map.end());
            POS_ASSERT(avg_counter_map_iter != this->_avg_counter_map.end());
            POS_ASSERT(reducer_names.count(max_map_iter->first) > 0);
            print_string += std::string("[Reducer Metric Report] ")
                            + reducer_names[max_map_iter->first] + std::string(":\n");
            print_string += std::string("  max: ") + std::to_string(max_map_iter->second) + std::string("\n");
            print_string += std::string("  min: ") + std::to_string(min_map_iter->second) + std::string("\n");
            print_string += std::string("  sum: ") + std::to_string(sum_map_iter->second) + std::string("\n");
            print_string += std::string("  avg: ")
                            + std::to_string((double)(sum_map_iter->second)/(double)(avg_counter_map_iter->second))
                            + std::string("\n");
            reducer_names.erase(max_map_iter->first);
        }

        for(name_map_iter=reducer_names.begin(); name_map_iter!=reducer_names.end(); name_map_iter++){
            print_string += std::string("[Reducer Metric Report] ")
                            + reducer_names[name_map_iter->first] + std::string(":\n");
            print_string += std::string("  max: N/A\n");
            print_string += std::string("  min: N/A\n");
            print_string += std::string("  sum: N/A\n");
            print_string += std::string("  avg: N/A\n");
        }

        return print_string;
    }


 private:
    std::unordered_map<K, uint64_t> _max_map;
    std::unordered_map<K, uint64_t> _min_map;
    std::unordered_map<K, uint64_t> _sum_map;
    std::unordered_map<K, V> _avg_counter_map;
};

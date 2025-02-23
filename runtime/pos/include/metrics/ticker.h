#pragma once

#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <numeric>

#include "pos/include/common.h"
#include "pos/include/utils/timer.h"

template<typename K>
class POSMetrics_TickerList {
 public:
    POSMetrics_TickerList(){}
    ~POSMetrics_TickerList() = default;

    inline void start(K index){
        this->_s_tick_map[index] = POSUtilTscTimer::get_tsc();
    }

    inline uint64_t end(K index){
        uint64_t e_tick = POSUtilTscTimer::get_tsc(), diff;

        auto s_tick_it = this->_s_tick_map.find(index);
        POS_ASSERT(s_tick_it != this->_s_tick_map.end());

        diff = e_tick - s_tick_it->second;
        this->_duration_tick_map[index].push_back(diff);

        return diff;
    }

    inline void add(K index, uint64_t& value){
        this->_duration_tick_map[index].push_back(value);
    }

    inline void get_tick(
        K index, double& avg, uint64_t& min, uint64_t& max, uint64_t& overall,
        uint64_t& p10, uint64_t& p50, uint64_t& p99
    ){
        std::vector<uint64_t> sorted_duration;
        size_t n;

        auto duration_tick_it = this->_duration_tick_map.find(index);
        if(unlikely(duration_tick_it == this->_duration_tick_map.end()))
            goto exit;

        sorted_duration = duration_tick_it->second;
        std::sort(sorted_duration.begin(), sorted_duration.end());
        
        min = sorted_duration.front();
        max = sorted_duration.back();

        n = sorted_duration.size();
        p10 = sorted_duration[static_cast<size_t>(0.1 * (n - 1))];
        p50 = sorted_duration[static_cast<size_t>(0.5 * (n - 1))];
        p99 = sorted_duration[static_cast<size_t>(0.99 * (n - 1))];
        
        overall = std::accumulate(duration_tick_it->second.begin(), duration_tick_it->second.end(), 0ULL);
        avg = static_cast<double>(overall) / n;

    exit:
        ;   
    }

    inline uint64_t get_tick(K index, uint64_t order){
        uint64_t retval = 0;

        auto duration_tick_it = this->_duration_tick_map.find(index);
        if(unlikely(duration_tick_it == this->_duration_tick_map.end()))
            goto exit;

        if(unlikely(duration_tick_it->second.size() <= order))
            goto exit;

        retval = duration_tick_it->second[order];

    exit:
        return retval;
    }


    inline void reset_tickers(){
        this->_s_tick_map.clear();
        this->_duration_tick_map.clear();
    }


    inline std::string str(std::unordered_map<K,std::string> ticker_names){
        std::string print_string("");
        typename std::unordered_map<K, std::string>::iterator name_map_iter;
        typename std::unordered_map<K, std::vector<uint64_t>>::iterator map_iter;
        uint64_t min_tick = 0, max_tick = 0, overall_tick = 0;
        uint64_t p10_tick = 0, p50_tick = 0, p99_tick = 0;
        double avg_tick = 0;

        for(map_iter = this->_duration_tick_map.begin();
            map_iter != this->_duration_tick_map.end();
            map_iter++
        ){
            POS_ASSERT(ticker_names.count(map_iter->first) > 0);
            this->get_tick(map_iter->first, avg_tick, min_tick, max_tick, overall_tick, p10_tick, p50_tick, p99_tick);
            print_string += std::string("[Ticker Metric Report] ")
                            + ticker_names[map_iter->first] + std::string(":\n");
            print_string += std::string("  max: ") 
                            + std::to_string(this->tsc_timer.tick_to_ms(max_tick))
                            + std::string(" ms\n");
            print_string += std::string("  min: ") 
                            + std::to_string(this->tsc_timer.tick_to_ms(min_tick))
                            + std::string(" ms\n");
            print_string += std::string("  avg: ") 
                            + std::to_string(this->tsc_timer.tick_to_ms((uint64_t)(avg_tick)))
                            + std::string(" ms\n");
            print_string += std::string("  sum: ") 
                            + std::to_string(this->tsc_timer.tick_to_ms(overall_tick))
                            + std::string(" ms\n");
            print_string += std::string("  p10: ") 
                            + std::to_string(this->tsc_timer.tick_to_ms(p10_tick))
                            + std::string(" ms\n");
            print_string += std::string("  p50: ") 
                            + std::to_string(this->tsc_timer.tick_to_ms(p50_tick))
                            + std::string(" ms\n");
            print_string += std::string("  p99: ") 
                            + std::to_string(this->tsc_timer.tick_to_ms(p99_tick))
                            + std::string(" ms\n");
            ticker_names.erase(map_iter->first);
        }

        for(name_map_iter=ticker_names.begin(); name_map_iter!=ticker_names.end(); name_map_iter++){
            print_string += std::string("[Ticker Metric Report] ") + name_map_iter->second + std::string(":\n");
            print_string += std::string("  max: N/A ms\n");
            print_string += std::string("  min: N/A ms\n");
            print_string += std::string("  avg: N/A ms\n");
            print_string += std::string("  sum: N/A ms\n");
            print_string += std::string("  p10: N/A ms\n");
            print_string += std::string("  p50: N/A ms\n");
            print_string += std::string("  p99: N/A ms\n");
        }

        return print_string;
    }

    POSUtilTscTimer tsc_timer;

 private:
    std::unordered_map<K, uint64_t> _s_tick_map;
    std::unordered_map<K, std::vector<uint64_t>> _duration_tick_map;
};

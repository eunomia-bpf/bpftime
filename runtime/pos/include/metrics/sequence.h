#pragma once

#include <iostream>
#include <unordered_map>
#include <string>

#include "pos/include/common.h"
#include "pos/include/utils/timer.h"


template<typename K, typename T>
class POSMetrics_SequenceList {
 public:
    POSMetrics_SequenceList(){}
    ~POSMetrics_SequenceList() = default;


    inline void add_spot(K index, T data){
        this->_map[index].insert(
            { this->_tsc_timer.get_relative_tsc(), data }
        );
    }

    inline void get_sequence(K index, std::map<uint64_t, T>& sequence){
        auto it = this->_map.find(index);
        if(unlikely(it != this->_map.end())){
            sequence = it->second;
        }
    }


    inline void reset_sequence(K index){
        auto it = this->_map.find(index);
        if(likely(it != this->_map.end())){
            it->second.clear();
        }
    }


    inline void reset_sequences(){ 
        this->_map.clear(); 
    }


    inline std::string str(std::vector<std::pair<K, std::string>>& sequence_names){
        std::string print_string(""), sequence_name;
        std::set<K> keys;
        K key;
        uint64_t i;
        typename std::unordered_map<K, std::map<uint64_t, T>>::iterator map_iter;

        auto __str_sequence = [&](std::map<uint64_t, T>& sequence) -> std::pair<std::string, std::string> {
            std::string _timespot_string("");
            std::string _value_string("");
            typename std::map<uint64_t, T>::iterator map_iter;

            for(map_iter=sequence.begin(); map_iter!=sequence.end(); map_iter++){
                if(std::next(map_iter) != sequence.end()){
                    _timespot_string += std::to_string(this->_tsc_timer.tick_to_ms(map_iter->first)) + std::string(", ");
                    _value_string += std::to_string(map_iter->second) + std::string(", ");
                } else {
                    _timespot_string += std::to_string(this->_tsc_timer.tick_to_ms(map_iter->first));
                    _value_string += std::to_string(map_iter->second);
                }
            }

            return std::pair<std::string, std::string>(_timespot_string, _value_string);
        };

        for(i=0; i<sequence_names.size(); i++){
            key = sequence_names[i].first;
            keys.insert(key);
            sequence_name = sequence_names[i].second;

            if(this->_map.count(key) > 0){
                auto formated = __str_sequence(this->_map[key]);
                print_string += std::string("[Sequence Metric Report] ") + sequence_name + std::string(": \n");
                print_string += std::string("    timeline (ms): ") + formated.first + std::string("\n");
                print_string += std::string("    values:        ") + formated.second + std::string("\n");
            } else {
                print_string += std::string("[Sequence Metric Report] ") + sequence_name + std::string(": \n");
                print_string += std::string("    timeline: N/A\n");
                print_string += std::string("    values:   N/A\n");
            }
        }

        return print_string;
    }

 private:
    POSUtilTscTimer _tsc_timer;
    std::unordered_map<K, std::map</* timespot */ uint64_t, /* data */ T>> _map;
};

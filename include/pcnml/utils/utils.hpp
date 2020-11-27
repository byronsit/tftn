//
// Created by bohuan on 2019/10/24.
//

#ifndef UTILS_HPP
#define UTILS_HPP

#include <bitset>
#include <array>

class Mask_t{
 public:
  typedef std::bitset<1<<13> BITSET_t;
  typedef std::array<BITSET_t, 1<<12> VIS_t;
  inline
  void Set(const size_t &x,const size_t &y){
    mask_.at(x).set(y, 1);
  }

  inline
  BITSET_t operator [] (const size_t &x) const{
    return mask_.at(x);
  }

  inline
  void Clear(){
    for (auto &it : mask_) {
      it.reset();
    }
  }

 private:
  size_t length_, width_;//长宽
  VIS_t mask_;
};

#endif //UTILS_HPP

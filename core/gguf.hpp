// Minimal standalone GGUF reader (mmap). No Gufo dependency: used by the CPU reference, the GPU engine tests
// and the tools. The Gufo adapter layer maps its own reader onto the same tensor descriptions.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace inkling {

enum GgmlType : uint32_t {
  T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q5_K = 13, T_BF16 = 30, T_NVFP4 = 40,
};

inline const char* type_name(uint32_t t) {
  switch (t) {
    case T_F32: return "F32"; case T_F16: return "F16"; case T_Q8_0: return "Q8_0";
    case T_Q5_K: return "Q5_K"; case T_BF16: return "BF16"; case T_NVFP4: return "NVFP4";
  }
  return "?";
}
// elements per block / bytes per block
inline int type_block(uint32_t t) {
  switch (t) { case T_F32: case T_F16: case T_BF16: return 1; case T_Q8_0: return 32; case T_Q5_K: return 256; case T_NVFP4: return 64; }
  throw std::runtime_error("unsupported ggml type " + std::to_string(t));
}
inline size_t type_bytes(uint32_t t) {
  switch (t) { case T_F32: return 4; case T_F16: case T_BF16: return 2; case T_Q8_0: return 34; case T_Q5_K: return 176; case T_NVFP4: return 36; }
  throw std::runtime_error("unsupported ggml type " + std::to_string(t));
}

struct Tensor {
  std::string name;
  std::vector<int64_t> ne;   // ggml order: ne[0] is the contiguous dimension
  uint32_t type = 0;
  uint64_t offset = 0;       // relative to the data section
  const uint8_t* data = nullptr;
  int64_t nelem() const { int64_t n = 1; for (auto d : ne) n *= d; return n; }
  size_t row_bytes() const { return (size_t)(ne[0] / type_block(type)) * type_bytes(type); }
  size_t nbytes() const { return (size_t)(nelem() / ne[0]) * row_bytes(); }
};

struct Meta {
  bool is_str = false, is_arr = false;
  double num = 0;
  std::string str;
  std::vector<double> arr;
  std::vector<std::string> sarr;
};

class Gguf {
 public:
  explicit Gguf(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    struct stat st; fstat(fd_, &st); size_ = st.st_size;
    base_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (base_ == MAP_FAILED) throw std::runtime_error("mmap failed");
    parse();
  }
  ~Gguf() { if (base_) munmap((void*)base_, size_); if (fd_ >= 0) ::close(fd_); }
  Gguf(const Gguf&) = delete; Gguf& operator=(const Gguf&) = delete;

  const Tensor* find(const std::string& n) const { auto it = tensors.find(n); return it == tensors.end() ? nullptr : &it->second; }
  const Tensor& get(const std::string& n) const { auto* t = find(n); if (!t) throw std::runtime_error("missing tensor " + n); return *t; }
  double num(const std::string& k) const { auto it = meta.find(k); if (it == meta.end()) throw std::runtime_error("missing key " + k); return it->second.num; }
  double num_or(const std::string& k, double d) const { auto it = meta.find(k); return it == meta.end() ? d : it->second.num; }
  const Meta& m(const std::string& k) const { auto it = meta.find(k); if (it == meta.end()) throw std::runtime_error("missing key " + k); return it->second; }
  bool has(const std::string& k) const { return meta.count(k) > 0; }

  std::map<std::string, Meta> meta;
  std::map<std::string, Tensor> tensors;

 private:
  int fd_ = -1; size_t size_ = 0; const uint8_t* base_ = nullptr; size_t o_ = 0;
  template <class T> T rd() { T v; memcpy(&v, base_ + o_, sizeof(T)); o_ += sizeof(T); return v; }
  std::string rstr() { uint64_t n = rd<uint64_t>(); std::string s((const char*)base_ + o_, n); o_ += n; return s; }
  double scalar(uint32_t t) {
    switch (t) {
      case 0: return rd<uint8_t>(); case 1: return rd<int8_t>(); case 2: return rd<uint16_t>(); case 3: return rd<int16_t>();
      case 4: return rd<uint32_t>(); case 5: return rd<int32_t>(); case 6: return rd<float>(); case 7: return rd<uint8_t>();
      case 10: return (double)rd<uint64_t>(); case 11: return (double)rd<int64_t>(); case 12: return rd<double>();
    }
    throw std::runtime_error("bad gguf scalar type");
  }
  void parse() {
    if (memcmp(base_, "GGUF", 4) != 0) throw std::runtime_error("not a GGUF file");
    o_ = 4; uint32_t ver = rd<uint32_t>(); if (ver != 3) throw std::runtime_error("unsupported GGUF version");
    uint64_t nt = rd<uint64_t>(), nk = rd<uint64_t>();
    for (uint64_t i = 0; i < nk; ++i) {
      std::string k = rstr(); uint32_t t = rd<uint32_t>(); Meta v;
      if (t == 8) { v.is_str = true; v.str = rstr(); }
      else if (t == 9) {
        v.is_arr = true; uint32_t et = rd<uint32_t>(); uint64_t n = rd<uint64_t>();
        if (et == 8) { v.sarr.reserve(n); for (uint64_t j = 0; j < n; ++j) v.sarr.push_back(rstr()); }
        else { v.arr.reserve(n); for (uint64_t j = 0; j < n; ++j) v.arr.push_back(scalar(et)); }
      } else v.num = scalar(t);
      meta[k] = std::move(v);
    }
    std::vector<Tensor> list;
    for (uint64_t i = 0; i < nt; ++i) {
      Tensor t; t.name = rstr(); uint32_t nd = rd<uint32_t>();
      for (uint32_t d = 0; d < nd; ++d) t.ne.push_back((int64_t)rd<uint64_t>());
      t.type = rd<uint32_t>(); t.offset = rd<uint64_t>(); list.push_back(std::move(t));
    }
    size_t align = (size_t)(has("general.alignment") ? num("general.alignment") : 32);
    size_t data_off = (o_ + align - 1) / align * align;
    for (auto& t : list) { t.data = base_ + data_off + t.offset; tensors[t.name] = t; }
  }
};

}  // namespace inkling

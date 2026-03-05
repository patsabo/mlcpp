#pragma once
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ml {
class Vector {
public:
  std::vector<double> data;

  explicit Vector(std::size_t n, double val = 0.0) : data(n, val) {}

  Vector(std::initializer_list<double> init) : data(init) {}

  std::size_t size() const { return data.size(); }

  double &operator[](std::size_t i) { return data[i]; }
  const double &operator[](std::size_t i) const { return data[i]; }

  Vector operator+(const Vector &other) const {
    check_size(other);
    Vector result(size());
    for (std::size_t i = 0; i < size(); ++i) {
      result[i] = data[i] + other[i];
    }
    return result;
  }

  Vector operator-(const Vector &other) const {
    check_size(other);
    Vector result(size());
    for (std::size_t i = 0; i < size(); ++i) {
      result[i] = data[i] - other[i];
    }
    return result;
  }

  Vector operator*(double scalar) const {
    Vector result(size());
    for (std::size_t i = 0; i < size(); ++i) {
      result[i] = data[i] * scalar;
    }
    return result;
  }

  Vector &operator+=(Vector &other) {
    check_size(other);
    for (std::size_t i = 0; i < size(); ++i) {
      data[i] += other[i];
    }
    return *this;
  }

  Vector &operator-=(Vector &other) {
    check_size(other);
    for (std::size_t i = 0; i < size(); ++i) {
      data[i] -= other[i];
    }
    return *this;
  }

  double dot(const Vector &other) const {
    check_size(other);
    return std::inner_product(data.begin(), data.end(), other.data.begin(),
                              0.0);
  }

  double norm() const { return std::sqrt(dot(*this)); }

private:
  void check_size(const Vector &other) const {
    if (size() != other.size())
      throw std::invalid_argument(
          "Vector size mismatch: " + std::to_string(size()) + " vs " +
          std::to_string(other.size()));
  }
};

class Matrix {};
} // namespace ml

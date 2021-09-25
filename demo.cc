#include <pybind11/pybind11.h>
#include <sys/stat.h>
#include <direct.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <typeinfo>

#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "include/api/serialization.h"
#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/include/vision.h"

#include <pybind11/stl.h>
#include <pybind11/complext.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>

std::vector<ms::MSTensor> GetImageTensor(const std::vector<py::array> &inputs); //私有
ms::MSTensor GetTensor(const py::array &input);                                 //私有
void TensorInfo(std::string name, std::vector<ms::MSTensor> tensor);

std::vector<ms::MSTensor> GetImageTensor(const std::vector<py::array> &inputs)
{
    // 校验一下输入size和model 定义size是否不一样
    std::vector<ms::MSTensor> inputTensors;
    for (auto iter = inputs.begin(); iter != inputs.end(); iter++)
    {
        ms::MSTensor tensor = GetTensor(*iter);
        inputTensors.emplace_back(tensor);
    }
    TensorInfo(inputTensors);
}

ms::MSTensor GetTensor(const py::array &input)
{
    py::buffer_info buf1 = input.request();
    int data_len = 1;
    for (int cnt = 0; cnt < buf1.ndim; cnt++)
    {
        data_len = data_len * buf1.shape[cnt];
    }
    unsigned char *ptr1 = (unsigned char *)buf1.ptr;
    auto input = ms::MSTensor("input", ms::DataType::kNumberTypeUint64, buf1.shape, buf1.ptr, data_len);
    return input;
}

void TensorInfo(std::string name, std::vector<ms::MSTensor> tensor)
{
    std::cout << "tensor vector " << name << " info=====" << std::endl;
    for (int i = 0; i < tensor.size(); i++)
    {
        std::cout << typeid(tensor[i]).name() << std::endl;
        std::cout << tensor[i].Name() << std::endl;
        std::cout << int(tensor[i].DataType()) << std::endl;
        for (int j = 0; j < tensor[i].Shape().size(); ++j)
        {
            std::cout << "Shape" << i << " " << tensor[i].Shape()[j] << std::endl;
        }
    }
    std::cout << "=========================" << std::endl;
}

enum DataType {
  kMSI_Unknown = 0,
  kMSI_Bool = 1,
  kMSI_Int8 = 2,
  kMSI_Int16 = 3,
  kMSI_Int32 = 4,
  kMSI_Int64 = 5,
  kMSI_Uint8 = 6,
  kMSI_Uint16 = 7,
  kMSI_Uint32 = 8,
  kMSI_Uint64 = 9,
  kMSI_Float16 = 10,
  kMSI_Float32 = 11,
  kMSI_Float64 = 12,
  kMSI_String = 13,  // for model STRING input
  kMSI_Bytes = 14,   // for image etc.
};


static std::vector<ssize_t> GetStrides(const std::vector<ssize_t> &shape, ssize_t item_size) {
  std::vector<ssize_t> strides;
  strides.reserve(shape.size());
  const auto ndim = shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    auto stride = item_size;
    for (size_t j = i + 1; j < ndim; ++j) {
      stride *= shape[j];
    }
    strides.push_back(stride);
  }
  return strides;
}

static std::string GetPyTypeFormat(DataType data_type) {
  switch (data_type) {
    case kMSI_Float16:
      return "e";
    case kMSI_Float32:
      return py::format_descriptor<float>::format();
    case kMSI_Float64:
      return py::format_descriptor<double>::format();
    case kMSI_Uint8:
      return py::format_descriptor<uint8_t>::format();
    case kMSI_Uint16:
      return py::format_descriptor<uint16_t>::format();
    case kMSI_Uint32:
      return py::format_descriptor<uint32_t>::format();
    case kMSI_Uint64:
      return py::format_descriptor<uint64_t>::format();
    case kMSI_Int8:
      return py::format_descriptor<int8_t>::format();
    case kMSI_Int16:
      return py::format_descriptor<int16_t>::format();
    case kMSI_Int32:
      return py::format_descriptor<int32_t>::format();
    case kMSI_Int64:
      return py::format_descriptor<int64_t>::format();
    case kMSI_Bool:
      return py::format_descriptor<bool>::format();
    default:
      std::cout << "Unsupported DataType " << data_type << "." <<std::endl;
      return "";
  }
}

size_t GetTypeSize(DataType type) {
  const std::map<DataType, size_t> type_size_map{
    {kMSI_Bool, sizeof(bool)},       {kMSI_Float64, sizeof(double)},   {kMSI_Int8, sizeof(int8_t)},
    {kMSI_Uint8, sizeof(uint8_t)},   {kMSI_Int16, sizeof(int16_t)},    {kMSI_Uint16, sizeof(uint16_t)},
    {kMSI_Int32, sizeof(int32_t)},   {kMSI_Uint32, sizeof(uint32_t)},  {kMSI_Int64, sizeof(int64_t)},
    {kMSI_Uint64, sizeof(uint64_t)}, {kMSI_Float16, sizeof(uint16_t)}, {kMSI_Float32, sizeof(float)},
  };
  auto it = type_size_map.find(type);
  if (it != type_size_map.end()) {
    return it->second;
  }
  return 0;
}

size_t GetDateType(ms::DataType type) {
  const std::map<ms::DataType, DataType> type_map{
    {ms::DataType::kTypeUnknown, kMSI_Unknown},       {ms::DataType::kObjectTypeString, kMSI_String},   
	{ms::DataType::kNumberTypeBool, kMSI_Bool},
    {ms::DataType::kNumberTypeInt8,kMSI_Int8},   {ms::DataType::kNumberTypeInt16, kMSI_Int16}, 
    {ms::DataType::kNumberTypeInt32,kMSI_Int32},   {ms::DataType::kNumberTypeInt64, kMSI_Int64}, 
    {ms::DataType::kNumberTypeUInt8,kMSI_Uint8},   {ms::DataType::kNumberTypeUInt16, kMSI_UInt16}, 
    {ms::DataType::kNumberTypeUInt32,kMSI_UInt32},   {ms::DataType::kNumberTypeUInt64, kMSI_UInt64},  	 	
    {ms::DataType::kNumberTypeFloat16,kMSI_Float16},   {ms::DataType::kNumberTypeFloat32, kMSI_Float32}, 
    {ms::DataType::kNumberTypeFloat64,kMSI_Float64},
  };
  auto it = type_size_map.find(type);
  if (it != type_size_map.end()) {
    return it->second;
  }
  return 0;
}

size_t GetDateType(ms::DataType type) {
  const std::map<ms::DataType, DataType> type_map{
    {ms::DataType::kTypeUnknown, kMSI_Unknown},       {ms::DataType::kObjectTypeString, kMSI_String},   
	{ms::DataType::kNumberTypeBool, kMSI_Bool},
    {ms::DataType::kNumberTypeInt8,kMSI_Int8},   {ms::DataType::kNumberTypeInt16, kMSI_Int16}, 
    {ms::DataType::kNumberTypeInt32,kMSI_Int32},   {ms::DataType::kNumberTypeInt64, kMSI_Int64}, 
    {ms::DataType::kNumberTypeUInt8,kMSI_Uint8},   {ms::DataType::kNumberTypeUInt16, kMSI_UInt16}, 
    {ms::DataType::kNumberTypeUInt32,kMSI_UInt32},   {ms::DataType::kNumberTypeUInt64, kMSI_UInt64},  	 	
    {ms::DataType::kNumberTypeFloat16,kMSI_Float16},   {ms::DataType::kNumberTypeFloat32, kMSI_Float32}, 
    {ms::DataType::kNumberTypeFloat64,kMSI_Float64},
  };
  auto it = type_size_map.find(type);
  if (it != type_size_map.end()) {
    return it->second;
  }
  return 0;
}

size_t GetDateType(ms::DataType type) {
  const std::map<ms::DataType, DataType> type_map{
    {ms::DataType::kTypeUnknown, kMSI_Unknown},       {ms::DataType::kObjectTypeString, kMSI_String},   
	{ms::DataType::kNumberTypeBool, kMSI_Bool},
    {ms::DataType::kNumberTypeInt8,kMSI_Int8},   {ms::DataType::kNumberTypeInt16, kMSI_Int16}, 
    {ms::DataType::kNumberTypeInt32,kMSI_Int32},   {ms::DataType::kNumberTypeInt64, kMSI_Int64}, 
    {ms::DataType::kNumberTypeUInt8,kMSI_Uint8},   {ms::DataType::kNumberTypeUInt16, kMSI_UInt16}, 
    {ms::DataType::kNumberTypeUInt32,kMSI_UInt32},   {ms::DataType::kNumberTypeUInt64, kMSI_UInt64},  	 	
    {ms::DataType::kNumberTypeFloat16,kMSI_Float16},   {ms::DataType::kNumberTypeFloat32, kMSI_Float32}, 
    {ms::DataType::kNumberTypeFloat64,kMSI_Float64},
  };
  auto it = type_size_map.find(type);
  if (it != type_size_map.end()) {
    return it->second;
  }
  return 0;
}


static bool IsCContiguous(const py::array &input) {
  auto flags = static_cast<unsigned int>(input.flags());
  return (flags & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_) != 0;
}



py::object AsPythonData(ms::MSTensor tensor, bool copy) {

    const auto &tensor_shape = tensor->shape();
    std::vector<ssize_t> shape(tensor_shape.begin(), tensor_shape.end());
    std::vector<ssize_t> strides = GetStrides(shape, GetTypeSize(GetDateType(tensor->data_type())));
    py::buffer_info info(reinterpret_cast<void *>(const_cast<uint8_t *>(tensor->data())),
                         static_cast<ssize_t>(GetTypeSize(GetDateType(tensor->data_type()))), GetPyTypeFormat(GetDateType(tensor->data_type())),
                         static_cast<ssize_t>(tensor_shape.size()), shape, strides);

    if (!copy) {
      py::object self = py::cast(tensor);
      return py::array(py::dtype(info), info.shape, info.strides, info.ptr, self);
    } else {
      return py::array(py::dtype(info), info.shape, info.strides, info.ptr);
    }
}

py::tuple AsNumpyTuple(const std::vector<ms::MSTensor> &instance_data) {
  py::tuple numpy_inputs_tuple(instance_data.size());
  for (size_t i = 0; i < instance_data.size(); i++) {  // inputs
    numpy_inputs_tuple[i] = AsPythonData(instance_data[i], false);
  }
  return numpy_inputs_tuple;
}

std::vector<ms::MSTensor> AsInstanceData(const py::tuple &tuple) {
  std::vector<ms::MSTensor> instance_data;
  for (auto &item : tuple) {
    ms::MSTensor tensor;
    try {
      tensor = MakeTensorNoCopy(py::cast<py::array>(item));
    } catch (const std::runtime_error &error) {
      MSI_LOG_EXCEPTION << "Get illegal result data with type " << py::str(item.get_type()).cast<std::string>();
    }
  }
    instance_data.push_back(tensor);
  }
  return instance_data;
}

/// Creates a Tensor from a numpy array without copyMakeTensorNoCopy
ms::MSTensor PyTensor::MakeTensorNoCopy(const py::array &input) {
  // Check format.
  if (!IsCContiguous(input)) {
    MSI_LOG(EXCEPTION) << "Array should be C contiguous.";
  }
  // Get input buffer info.
  py::buffer_info buf = input.request();
  // Get tensor dtype and check it.
  auto dtype = GetDataType(buf);
  if (dtype == kMSI_Unknown) {
    MSI_LOG(EXCEPTION) << "Unsupported data type!";
  }
  // Make a tensor with shared data with numpy array.
  auto tensor_data = std::make_shared<NumpyTensor>(std::move(buf));
  return tensor_data;
}

DataType GetDataType(const py::buffer_info &buf) {
  std::set<char> fp_format = {'e', 'f', 'd'};
  std::set<char> int_format = {'b', 'h', 'i', 'l', 'q'};
  std::set<char> uint_format = {'B', 'H', 'I', 'L', 'Q'};
  if (buf.format.size() == 1) {
    char format = buf.format.front();
    if (fp_format.find(format) != fp_format.end()) {
      switch (buf.itemsize) {
        case 2:
          return kMSI_Float16;
        case 4:
          return kMSI_Float32;
        case 8:
          return kMSI_Float64;
      }
    } else if (int_format.find(format) != int_format.end()) {
      switch (buf.itemsize) {
        case 1:
          return kMSI_Int8;
        case 2:
          return kMSI_Int16;
        case 4:
          return kMSI_Int32;
        case 8:
          return kMSI_Int64;
      }
    } else if (uint_format.find(format) != uint_format.end()) {
      switch (buf.itemsize) {
        case 1:
          return kMSI_Uint8;
        case 2:
          return kMSI_Uint16;
        case 4:
          return kMSI_Uint32;
        case 8:
          return kMSI_Uint64;
      }
    } else if (format == '?') {
      return kMSI_Bool;
    }
  }
  MSI_LOG(WARNING) << "Unsupported DataType format " << buf.format << " item size " << buf.itemsize;
  return kMSI_Unknown;
}



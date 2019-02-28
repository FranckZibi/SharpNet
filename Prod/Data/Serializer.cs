using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using SharpNet.CPU;
using SharpNet.GPU;

namespace SharpNet.Data
{
    public class Serializer
    {
        private readonly StringBuilder _sb = new StringBuilder();
        public Serializer Add(string description, int value)
        {
            _sb.Append("int;" + description + ";" + value + ";");
            return this;
        }
        public Serializer Add(string description, int[] values)
        {
            _sb.Append("intVector;" + description + ";" + values.Length + ";" + ToString(values)+ ";");
            return this;
        }
        public Serializer Add(string description, double value)
        {
            _sb.Append("double;" + description + ";" + value.ToString(CultureInfo.InvariantCulture) + ";");
            return this;
        }
        public Serializer Add(string description, double[] values)
        {
            _sb.Append("doubleVector;" + description + ";" + values.Length + ";" + ToString(values) + ";");
            return this;
        }
        public Serializer Add(string description, Type value)
        {
            _sb.Append("Type;" + description + ";" + value.Name + ";");
            return this;
        }
        public Serializer Add(string description, string value)
        {
            _sb.Append("string;" + description + ";" + value + ";");
            return this;
        }

        public Serializer Add(string value)
        {
            if (!string.IsNullOrEmpty(value))
            {
                _sb.Append(value);
            }
            return this;
        }
        public Serializer Add(string description, float value)
        {
            _sb.Append("single;" + description + ";" + value.ToString(CultureInfo.InvariantCulture) + ";");
            return this;
        }
        public Serializer Add(string description, float[] values)
        {
            _sb.Append("singleVector;" + description + ";" + values.Length + ";" + ToString(values) + ";");
            return this;
        }
        public Serializer Add(string description, bool value)
        {
            _sb.Append("bool;" + description + ";" + value.ToString(CultureInfo.InvariantCulture) + ";");
            return this;
        }
        public Serializer Add(Tensor value)
        {
            _sb.Append(Serialize(value) + ";");
            return this;
        }
        public override string ToString()
        {
            return _sb.ToString();
        }

        private static T[] Deserialize<T>(string[] splitted, int count, int startIndex, Func<string, T> convert)
        {
            var data = new T[count];
            for (int i = 0; i < count; ++i)
            {
                data[i] = convert(splitted[startIndex+i]);
            }
            return data;
        }

        private static double ParseDouble(string str) {return double.Parse(str, CultureInfo.InvariantCulture);}
        private static float ParseFloat(string str) {return float.Parse(str, CultureInfo.InvariantCulture);}
        private static int ParseInt(string str) {return int.Parse(str, CultureInfo.InvariantCulture);}

        public static IDictionary<string, object> Deserialize(string serialized, GPUWrapper gpuWrapper)
        {
            var result = new Dictionary<string, object>();
            var splitted = serialized.Split(new[] {';'}, StringSplitOptions.RemoveEmptyEntries);
            int startIndex = 0;
            while (startIndex < splitted.Length)
            {
                var type = splitted[startIndex];
                if (string.Equals(type, "double", StringComparison.OrdinalIgnoreCase))
                {
                    ++startIndex;
                    var desc = splitted[startIndex++];
                    var data = ParseDouble(splitted[startIndex++]);
                    result[desc] = data;
                }
                else if (string.Equals(type, "doubleVector", StringComparison.OrdinalIgnoreCase))
                {
                    ++startIndex;
                    var desc = splitted[startIndex++];
                    var count = ParseInt(splitted[startIndex++]);
                    result[desc] = Deserialize(splitted, count, startIndex, ParseDouble);
                    startIndex += count;
                }
                else if (string.Equals(type, "single", StringComparison.OrdinalIgnoreCase))
                {
                    ++startIndex;
                    var desc = splitted[startIndex++];
                    var data = ParseFloat(splitted[startIndex++]);
                    result[desc] = data;
                }
                else if (string.Equals(type, "singleVector", StringComparison.OrdinalIgnoreCase))
                {
                    ++startIndex;
                    var desc = splitted[startIndex++];
                    var count = ParseInt(splitted[startIndex++]);
                    result[desc] = Deserialize(splitted, count, startIndex, ParseFloat);
                    startIndex += count;
                }
                else if (string.Equals(type, "bool", StringComparison.OrdinalIgnoreCase))
                {
                    ++startIndex;
                    var desc = splitted[startIndex++];
                    var data = bool.Parse(splitted[startIndex++]);
                    result[desc] = data;
                }
                else if (string.Equals(type, "string", StringComparison.OrdinalIgnoreCase))
                {
                    ++startIndex;
                    var desc = splitted[startIndex++];
                    var data = splitted[startIndex++];
                    result[desc] = data;
                }
                else if (string.Equals(type, "int", StringComparison.OrdinalIgnoreCase))
                {
                    ++startIndex;
                    var desc = splitted[startIndex++];
                    var data = ParseInt(splitted[startIndex++]);
                    result[desc] = data;
                }
                else if (string.Equals(type, "intVector", StringComparison.OrdinalIgnoreCase))
                {
                    ++startIndex;
                    var desc = splitted[startIndex++];
                    var count = ParseInt(splitted[startIndex++]);
                    result[desc] = Deserialize(splitted, count, startIndex, ParseInt);
                    startIndex += count;
                }
                else if (string.Equals(type, "Type", StringComparison.OrdinalIgnoreCase))
                {
                    ++startIndex;
                    var desc = splitted[startIndex++];
                    var data = splitted[startIndex++]; //!D TODO parse string to 'Type'
                    result[desc] = data;
                }
                else if (string.Equals(type, "GPUTensor", StringComparison.OrdinalIgnoreCase) || string.Equals(type, "CpuTensor", StringComparison.OrdinalIgnoreCase))
                {
                    var data = TensorDeserialize(splitted, gpuWrapper, ref startIndex);
                    result[data.Description] = data;
                }
                else
                {
                    throw new NotImplementedException("don't know how to parse "+type);
                }
            }
            return result;
        }
        private static string Serialize(Tensor t)
        {
            if (t.UseDoublePrecision)
            {
                var data = t.UseGPU?t.AsGPU<double>().DeviceContent():t.AsDoubleCpuContent;
                return Serialize(t.UseGPU, t.Description, typeof(double),t.Shape, ToString(data));
            }
            else
            {
                var data = t.UseGPU ? t.AsGPU<float>().DeviceContent() : t.AsFloatCpuContent;
                return Serialize(t.UseGPU, t.Description, typeof(float), t.Shape, ToString(data));
            }
        }

        private static string ToString(float[] data)
        {
            return string.Join(";", data.Select(x => x.ToString(CultureInfo.InvariantCulture)));
        }

        private static string ToString(double[] data)
        {
            return string.Join(";", data.Select(x => x.ToString(CultureInfo.InvariantCulture)));
        }
        private static string ToString(int[] data)
        {
            return string.Join(";", data.Select(x => x.ToString(CultureInfo.InvariantCulture)));
        }

        private static string Serialize(bool isGpu, string description, Type type, int[] shape, string serializedContent)
        {
            var tensorName = isGpu ? "GPUTensor" : "CpuTensor";
            return tensorName + ";" + description.Replace(";", "_") + ";" + type.Name + ";" + shape.Length + ";" + string.Join(";", shape) + ";" + serializedContent;
        }
        private static Tensor TensorDeserialize(string[] splitted, GPUWrapper gpuWrapper, ref int startIndex)
        {
            bool isGpu = string.Equals(splitted[startIndex++], "GPUTensor", StringComparison.OrdinalIgnoreCase);
            var description = splitted[startIndex++];
            var typeAsString = splitted[startIndex++];
            var dimension = int.Parse(splitted[startIndex++]);
            var shape = new int[dimension];
            for (int i = 0; i < dimension; ++i)
            {
                shape[i] = int.Parse(splitted[startIndex++]);
            }
            int count = Utils.Product(shape);
            if (string.Equals(typeAsString, "double", StringComparison.OrdinalIgnoreCase))
            {
                var data = Deserialize(splitted, count, startIndex, ParseDouble);
                startIndex += count;
                return isGpu ? (Tensor)new GPUTensor<double>(shape, data, description, gpuWrapper) : new CpuTensor<double>(shape, data, description);
            }
            if (string.Equals(typeAsString, "single", StringComparison.OrdinalIgnoreCase))
            {
                var data = Deserialize(splitted, count, startIndex, ParseFloat);
                startIndex += count;
                return isGpu ? (Tensor)new GPUTensor<float>(shape, data, description, gpuWrapper) : new CpuTensor<float>(shape, data, description);
            }
            throw new NotImplementedException("do not know how to parse type " + typeAsString);
        }
    }
}

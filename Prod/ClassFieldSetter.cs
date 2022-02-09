using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace SharpNet
{
    public static class ClassFieldSetter
    {
        #region private fields
        private static readonly Dictionary<Type,  Dictionary<string, FieldInfo>> type2name2FieldInfo = new Dictionary<Type, Dictionary<string, FieldInfo>>();
        #endregion

        public static void Set(object o, IDictionary<string, object> dico)
        {
            Type objectType = o.GetType();
            foreach (var e in dico)
            {
                Set(o, objectType, e.Key, e.Value);
            }
        }
        public static void Set(object o, string fieldName, object fieldValue)
        {
            Set(o, o.GetType(), fieldName, fieldValue);
        }

        public static object Get(object o, string fieldName)
        {
            return GetFieldInfo(o.GetType(), fieldName).GetValue(o);
        }

        /// <summary>
        /// public for testing purpose only
        /// </summary>
        /// <param name="t"></param>
        /// <param name="ignoreDefaultValue"></param>
        /// <param name="mandatoryParametersInConfigFile"></param>
        /// <returns></returns>
        public static string ToConfigContent(object t, bool ignoreDefaultValue, HashSet<string> mandatoryParametersInConfigFile = null)
        {
            var type = t.GetType();
            var defaultT = type.GetConstructor(Type.EmptyTypes).Invoke(null);
            var result = new List<string>();
            foreach (var (parameterName, fieldInfo) in GetFieldName2FieldInfo(type).OrderBy(f => f.Key))
            {
                if (ignoreDefaultValue
                    && Equals(fieldInfo.GetValue(t), fieldInfo.GetValue(defaultT))
                    && (mandatoryParametersInConfigFile == null || !mandatoryParametersInConfigFile.Contains(parameterName))
                   )
                {
                    continue;
                }
                result.Add($"{parameterName} = {Utils.FieldValueToString(fieldInfo.GetValue(t))}");
            }
            return string.Join(Environment.NewLine, result) + Environment.NewLine;
        }



        /// <param name="t"></param>
        /// <param name="path"></param>
        /// <param name="ignoreDefaultValue">if a parameter config is already at its default value, we do not save it</param>
        /// <param name="mandatoryParametersInConfigFile"></param>
        public static void Save<T>(this T t, string path, bool ignoreDefaultValue, HashSet<string> mandatoryParametersInConfigFile = null) where T : new()
        {
            var configContent = ToConfigContent(t, ignoreDefaultValue, mandatoryParametersInConfigFile);
            System.IO.File.WriteAllText(path, configContent);
        }

        #region private methods
        private static void Set(object o, Type objectType, string fieldName, object fieldValue)
        {
            var f = GetFieldInfo(objectType, fieldName);
            if (f.FieldType == typeof(bool))
            {
                SetBoolField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType == typeof(int))
            {
                SetIntField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType == typeof(float))
            {
                SetFloatField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType == typeof(double))
            {
                SetDoubleField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType == typeof(double[]))
            {
                SetDoubleVectorField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType == typeof(string))
            {
                SetStringField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType.IsEnum && fieldValue != null)
            {
                // ReSharper disable once AssignNullToNotNullAttribute
                f.SetValue(o, Enum.Parse(f.FieldType, fieldValue.ToString()));
            }
            else
            {
                throw new Exception($"invalid field {fieldName} with value {fieldValue}");
            }
        }
        //private static object GetValue(object o, Type objectType, string fieldName)
        //{
        //    return GetFieldInfo(objectType, fieldName).GetValue(o);
        //}
        public static FieldInfo GetFieldInfo(Type t, string fieldName)
        {
            return GetFieldName2FieldInfo(t)[fieldName];
        }

        public static bool HasField(Type t, string fieldName)
        {
            return GetFieldName2FieldInfo(t).ContainsKey(fieldName);
        }


        public static IDictionary<string,FieldInfo> GetFieldName2FieldInfo(Type t)
        {
            if (!type2name2FieldInfo.ContainsKey(t))
            {
                var name2FieldInfo = new Dictionary<string, FieldInfo>();
                foreach (var e in t.GetFields(BindingFlags.Public | BindingFlags.Instance))
                {
                    name2FieldInfo[e.Name] = e;
                }
                type2name2FieldInfo[t] = name2FieldInfo;
            }
            return type2name2FieldInfo[t];
        }
        private static double[] ParseDoubleVector(string data)
        {
            if (string.IsNullOrEmpty(data))
            {
                return null;
            }
            return data.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(double.Parse).ToArray();
        }
        private static void SetBoolField(object o, Type objectType, string fieldName, object value)
        {
            var f = GetFieldInfo(objectType, fieldName);
            if (value is string)
            {
                value = bool.Parse((string)value);
            }
            f.SetValue(o, value);
        }
        private static void SetIntField(object o, Type objectType, string fieldName, object value)
        {
            var f = GetFieldInfo(objectType, fieldName);
            if (value is string)
            {
                value = int.Parse((string)value);
            }
            f.SetValue(o, value);
        }
        private static void SetFloatField(object o, Type objectType, string fieldName, object value)
        {
            var f = GetFieldInfo(objectType, fieldName);
            if (value is string)
            {
                value = float.Parse((string)value);
            }
            else if (value is int)
            {
                value = (float)(int)value;
            }
            f.SetValue(o, value);
        }
        private static void SetDoubleField(object o, Type objectType, string fieldName, object value)
        {
            var f = GetFieldInfo(objectType, fieldName);
            if (value is string)
            {
                value = double.Parse((string)value);
            }
            else if (value is int)
            {
                value = (double)(int)value;
            }
            else if (value is float)
            {
                value = (double)(float)value;
            }
            f.SetValue(o, value);
        }
        private static void SetDoubleVectorField(object o, Type objectType, string fieldName, object value)
        {
            var f = GetFieldInfo(objectType, fieldName);
            if (value is string)
            {
                value = ParseDoubleVector((string)value);
            }
            f.SetValue(o, value);
        }
        private static void SetStringField(object o, Type objectType, string fieldName, object value)
        {
            var f = GetFieldInfo(objectType, fieldName);
            f.SetValue(o, value);
        }

        private static T LoadFromConfigContent<T>(string configContent) where T : new()
        {
            var t = new T();
            var dico = new Dictionary<string, object>();
            foreach (var l in configContent.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries))
            {
                var fieldAndValue = l.Split('=', StringSplitOptions.RemoveEmptyEntries).Select(p => p.Trim()).ToArray();
                if (fieldAndValue.Length == 0)
                {
                    continue;
                }
                if (fieldAndValue.Length >= 3)
                {
                    throw new Exception($"Invalid config line {l}");
                }
                dico[fieldAndValue[0]] = fieldAndValue.Length == 2 ? fieldAndValue[1] : "";
            }
            Set(t, dico);
            return t;
        }
        #endregion
    }
}

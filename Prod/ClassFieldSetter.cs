using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Reflection;

namespace SharpNet
{
    public static class ClassFieldSetter
    {
        #region private fields
        private static readonly Dictionary<Type,  Dictionary<string, FieldInfo>> type2name2FieldInfo = new Dictionary<Type, Dictionary<string, FieldInfo>>();
        #endregion

        private static void Set(object o, IDictionary<string, object> dico)
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
        /// <returns></returns>
        public static string ToConfigContent(object t)
        {
            var type = t.GetType();
            var result = new List<string>();
            foreach (var (parameterName, fieldInfo) in GetFieldName2FieldInfo(type).OrderBy(f => f.Key))
            {
                result.Add($"{parameterName} = {Utils.FieldValueToString(fieldInfo.GetValue(t))}");
            }
            return string.Join(Environment.NewLine, result) + Environment.NewLine;
        }

        public static Type GetFieldType(Type t, string fieldName)
        {
            return GetFieldName2FieldInfo(t)[fieldName].FieldType;
        }

        public static IEnumerable<string> FieldNames(Type t)
        {
            return GetFieldName2FieldInfo(t).Keys;
        }



        #region private methods
        private static void Set(object o, Type objectType, string fieldName, object fieldValue)
        {
            var f = GetFieldInfo(objectType, fieldName);
            if (fieldValue.GetType() == f.FieldType)
            {
                f.SetValue(o, fieldValue);
                return;
            }
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
            else if (f.FieldType == typeof(string))
            {
                SetStringField(o, objectType, fieldName, fieldValue);
            }
            else if (f.FieldType.IsEnum)
            {
                // ReSharper disable once AssignNullToNotNullAttribute
                f.SetValue(o, Enum.Parse(f.FieldType, fieldValue.ToString()));
            }
            else if (fieldValue is string && f.FieldType.IsGenericType && f.FieldType.GetGenericTypeDefinition() == typeof(List<>))
            {
                var res = ParseStringToListOrArray((string)fieldValue, f.FieldType.GetGenericArguments()[0], true);
                f.SetValue(o, res);
            }
            else if (fieldValue is string && f.FieldType.IsArray)
            {
                var res = ParseStringToListOrArray((string)fieldValue, f.FieldType.GetElementType(), false);
                f.SetValue(o, res);
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
        private static FieldInfo GetFieldInfo(Type t, string fieldName)
        {
            return GetFieldName2FieldInfo(t)[fieldName];
        }

        private static IDictionary<string,FieldInfo> GetFieldName2FieldInfo(Type t)
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
        private static void SetBoolField(object o, Type objectType, string fieldName, object value)
        {
            if (value is string)
            {
                value = bool.Parse((string)value);
            }
            GetFieldInfo(objectType, fieldName).SetValue(o, value);
        }
        private static void SetIntField(object o, Type objectType, string fieldName, object value)
        {
            if (value is string)
            {
                value = int.Parse((string)value);
            }
            GetFieldInfo(objectType, fieldName).SetValue(o, value);
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

        [SuppressMessage("ReSharper", "PossibleNullReferenceException")]
        private static object ParseStringToListOrArray(string s, Type containedType, bool outputIsList)
        {
            var targetListType = typeof(List<>).MakeGenericType(new[] { containedType });
            var list = (IList)Activator.CreateInstance(targetListType);
            foreach (var e in s.Split(','))
            {
                list.Add(ParseStringToScalar(e, containedType));
            }

            if (outputIsList)
            {
                return list;
            }
            else
            {
                //output is an Array
                var array = Array.CreateInstance(containedType, list.Count);
                list.CopyTo(array, 0);
                return array;
            }
        }


        private static object ParseStringToScalar(string s, Type targetScalarType)
        {
            if (targetScalarType == typeof(bool))
            {
                return bool.Parse(s);
            }
            if (targetScalarType == typeof(int))
            {
                return int.Parse(s);
            }
            if (targetScalarType == typeof(float))
            {
                return float.Parse(s, CultureInfo.InvariantCulture);
            }
            if (targetScalarType == typeof(double))
            {
                return double.Parse(s, CultureInfo.InvariantCulture);
            }
            if (targetScalarType == typeof(string))
            {
                return s;
            }
            if (targetScalarType.IsEnum)
            {
                return Enum.Parse(targetScalarType, s);
            }

            throw new ArgumentException($"can t parse {s} to {targetScalarType}");

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

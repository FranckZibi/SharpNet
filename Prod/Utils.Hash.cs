using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace SharpNet
{
    public static partial class Utils
    {
        public static string ComputeHash(string input, int maxLength)
        {
            // Use input string to calculate MD5 hash
            var sb = new StringBuilder();
            using MD5 md5 = MD5.Create();
            var inputBytes = Encoding.ASCII.GetBytes(input);
            var hashBytes = md5.ComputeHash(inputBytes);

            // Convert the byte array to hexadecimal string
            foreach (var t in hashBytes)
            {
                sb.Append(t.ToString("X2"));
            }

            return sb.ToString().Substring(0, maxLength);
        }
        public static string FieldValueToJsonString(object fieldValue)
        {
            if (fieldValue == null)
            {
                return "";
            }

            if (fieldValue is IList)
            {
                List<string> elements = new();
                foreach (var o in (IList)fieldValue)
                {
                    elements.Add(FieldValueToJsonString(o));
                }
                return "["+string.Join(",", elements)+"]";
            }
            if (fieldValue is bool)
            {
                // ReSharper disable once PossibleNullReferenceException
                return fieldValue.ToString().ToLower();
            }

            var asString = FieldValueToString(fieldValue);
            if (fieldValue is string || fieldValue.GetType().IsEnum)
            {
                asString = "\""+asString+"\"";
            }
            return asString;
        }
        public static string FieldValueToString(object fieldValue)
        {
            if (fieldValue == null)
            {
                return "";
            }
            if (fieldValue is string)
            {
                return (string)fieldValue;
            }
            if (fieldValue is bool ||  fieldValue is int)
            {
                return fieldValue.ToString();
            }
            if (fieldValue is float)
            {
                return ((float)fieldValue).ToString(CultureInfo.InvariantCulture);
            }
            if (fieldValue is double)
            {
                return ((double)fieldValue).ToString(CultureInfo.InvariantCulture);
            }
            if (fieldValue.GetType().IsEnum)
            {
                return fieldValue.ToString();
            }

            if (fieldValue is IList)
            {
                List<string> elements = new();
                foreach (var o in (IList)fieldValue)
                {
                    elements.Add(FieldValueToString(o));
                }
                return string.Join(",", elements);
            }

            throw new ArgumentException($"can transform to string field {fieldValue} of type {fieldValue.GetType()}");
        }
        public static IDictionary<string, object> FromString2String_to_String2Object(IDictionary<string, string> dicoString2String)
        {
            var dicoString2Object = new Dictionary<string, object>();
            foreach (var (key, value) in dicoString2String)
            {
                dicoString2Object[key] = value;
            }
            return dicoString2Object;
        }
    }
}

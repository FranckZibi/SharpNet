using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using SharpNet.MathTools;

namespace SharpNet.HPO
{
    public abstract class AbstractHPO<T> where T : class, new()
    {
        private readonly IDictionary<string, object> _searchSpace;
        private readonly IDictionary<string, List<HyperParameterStatistics>> _parameter2Stats;
        private readonly DoubleAccumulator _allResults = new DoubleAccumulator();

        protected AbstractHPO(IDictionary<string, object> searchSpace)
        {
            _searchSpace = searchSpace;
            _parameter2Stats = new Dictionary<string, List<HyperParameterStatistics>>();

            foreach (var (parameterName, searchSpaceForParameter) in _searchSpace.OrderBy(e => e.Key))
            {
                int searchSpaceSizeForParameter = ComputeParameterValuesLength(searchSpaceForParameter);
                var parameterStatistics = new List<HyperParameterStatistics>();
                while (parameterStatistics.Count < searchSpaceSizeForParameter)
                {
                    parameterStatistics.Add(new HyperParameterStatistics());
                }
                _parameter2Stats[parameterName] = parameterStatistics;
            }
        }

        public void RegisterSearchSpaceResult(int searchSpaceIndex, double result)
        {
            _allResults.Add(result,1);
            foreach (var (parameterName, parameterNameStatistics) in _parameter2Stats.OrderBy(e => e.Key))
            {
                int searchSpaceForParameterLength = ParameterValuesLength(parameterName);
                parameterNameStatistics[searchSpaceIndex % searchSpaceForParameterLength].RegisterResult(result);
                searchSpaceIndex /= searchSpaceForParameterLength;
            }
        }


        public abstract T Next { get; }

        /// <summary>
        /// number of processed search spaces
        /// </summary>
        public int Processed => _allResults.Count;

        public int SearchSpaceSize
        {
            get
            {
                int result = 1;
                foreach (var v in _parameter2Stats.Values)
                {
                    result *= v.Count;
                }
                return result;
            }
        }

        /// <summary>
        /// return the complete set of hyper parameters to be used for 'searchSpaceIndex'
        /// </summary>
        /// <param name="searchSpaceIndex"></param>
        /// <returns></returns>
        public T GetHyperParameters(int searchSpaceIndex)
        {
            Debug.Assert(searchSpaceIndex >= 0);
            Debug.Assert(searchSpaceIndex < SearchSpaceSize);
            var t = new T();
            ClassFieldSetter.Set(t, GetSearchSpaceHyperParameters(searchSpaceIndex));
            return t;
        }

        /// <summary>
        /// return the sub set of hyper parameters for 'searchSpaceIndex' (with their associated values) 
        /// </summary>
        /// <param name="searchSpaceIndex"></param>
        /// <returns></returns>
        private IDictionary<string, object> GetSearchSpaceHyperParameters(int searchSpaceIndex)
        {
            var searchSpaceParameters = new Dictionary<string, object>();
            foreach (var e in _searchSpace.OrderBy(e => e.Key))
            {
                var parameterName = e.Key;
                int parameterValuesLength = ParameterValuesLength(parameterName);
                searchSpaceParameters[parameterName] = ExtractParameterValueForIndex(parameterValuesLength, searchSpaceIndex % parameterValuesLength);
                searchSpaceIndex /= parameterValuesLength;
            }
            return searchSpaceParameters;
        }

        private int ParameterValuesLength(string parameterName)
        {
            return _parameter2Stats[parameterName].Count;
        }

        private static int ComputeParameterValuesLength(object parameterValues)
        {
            if (parameterValues is bool[])
            {
                return ((bool[])parameterValues).Length;
            }
            if (parameterValues is int[])
            {
                return ((int[])parameterValues).Length;
            }
            if (parameterValues is float[])
            {
                return ((float[])parameterValues).Length;
            }
            if (parameterValues is double[])
            {
                return ((double[])parameterValues).Length;
            }
            if (parameterValues is string[])
            {
                return ((string[])parameterValues).Length;
            }
            if (parameterValues is bool || parameterValues is int || parameterValues is float || parameterValues is double || parameterValues is string)
            {
                return 1;
            }
            throw new InvalidEnumArgumentException($"can not get size of {parameterValues}");
        }

        private static object ExtractParameterValueForIndex(object values, int index)
        {
            if (values is bool[])
            {
                return ((bool[])values)[index];
            }
            if (values is int[])
            {
                return ((int[])values)[index];
            }
            if (values is float[])
            {
                return ((float[])values)[index];
            }
            if (values is double[])
            {
                return ((double[])values)[index];
            }
            if (values is string[])
            {
                return ((string[])values)[index];
            }
            if (values is bool || values is int || values is float || values is double || values is string)
            {
                return values;
            }
            throw new InvalidEnumArgumentException($"can not extract value of {values} at index {index}");
        }
    }
}

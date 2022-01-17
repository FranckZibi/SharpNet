using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using SharpNet.MathTools;

namespace SharpNet.HPO
{
    public abstract class AbstractHPO<T> where T : class
    {
        private readonly IDictionary<string, HyperParameterSearchSpace> _searchSpace;
        private readonly Func<T> _createDefaultHyperParameters;
        private readonly DoubleAccumulator _allResults = new DoubleAccumulator();

        protected AbstractHPO(IDictionary<string, object> searchSpace, Func<T> createDefaultHyperParameters)
        {
            _searchSpace = new Dictionary<string, HyperParameterSearchSpace>();
            foreach (var e in searchSpace)
            {
                _searchSpace[e.Key] = new HyperParameterSearchSpace(e.Key, e.Value);
            }
            _createDefaultHyperParameters = createDefaultHyperParameters;
        }


        private string StatisticsDescription()
        {
            string res = "";
            foreach (var e in _searchSpace.OrderBy(e=>e.Key))
            {
                res += "Stats for " + e.Key + ":"+Environment.NewLine+e.Value;
            }
            return res;
        }

        public void Process(int nbCpuCores, Func<T, double> toPerformOnEachHyperParameters, Action<string> log)
        {
            log("Computation(s) will be done on " + nbCpuCores + " cores");
            var cpuTasks = new Task[nbCpuCores];
            for (int i = 0; i < cpuTasks.Length; ++i)
            {
                cpuTasks[i] = new Task(() => PerformActionsInSingleThread(toPerformOnEachHyperParameters, log));
                cpuTasks[i].Start();
            }
            Task.WaitAll(cpuTasks);
        }

        /// <summary>
        /// perform as much actions as possible among 'allActionsToPerform'
        /// </summary>
        /// <param name="toPerformOnEachHyperParameters"></param>
        /// <param name="log"></param>
        private void PerformActionsInSingleThread([NotNull] Func<T, double> toPerformOnEachHyperParameters, [NotNull] Action<string> log)
        {
            for (; ; )
            {
                T next;
                lock (this)
                {
                    next = Next;
                }
                if (next == null)
                {
                    log("Finished processing all Hyper-Parameters");
                    return;
                }
                try
                {
                    log("starting new computation");
                    double result = toPerformOnEachHyperParameters(next);

                    _allResults.Add(result, 1);
                    foreach (var (parameterName, parameterSearchSpace) in _searchSpace)
                    {
                        var parameterValue = ClassFieldSetter.Get(next, parameterName);
                        parameterSearchSpace.RegisterError(parameterValue, result);
                    }
                    log("ended new computation");
                    log($"{Processed} processed search spaces");
                    log(StatisticsDescription());
                }
                catch (Exception e)
                {
                    log(e.ToString());
                    log("ignoring error");
                }
            }
        }

        /// <summary>
        /// number of processed search spaces
        /// </summary>
        private int Processed => _allResults.Count;

        protected int SearchSpaceSize
        {
            get
            {
                int result = 1;
                foreach (var v in _searchSpace.Values)
                {
                    result *= v.Length;
                }
                return result;
            }
        }

        /// <summary>
        /// return the complete set of hyper parameters to be used for 'searchSpaceIndex'
        /// </summary>
        /// <param name="searchSpaceIndex"></param>
        /// <returns></returns>
        protected T GetHyperParameters(int searchSpaceIndex)
        {
            Debug.Assert(searchSpaceIndex >= 0);
            Debug.Assert(searchSpaceIndex < SearchSpaceSize);
            var t = _createDefaultHyperParameters();

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
            foreach (var (parameterName, parameterValues) in _searchSpace.OrderBy(e => e.Key))
            {
                int parameterValuesLength = ParameterValuesLength(parameterName);
                searchSpaceParameters[parameterName] = parameterValues.ExtractParameterValueForIndex(searchSpaceIndex % parameterValuesLength);
                searchSpaceIndex /= parameterValuesLength;
            }
            return searchSpaceParameters;
        }

        private int ParameterValuesLength(string parameterName)
        {
            return _searchSpace[parameterName].Length;
        }


        protected abstract T Next { get; }
    }
}

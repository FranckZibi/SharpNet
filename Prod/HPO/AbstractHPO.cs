using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using JetBrains.Annotations;
using SharpNet.MathTools;

namespace SharpNet.HPO
{
    public abstract class AbstractHPO<T> where T : class
    {
        protected readonly IDictionary<string, HyperParameterSearchSpace> _searchSpace;
        protected readonly Func<T> _createDefaultHyperParameters;
        protected readonly Action<T> _postBuild;
        protected readonly Func<T, bool> _isValid;
        private readonly DoubleAccumulator _allResults = new DoubleAccumulator();

        protected AbstractHPO(IDictionary<string, object> searchSpace, Func<T> createDefaultHyperParameters, Action<T> postBuild, Func<T, bool> isValid)
        {
            _searchSpace = new Dictionary<string, HyperParameterSearchSpace>();
            foreach (var e in searchSpace)
            {
                _searchSpace[e.Key] = new HyperParameterSearchSpace(e.Key, e.Value);
            }
            _createDefaultHyperParameters = createDefaultHyperParameters;
            _postBuild = postBuild;
            _isValid = isValid;
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
                    var sw = Stopwatch.StartNew();
                    double result = toPerformOnEachHyperParameters(next);
                    double elapsedTimeInSeconds = sw.Elapsed.TotalSeconds;

                    _allResults.Add(result, 1);
                    foreach (var (parameterName, parameterSearchSpace) in _searchSpace)
                    {
                        var parameterValue = ClassFieldSetter.Get(next, parameterName);
                        parameterSearchSpace.RegisterError(parameterValue, result, elapsedTimeInSeconds);
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

        protected static IDictionary<string, object> FromString2String_to_String2Object(IDictionary<string, string> dicoString2String)
        {
            var dicoString2Object = new Dictionary<string, object>();
            foreach (var (key, value) in dicoString2String)
            {
                dicoString2Object[key] = value;
            }
            return dicoString2Object;
        }


      

        protected static string ComputeHash(IDictionary<string, string> dico)
        {
            var sb = new StringBuilder();
            foreach (var e in dico.OrderBy(t => t.Key))
            {
                sb.Append(e.Key + ":" + e.Value + " ");
            }
            return Utils.ComputeHash(sb.ToString(), 8);
        }

        private int ParameterValuesLength(string parameterName)
        {
            return _searchSpace[parameterName].Length;
        }


        protected abstract T Next { get; }
    }
}

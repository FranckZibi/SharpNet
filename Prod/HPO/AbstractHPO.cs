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
    public abstract class AbstractHpo<T> where T : class
    {
        protected readonly IDictionary<string, HyperParameterSearchSpace> SearchSpace;
        protected readonly Func<T> CreateDefaultSample;
        protected readonly Action<T> PostBuild;
        protected readonly Func<T, bool> IsValidSample;
        private readonly DoubleAccumulator _allCost = new();

        protected AbstractHpo(IDictionary<string, object> searchSpace, Func<T> createDefaultSample, Action<T> postBuild, Func<T, bool> isValidSample)
        {
            SearchSpace = new Dictionary<string, HyperParameterSearchSpace>();
            foreach (var (hyperParameterName, value) in searchSpace)
            {
                SearchSpace[hyperParameterName] = new HyperParameterSearchSpace(hyperParameterName, value);
            }
            CreateDefaultSample = createDefaultSample;
            PostBuild = postBuild;
            IsValidSample = isValidSample;
        }

        private string StatisticsDescription()
        {
            string res = "";
            foreach (var e in SearchSpace.OrderBy(e=>e.Key))
            {
                res += "Stats for " + e.Key + ":"+Environment.NewLine+e.Value;
            }
            return res;
        }

        public void Process(int nbCpuCores, Func<T, double> objectiveFunction, Action<string> log)
        {
            log("Computation(s) will be done on " + nbCpuCores + " cores");
            var cpuTasks = new Task[nbCpuCores];
            for (int i = 0; i < cpuTasks.Length; ++i)
            {
                cpuTasks[i] = new Task(() => ProcessNextSample(objectiveFunction, log));
                cpuTasks[i].Start();
            }
            Task.WaitAll(cpuTasks);
        }

        /// <summary>
        /// process next available sample
        /// </summary>
        /// <param name="objectiveFunction"></param>
        /// <param name="log"></param>
        private void ProcessNextSample([NotNull] Func<T, double> objectiveFunction, [NotNull] Action<string> log)
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
                    log("Finished processing entire search space");
                    return;
                }
                try
                {
                    log("starting new sample computation");
                    var sw = Stopwatch.StartNew();
                    double cost = objectiveFunction(next);
                    double elapsedTimeInSeconds = sw.Elapsed.TotalSeconds;

                    _allCost.Add(cost, 1);
                    foreach (var (parameterName, parameterSearchSpace) in SearchSpace)
                    {
                        var parameterValue = ClassFieldSetter.Get(next, parameterName);
                        parameterSearchSpace.RegisterCost(parameterValue, cost, elapsedTimeInSeconds);
                    }
                    log("ended new computation");
                    log($"{Processed} processed samples");
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
        private int Processed => _allCost.Count;

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

        protected abstract T Next { get; }
    }
}

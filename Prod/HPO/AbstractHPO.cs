using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using JetBrains.Annotations;
using log4net;
using SharpNet.MathTools;

namespace SharpNet.HPO
{
    public abstract class AbstractHpo<T> where T : class
    {
        protected readonly IDictionary<string, AbstractHyperParameterSearchSpace> SearchSpace;
        protected readonly Func<T> CreateDefaultSample;

        /// <summary>
        /// method to be called ofter building a new sample
        /// it will update it using any needed standardization/ normalization /etc...
        /// return true if everything working OK
        /// return false if the sample is invalid and should be discarded
        /// </summary>
        protected readonly Func<T, bool> PostBuild;
        /// <summary>
        /// maximum number of samples to process
        /// </summary>
        private readonly int _maxSamplesToProcess;
        protected readonly ILog Log;
        protected readonly DoubleAccumulator _allCost = new();
        protected int _nextSampleId = 0;

        /// <summary>
        /// the best sample (lowest cost) found so far (or null if no sample has been analyzed)
        /// </summary>
        // ReSharper disable once MemberCanBePrivate.Global
        public T BestSampleFoundSoFar { get; protected set; } = null;
        /// <summary>
        /// the cost associated with the best sample found sample (or NaN if no sample has been analyzed)
        /// </summary>
        // ReSharper disable once MemberCanBePrivate.Global
        public float CostOfBestSampleFoundSoFar { get; protected set; } = float.NaN;

        protected AbstractHpo(IDictionary<string, object> searchSpace, Func<T> createDefaultSample, Func<T, bool> postBuild, ILog log, int maxSamplesToProcess, HashSet<string> mandatoryCategoricalHyperParameters)
        {
            SearchSpace = new Dictionary<string, AbstractHyperParameterSearchSpace>();

            foreach (var (hyperParameterName, hyperParameterSearchSpace) in searchSpace)
            {
                var isCategoricalHyperParameter = IsCategoricalHyperParameter(typeof(T), hyperParameterName,  mandatoryCategoricalHyperParameters);
                SearchSpace[hyperParameterName] = AbstractHyperParameterSearchSpace.ValueOf(hyperParameterSearchSpace, isCategoricalHyperParameter);
            }
            CreateDefaultSample = createDefaultSample;
            PostBuild = postBuild;
            _maxSamplesToProcess = maxSamplesToProcess;
            Log = log;
        }


        private static bool IsCategoricalHyperParameter(Type sampleType, string hyperParameterName, HashSet<string> mandatoryCategoricalHyperParameters)
        {
            if (mandatoryCategoricalHyperParameters.Contains(hyperParameterName))
            {
                return true;
            }

            var hyperParameterType = ClassFieldSetter.GetFieldInfo(sampleType, hyperParameterName).FieldType;
            if (hyperParameterType == typeof(double) || hyperParameterType == typeof(float) ||  hyperParameterType == typeof(int))
            {
                return false;
            }
            if (hyperParameterType == typeof(string) || hyperParameterType == typeof(bool) || hyperParameterType.IsEnum)
            {
                return true;
            }
            throw new ArgumentException( $"can't determine if {hyperParameterName} ({hyperParameterType}) field of class {sampleType} is categorical");
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

        public void Process(int numModelTrainingInParallel, Func<T, float> objectiveFunction)
        {
            Log.Info("Computation(s) will be done on " + numModelTrainingInParallel + " cores");
            var threadTasks = new Task[numModelTrainingInParallel];
            for (int i = 0; i < threadTasks.Length; ++i)
            {
                threadTasks[i] = new Task(() => ProcessNextSample(objectiveFunction));
                threadTasks[i].Start();
            }
            Task.WaitAll(threadTasks);
        }

        /// <summary>
        /// process next available sample
        /// </summary>
        /// <param name="objectiveFunction"></param>
        private void ProcessNextSample([NotNull] Func<T, float> objectiveFunction)
        {
            for (; ; )
            {
                T sample;
                int sampleId;
                string sampleDescription;
                lock (this)
                {
                    (sample, sampleId, sampleDescription) = Next;
                }
                if (sample == null)
                {
                    Log.Info("Finished processing entire search space");
                    return;
                }
                try
                {
                    Log.Debug($"starting computation of sampleId {sampleId}");

                    var sw = Stopwatch.StartNew();
                    float cost = objectiveFunction(sample);
                    if (BestSampleFoundSoFar == null || float.IsNaN(CostOfBestSampleFoundSoFar) || cost < CostOfBestSampleFoundSoFar)
                    {
                        BestSampleFoundSoFar = sample;
                        CostOfBestSampleFoundSoFar = cost;
                        Log.Info($"new lowest cost {CostOfBestSampleFoundSoFar} with sampleId {sampleId}"+Environment.NewLine+ sampleDescription);
                    }
                    double elapsedTimeInSeconds = sw.Elapsed.TotalSeconds;
                    RegisterSampleCost(sample, sampleId, cost, elapsedTimeInSeconds);
                    Log.Debug("ended new computation");
                    Log.Debug($"{Processed} processed samples");
                    Log.Debug(StatisticsDescription());
                }
                catch (Exception e)
                {
                    Log.Error(e.ToString());
                    Log.Error("ignoring error");
                }

                if (sampleId > _maxSamplesToProcess)
                {
                    Log.Info($"maximum number of samples to process ({_maxSamplesToProcess}) has been reached");
                    return;
                }
            }
        }


        protected virtual void RegisterSampleCost(object sample, int sampleId, float cost, double elapsedTimeInSeconds)
        {
            _allCost.Add(cost, 1);
            RegisterSampleCost(SearchSpace, sample, cost, elapsedTimeInSeconds);
        }


        protected static void RegisterSampleCost(IDictionary<string, AbstractHyperParameterSearchSpace> searchSpace, object sample, float cost, double elapsedTimeInSeconds)
        {
            foreach (var (parameterName, parameterSearchSpace) in searchSpace)
            {
                var parameterValue = ClassFieldSetter.Get(sample, parameterName);
                parameterSearchSpace.RegisterCost(parameterValue, cost, elapsedTimeInSeconds);
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

        protected static string ToSampleDescription(IDictionary<string, string> dico)
        {
            var description = "";
            foreach (var (key, value) in dico.OrderBy(t=>t.Key))
            {
                description+= key+ " = "+value+Environment.NewLine;
            }
            return description.Trim();
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

        protected abstract (T,int, string) Next { get; }
    }
}

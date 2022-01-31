using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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
        private readonly int numModelTrainingInParallel;
        /// <summary>
        /// method to be called ofter building a new sample
        /// it will update it using any needed standardization/ normalization /etc...
        /// return true if everything working OK
        /// return false if the sample is invalid and should be discarded
        /// </summary>
        protected readonly Func<T, bool> PostBuild;
        /// <summary>
        /// maximum time (in seconds) for the HPO
        /// </summary>
        private readonly double _maxAllowedSecondsForAllComputation;

        [NotNull] protected readonly string _workingDirectory;
        private readonly DateTime _creationTime = DateTime.Now;
        protected static readonly ILog Log = LogManager.GetLogger(typeof(AbstractHpo<T>));

        protected readonly DoubleAccumulator _allCost = new();
        protected int _nextSampleId = 0;
        // the last time we have displayed statistics about the search, or null if we have never displayed statistics
        private DateTime? lastTimeDisplayedStatisticsDateTime;


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

        protected AbstractHpo(
            IDictionary<string, object> searchSpace, 
            Func<T> createDefaultSample, 
            Func<T, bool> postBuild, 
            double maxAllowedSecondsForAllComputation, 
            [NotNull] string workingDirectory,
            HashSet<string> mandatoryCategoricalHyperParameters)
        {
            if (!Directory.Exists(workingDirectory))
            {
                Directory.CreateDirectory(workingDirectory);
            }
            Utils.ConfigureGlobalLog4netProperties(workingDirectory, "log");
            Utils.ConfigureThreadLog4netProperties(workingDirectory, "log");

            CreateDefaultSample = createDefaultSample;
            PostBuild = postBuild;
            _maxAllowedSecondsForAllComputation = maxAllowedSecondsForAllComputation;
            _workingDirectory = workingDirectory;
            SearchSpace = new Dictionary<string, AbstractHyperParameterSearchSpace>();
            foreach (var (hyperParameterName, hyperParameterSearchSpace) in searchSpace)
            {
                var isCategoricalHyperParameter = IsCategoricalHyperParameter(typeof(T), hyperParameterName,  mandatoryCategoricalHyperParameters);
                SearchSpace[hyperParameterName] = AbstractHyperParameterSearchSpace.ValueOf(hyperParameterSearchSpace, isCategoricalHyperParameter);
            }

            int coreCount = Utils.CoreCount;
            // ReSharper disable once ConvertToConstant.Local
            // number of parallel threads in each single training
            int numThreadsForEachModelTraining = 1;//single core
            numModelTrainingInParallel = coreCount;

            if (coreCount % numThreadsForEachModelTraining != 0)
            {
                throw new ArgumentException($"invalid number of threads {numThreadsForEachModelTraining} : core count {coreCount} must be a multiple of it");
            }
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

        public void Process(Func<T, float> objectiveFunction)
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
                    Log.Debug("ended new computation");
                    Log.Debug($"{Processed} processed samples");
                    //we display statistics only once every 10s
                    if (lastTimeDisplayedStatisticsDateTime == null ||  (DateTime.Now - lastTimeDisplayedStatisticsDateTime.Value).TotalSeconds > 10)
                    {
                        lastTimeDisplayedStatisticsDateTime = DateTime.Now;
                        Log.Debug(StatisticsDescription());
                    }
                }
                catch (Exception e)
                {
                    Log.Error(e.ToString());
                    Log.Error("ignoring error");
                }

                if (_maxAllowedSecondsForAllComputation > 0 && ((DateTime.Now - _creationTime).TotalSeconds) > _maxAllowedSecondsForAllComputation)
                {
                    Log.Info($"maximum time to process all samples ({_maxAllowedSecondsForAllComputation}s) has been used");
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

        protected static string ToSampleDescription(IDictionary<string, string> dico, T sample)
        {
            var description = "";
            foreach (var (hyperParameterName, _) in dico.OrderBy(t=>t.Key))
            {
                var hyperParameterValueAsString = ClassFieldSetter.FieldValueToString(ClassFieldSetter.Get(sample, hyperParameterName));
                description+= hyperParameterName+ " = "+ hyperParameterValueAsString + Environment.NewLine;
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

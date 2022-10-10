using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using log4net;
using SharpNet.HyperParameters;
using SharpNet.MathTools;

namespace SharpNet.HPO
{
    public abstract class AbstractHpo
    {
        #region private and protected fields
        protected readonly IDictionary<string, AbstractHyperParameterSearchSpace> SearchSpace;
        protected readonly Func<ISample> CreateDefaultSample;
        private readonly int numModelTrainingInParallel;
        [NotNull] protected readonly string _workingDirectory;
        private readonly DateTime _creationTime = DateTime.Now;
        protected static readonly ILog Log = LogManager.GetLogger(typeof(AbstractHpo));
        protected readonly DoubleAccumulator _allAsctualScores = new();
        protected int _nextSampleId = 0;
        // the last time we have displayed statistics about the search, or null if we have never displayed statistics
        private DateTime? lastTimeDisplayedStatisticsDateTime;
        #endregion

        #region public fields
        /// <summary>
        /// the best sample (best score) found so far (or null if no sample has been analyzed)
        /// </summary>
        // ReSharper disable once MemberCanBePrivate.Global
        // ReSharper disable once UnusedAutoPropertyAccessor.Global
        public ISample BestSampleFoundSoFar { get; protected set; }
        /// <summary>
        /// the score associated with the best sample found (or NaN if no sample has been analyzed)
        /// </summary>
        // ReSharper disable once MemberCanBePrivate.Global
        public IScore ScoreOfBestSampleFoundSoFar { get; protected set; }
        #endregion

        #region constructor
        protected AbstractHpo(
            IDictionary<string, object> searchSpace, 
            Func<ISample> createDefaultSample,
            [NotNull] string workingDirectory)
        {
            if (!Directory.Exists(workingDirectory))
            {
                Directory.CreateDirectory(workingDirectory);
            }
            Utils.ConfigureGlobalLog4netProperties(workingDirectory, "log");
            Utils.ConfigureThreadLog4netProperties(workingDirectory, "log");

            CreateDefaultSample = createDefaultSample;
            _workingDirectory = workingDirectory;
            SearchSpace = new Dictionary<string, AbstractHyperParameterSearchSpace>();

            var defaultSample = createDefaultSample();

            foreach (var (hyperParameterName, hyperParameterSearchSpace) in searchSpace)
            {
                var isCategoricalHyperParameter = defaultSample.IsCategoricalHyperParameter(hyperParameterName);
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
        #endregion

        public void Process(Func<ISample, IScore> objectiveFunction, float maxAllowedSecondsForAllComputation = 0 /* no time limit by default */)
        {
            Log.Info("Computation(s) will be done on " + numModelTrainingInParallel + " cores");
            var threadTasks = new Task[numModelTrainingInParallel];
            for (int i = 0; i < threadTasks.Length; ++i)
            {
                threadTasks[i] = new Task(() => ProcessNextSample(objectiveFunction, maxAllowedSecondsForAllComputation));
                threadTasks[i].Start();
            }
            Task.WaitAll(threadTasks);
        }


        protected virtual void RegisterSampleScore(ISample sample, int sampleId, IScore actualScore, double elapsedTimeInSeconds)
        {
            _allAsctualScores.Add(actualScore.Value, 1);
            RegisterSampleScore(SearchSpace, sample, actualScore, elapsedTimeInSeconds);
        }
        protected static void RegisterSampleScore(IDictionary<string, AbstractHyperParameterSearchSpace> searchSpace, ISample sample, [NotNull] IScore actualScore, double elapsedTimeInSeconds)
        {
            foreach (var (parameterName, parameterSearchSpace) in searchSpace)
            {
                var parameterValue = sample.Get(parameterName);
                parameterSearchSpace.RegisterScore(parameterValue, actualScore, elapsedTimeInSeconds);
            }
        }
        protected static string ToSampleDescription(IDictionary<string, string> dico, ISample sample)
        {
            var description = "";
            foreach (var (hyperParameterName, _) in dico.OrderBy(t=>t.Key))
            {
                var hyperParameterValueAsString = Utils.FieldValueToString(sample.Get(hyperParameterName));
                description+= hyperParameterName+ " = "+ hyperParameterValueAsString + Environment.NewLine;
            }
            return description.Trim();
        }
        protected abstract (ISample,int, string) Next { get; }

        /// <summary>
        /// process next available sample
        /// </summary>
        /// <param name="objectiveFunction"></param>
        /// <param name="maxAllowedSecondsForAllComputation"></param>
        private void ProcessNextSample([NotNull] Func<ISample, IScore> objectiveFunction, float maxAllowedSecondsForAllComputation)
        {
            for (; ; )
            {
                ISample sample;
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
                    var score = objectiveFunction(sample);
                    if (score != null && score.IsBetterThan(ScoreOfBestSampleFoundSoFar))
                    {
                        Log.Info($"new best score {score} with sampleId {sampleId} (was {ScoreOfBestSampleFoundSoFar})" + Environment.NewLine + sampleDescription);
                        BestSampleFoundSoFar = sample;
                        ScoreOfBestSampleFoundSoFar = score;
                    }
                    double elapsedTimeInSeconds = sw.Elapsed.TotalSeconds;
                    RegisterSampleScore(sample, sampleId, score, elapsedTimeInSeconds);
                    Log.Debug("ended new computation");
                    Log.Debug($"{Processed} processed samples");
                    //we display statistics only once every 10s
                    if (lastTimeDisplayedStatisticsDateTime == null || (DateTime.Now - lastTimeDisplayedStatisticsDateTime.Value).TotalSeconds > 10)
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

                if (maxAllowedSecondsForAllComputation > 0 && ((DateTime.Now - _creationTime).TotalSeconds) > maxAllowedSecondsForAllComputation)
                {
                    Log.Info($"maximum time to process all samples ({maxAllowedSecondsForAllComputation}s) has been used");
                    return;
                }
            }
        }
        /// <summary>
        /// number of processed search spaces
        /// </summary>
        private int Processed => _allAsctualScores.Count;
        private string StatisticsDescription()
        {
            string res = "";
            foreach (var e in SearchSpace.OrderBy(e => e.Key))
            {
                res += "Stats for " + e.Key + ":" + Environment.NewLine + e.Value;
            }
            return res;
        }
    }
}

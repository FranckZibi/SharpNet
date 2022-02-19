﻿using System;
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
        protected readonly DoubleAccumulator _allCost = new();
        protected int _nextSampleId = 0;
        // the last time we have displayed statistics about the search, or null if we have never displayed statistics
        private DateTime? lastTimeDisplayedStatisticsDateTime;
        #endregion

        /// <summary>
        /// the best sample (lowest cost) found so far (or null if no sample has been analyzed)
        /// </summary>
        // ReSharper disable once MemberCanBePrivate.Global
        // ReSharper disable once UnusedAutoPropertyAccessor.Global
        public ISample BestSampleFoundSoFar { get; protected set; }
        /// <summary>
        /// the cost associated with the best sample found sample (or NaN if no sample has been analyzed)
        /// </summary>
        // ReSharper disable once MemberCanBePrivate.Global
        public float CostOfBestSampleFoundSoFar { get; protected set; } = float.NaN;

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


        private string StatisticsDescription()
        {
            string res = "";
            foreach (var e in SearchSpace.OrderBy(e=>e.Key))
            {
                res += "Stats for " + e.Key + ":"+Environment.NewLine+e.Value;
            }
            return res;
        }

        public void Process(Func<ISample, float> objectiveFunction, float maxAllowedSecondsForAllComputation = 0 /* no time limit by default */)
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

        /// <summary>
        /// process next available sample
        /// </summary>
        /// <param name="objectiveFunction"></param>
        /// <param name="maxAllowedSecondsForAllComputation"></param>
        private void ProcessNextSample([NotNull] Func<ISample, float> objectiveFunction, float maxAllowedSecondsForAllComputation)
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
                    float cost = objectiveFunction(sample);
                    if (float.IsNaN(CostOfBestSampleFoundSoFar) || cost < CostOfBestSampleFoundSoFar)
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

                if (maxAllowedSecondsForAllComputation > 0 && ((DateTime.Now - _creationTime).TotalSeconds) > maxAllowedSecondsForAllComputation)
                {
                    Log.Info($"maximum time to process all samples ({maxAllowedSecondsForAllComputation}s) has been used");
                    return;
                }
            }
        }


        protected virtual void RegisterSampleCost(ISample sample, int sampleId, float cost, double elapsedTimeInSeconds)
        {
            _allCost.Add(cost, 1);
            RegisterSampleCost(SearchSpace, sample, cost, elapsedTimeInSeconds);
        }


        protected static void RegisterSampleCost(IDictionary<string, AbstractHyperParameterSearchSpace> searchSpace, ISample sample, float cost, double elapsedTimeInSeconds)
        {
            foreach (var (parameterName, parameterSearchSpace) in searchSpace)
            {
                var parameterValue = sample.Get(parameterName);
                parameterSearchSpace.RegisterCost(parameterValue, cost, elapsedTimeInSeconds);
            }
        }


        /// <summary>
        /// number of processed search spaces
        /// </summary>
        private int Processed => _allCost.Count;

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
    }
}

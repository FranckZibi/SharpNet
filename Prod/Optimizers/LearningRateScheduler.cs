﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public class LearningRateScheduler : ILearningRateScheduler
    {
        #region private fields
        private readonly List<Tuple<double, double>> Values;
        private readonly bool _constantByInterval;
        /// <summary>
        /// if true:
        ///     in each epoch, use the same learning rate
        /// else:
        ///     use a linear learning rate from the start to the end of the epoch
        /// </summary>
        private readonly bool ConstantInEachEpoch;

        #endregion

        public bool ShouldCreateSnapshotForEpoch(int epoch) { return false; }

        #region Constructors
        private LearningRateScheduler(List<Tuple<double, double>> values, bool constantByInterval, bool constantInEachEpoch)
        {
            Debug.Assert(values != null);
            Debug.Assert(values.Count >= 1);
            Values = values;
            _constantByInterval = constantByInterval;
            ConstantInEachEpoch = constantInEachEpoch;
            MaxLearningRate = Values.Select(v => v.Item2).Max();
        }

        public double MaxLearningRate { get; }

        public static LearningRateScheduler Constant(double learningRate)
        {
            var values = new List<Tuple<double, double>> { Tuple.Create(1.0, learningRate) };
            return new LearningRateScheduler(values, false, true);
        }
        public static LearningRateScheduler ConstantByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, true, true);
        }
        public static LearningRateScheduler ConstantByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, epoch3, learningRate3, true);
        }
        public static LearningRateScheduler ConstantByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, int epoch4, double learningRate4)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, epoch3, learningRate3, epoch4, learningRate4, true);
        }
        public static LearningRateScheduler InterpolateByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, false, true);
        }
        public static LearningRateScheduler Linear(double initialLearningRate, int lastEpoch, double lastLearningRate)
        {
            return ByInterval(0, initialLearningRate, lastEpoch, lastLearningRate, false, false);
        }
        public static LearningRateScheduler InterpolateByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, epoch3, learningRate3, false);
        }
        public static LearningRateScheduler InterpolateByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, int epoch4, double learningRate4)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, epoch3, learningRate3, epoch4, learningRate4, false);
        }
        #endregion

        public double LearningRate(int epoch, double percentagePerformedInEpoch)
        {
            return ConstantInEachEpoch
                ? Utils.Interpolate(Values, epoch, _constantByInterval)
                : Utils.Interpolate(Values, epoch-1+ percentagePerformedInEpoch, _constantByInterval);
        }

        public string Serialize()
        {
            return new Serializer()
                .Add(nameof(_constantByInterval), _constantByInterval)
                .Add(nameof(Values) + "Key", Values.Select(x => x.Item1).ToArray())
                .Add(nameof(Values) + "Value", Values.Select(x => x.Item2).ToArray())
                .ToString();
        }
        public static LearningRateScheduler ValueOf(IDictionary<string, object> serialized)
        {
            var constantByInterval = (bool)serialized[nameof(_constantByInterval)];
            var epochs = (double[])serialized[nameof(Values) + "Key"];
            var learningRates = (double[])serialized[nameof(Values) + "Value"];
            var values = epochs.Zip(learningRates, Tuple.Create).ToList();
            var result = new LearningRateScheduler(values, constantByInterval, true);
            return result;
        }

        private static LearningRateScheduler ByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, bool constantByInterval, bool constantInEachEpoch)
        {
            return new LearningRateScheduler(new List<Tuple<double, double>>
                                             {
                                                 new (epoch1, learningRate1),
                                                 new (epoch2, learningRate2)
                                             }, constantByInterval, constantInEachEpoch);
        }
        private static LearningRateScheduler ByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, int epoch4, double learningRate4, bool constantByInterval)
        {
            return new LearningRateScheduler(new List<Tuple<double, double>>
                                             {
                                                 new (epoch1, learningRate1),
                                                 new (epoch2, learningRate2),
                                                 new (epoch3, learningRate3),
                                                 new (epoch4, learningRate4)
                                             }, constantByInterval, true);
        }
        private static LearningRateScheduler ByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, bool constantByInterval)
        {
            return new LearningRateScheduler(new List<Tuple<double, double>>
                                             {
                                                 new (epoch1, learningRate1),
                                                 new (epoch2, learningRate2),
                                                 new (epoch3, learningRate3),
                                             }, constantByInterval, true);
        }
    }
}

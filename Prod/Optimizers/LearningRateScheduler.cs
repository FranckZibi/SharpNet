using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public class LearningRateScheduler : ILearningRateScheduler
    {
        #region private fields
        private readonly List<Tuple<double, double>> _values;
        private readonly bool _constantByInterval;
        #endregion

        public bool ShouldCreateSnapshotForEpoch(int epoch) {return false;}

        #region Constructors
        private LearningRateScheduler(List<Tuple<double, double>> values, bool constantByInterval)
        {
            Debug.Assert(values != null);
            Debug.Assert(values.Count >= 1);
            _values = values;
            _constantByInterval = constantByInterval;
        }
        public static LearningRateScheduler Constant(double learningRate)
        {
            var values = new List<Tuple<double, double>> { Tuple.Create(1.0, learningRate)};
            return new LearningRateScheduler(values, false);
        }
        public static LearningRateScheduler ConstantByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, true);
        }
        public static LearningRateScheduler ConstantByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, epoch3, learningRate3, true);
        }

        /// <summary>
        /// We start with a learning rate of 'initialLearningRate' (epoch = 1)
        /// every 'XEpoch' epochs, we divide the learning rate by 'divideConstant'
        /// </summary>
        /// <param name="initialLearningRate"></param>
        /// <param name="divideConstant"></param>
        /// <param name="XEpoch"></param>
        /// <param name="isConstantByInterval"></param>
        /// <returns></returns>
        public static LearningRateScheduler DivideByConstantEveryXEpoch(double initialLearningRate, int divideConstant, int XEpoch, bool isConstantByInterval)
        {
            var values = new List<Tuple<double, double>> {Tuple.Create(1.0, initialLearningRate)};
            while (values.Count < 100)
            {
                values.Add(Tuple.Create(values.Last().Item1+XEpoch, values.Last().Item2/divideConstant));
            }
            return new LearningRateScheduler(values, isConstantByInterval);
        }
        public static LearningRateScheduler ConstantByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, int epoch4, double learningRate4)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, epoch3, learningRate3, epoch4, learningRate4, true);
        }
        public static LearningRateScheduler InterpolateByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, false);
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

        public double LearningRate(int epoch,
            double percentagePerformedInEpoch)
        {
            return Utils.Interpolate(_values, epoch, _constantByInterval);
        }

        public string Serialize()
        {
            return new Serializer()
                .Add(nameof(_constantByInterval), _constantByInterval)
                .Add(nameof(_values) + "Key", _values.Select(x => x.Item1).ToArray())
                .Add(nameof(_values) + "Value", _values.Select(x => x.Item2).ToArray())
                .ToString();
        }
        public static LearningRateScheduler ValueOf(IDictionary<string, object> serialized)
        {
            var constantByInterval = (bool) serialized[nameof(_constantByInterval)];
            var epochs = (double[]) serialized[nameof(_values) + "Key"];
            var learningRates = (double[]) serialized[nameof(_values) + "Value"];
            var values  = epochs.Zip(learningRates, Tuple.Create).ToList();
            var result = new LearningRateScheduler(values, constantByInterval);
            return result;
        }

        private static LearningRateScheduler ByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, bool constantByInterval)
        {
            return new LearningRateScheduler(new List<Tuple<double, double>>
            {
                new Tuple<double, double>(epoch1, learningRate1),
                new Tuple<double, double>(epoch2, learningRate2)
            }, constantByInterval);
        }
        private static LearningRateScheduler ByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, int epoch4, double learningRate4, bool constantByInterval)
        {
            return new LearningRateScheduler(new List<Tuple<double, double>>
            {
                new Tuple<double, double>(epoch1, learningRate1),
                new Tuple<double, double>(epoch2, learningRate2),
                new Tuple<double, double>(epoch3, learningRate3),
                new Tuple<double, double>(epoch4, learningRate4)
            }, constantByInterval);
        }
        private static LearningRateScheduler ByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, bool constantByInterval)
        {
            return new LearningRateScheduler(new List<Tuple<double, double>>
            {
                new Tuple<double, double>(epoch1, learningRate1),
                new Tuple<double, double>(epoch2, learningRate2),
                new Tuple<double, double>(epoch3, learningRate3),
            }, constantByInterval);
        }
    }
}

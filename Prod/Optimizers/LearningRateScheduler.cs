using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public class LearningRateScheduler
    {
        #region private fields
        private readonly List<KeyValuePair<int,double>> _values;
        private readonly bool _constantByInterval;
        #endregion



        #region Constructors
        private LearningRateScheduler(List<KeyValuePair<int, double>> values, bool constantByInterval)
        {
            Debug.Assert(values != null);
            Debug.Assert(values.Count >= 1);
            _values = values;
            _constantByInterval = constantByInterval;
        }
        public static LearningRateScheduler Constant(double learningRate)
        {
            var values = new List<KeyValuePair<int, double>> {new KeyValuePair<int, double>(1, learningRate)};
            return new LearningRateScheduler(values, false);
        }
        public static LearningRateScheduler ConstantByInterval(List<KeyValuePair<int, double>> values)
        {
            return new LearningRateScheduler(values, true);
        }
        public static LearningRateScheduler ConstantByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, true);
        }
        public static LearningRateScheduler ConstantByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, epoch3, learningRate3, true);
        }

        //we start with a learning rate of 'initialLearningRate' (epoxh = 1)
        //every 'XEpoch' epochs, we divide the learning rate by 'divideConstant'
        public static LearningRateScheduler DivideByConstantEveryXEpoch(double initialLearningRate, int divideConstant, int XEpoch, bool isConstantByInterval)
        {
            var values = new List<KeyValuePair<int, double>> {new KeyValuePair<int, double>(1, initialLearningRate)};
            while (values.Count < 100)
            {
                values.Add(new KeyValuePair<int, double>(values.Last().Key+XEpoch, values.Last().Value/divideConstant));
            }
            return new LearningRateScheduler(values, isConstantByInterval);
        }
        public static LearningRateScheduler ConstantByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, int epoch4, double learningRate4)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, epoch3, learningRate3, epoch4, learningRate4, true);
        }
        public static LearningRateScheduler ConstantByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, int epoch4, double learningRate4, int epoch5, double learningRate5)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, epoch3, learningRate3, epoch4, learningRate4, epoch5, learningRate5, true);
        }
        public static LearningRateScheduler ConstantByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, int epoch4, double learningRate4, int epoch5, double learningRate5, int epoch6, double learningRate6)
        {
            return ByInterval(epoch1, learningRate1, epoch2, learningRate2, epoch3, learningRate3, epoch4, learningRate4, epoch5, learningRate5, epoch6, learningRate6, true);
        }
        public static LearningRateScheduler InterpolateByInterval(List<KeyValuePair<int, double>> values)
        {
            return new LearningRateScheduler(values, false);
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



        public double LearningRate(int epoch)
        {
            if (_values.Count == 1)
            {
                return _values[0].Value;
            }
            for (int i = 0; i < _values.Count; ++i)
            {
                if (epoch > _values[i].Key)
                {
                    continue;
                }
                if (_values[i].Key == epoch || i == 0)
                {
                    return _values[i].Value;
                }
                Debug.Assert(epoch < _values[i].Key);
                Debug.Assert(epoch > _values[i-1].Key);
                if (_constantByInterval)
                {
                    return _values[i - 1].Value;
                }

                double dEpoch = ((double)epoch- _values[i-1].Key)/(_values[i].Key-_values[i - 1].Key);
                double deltaLearningRate = (_values[i].Value - _values[i - 1].Value);
                return _values[i - 1].Value + dEpoch * deltaLearningRate;
            }
            return _values.Last().Value;
        }

        public string Serialize()
        {
            return new Serializer()
                .Add(nameof(_constantByInterval), _constantByInterval)
                .Add(nameof(_values) + "Key", _values.Select(x => x.Key).ToArray())
                .Add(nameof(_values) + "Value", _values.Select(x => x.Value).ToArray())
                .ToString();
        }
        public static LearningRateScheduler ValueOf(IDictionary<string, object> serialized)
        {
            var constantByInterval = (bool) serialized[nameof(_constantByInterval)];
            var epochs = (int[]) serialized[nameof(_values) + "Key"];
            var learningRates = (double[]) serialized[nameof(_values) + "Value"];
            var values  = epochs.Zip(learningRates, (x, y) => new KeyValuePair<int, double>(x, y)).ToList();
            var result = new LearningRateScheduler(values, constantByInterval);
            return result;
        }

        private static LearningRateScheduler ByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, bool constantByInterval)
        {
            return new LearningRateScheduler(new List<KeyValuePair<int, double>>
            {
                new KeyValuePair<int, double>(epoch1, learningRate1),
                new KeyValuePair<int, double>(epoch2, learningRate2)
            }, constantByInterval);
        }
        private static LearningRateScheduler ByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, int epoch4, double learningRate4, int epoch5, double learningRate5, int epoch6, double learningRate6, bool constantByInterval)
        {
            return new LearningRateScheduler(new List<KeyValuePair<int, double>>
            {
                new KeyValuePair<int, double>(epoch1, learningRate1),
                new KeyValuePair<int, double>(epoch2, learningRate2),
                new KeyValuePair<int, double>(epoch3, learningRate3),
                new KeyValuePair<int, double>(epoch4, learningRate4),
                new KeyValuePair<int, double>(epoch5, learningRate5),
                new KeyValuePair<int, double>(epoch6, learningRate6)
            }, constantByInterval);
        }
        private static LearningRateScheduler ByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, int epoch4, double learningRate4, int epoch5, double learningRate5, bool constantByInterval)
        {
            return new LearningRateScheduler(new List<KeyValuePair<int, double>>
            {
                new KeyValuePair<int, double>(epoch1, learningRate1),
                new KeyValuePair<int, double>(epoch2, learningRate2),
                new KeyValuePair<int, double>(epoch3, learningRate3),
                new KeyValuePair<int, double>(epoch4, learningRate4),
                new KeyValuePair<int, double>(epoch5, learningRate5)
            }, constantByInterval);
        }
        private static LearningRateScheduler ByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, int epoch4, double learningRate4, bool constantByInterval)
        {
            return new LearningRateScheduler(new List<KeyValuePair<int, double>>
            {
                new KeyValuePair<int, double>(epoch1, learningRate1),
                new KeyValuePair<int, double>(epoch2, learningRate2),
                new KeyValuePair<int, double>(epoch3, learningRate3),
                new KeyValuePair<int, double>(epoch4, learningRate4)
            }, constantByInterval);
        }
        private static LearningRateScheduler ByInterval(int epoch1, double learningRate1, int epoch2, double learningRate2, int epoch3, double learningRate3, bool constantByInterval)
        {
            return new LearningRateScheduler(new List<KeyValuePair<int, double>>
            {
                new KeyValuePair<int, double>(epoch1, learningRate1),
                new KeyValuePair<int, double>(epoch2, learningRate2),
                new KeyValuePair<int, double>(epoch3, learningRate3),
            }, constantByInterval);
        }


    }
}
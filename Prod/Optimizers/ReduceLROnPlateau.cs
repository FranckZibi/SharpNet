using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public class ReduceLROnPlateau
    {
        #region fields & properties
        public double FactorForReduceLrOnPlateau { get; }
        private readonly int _patienceForReduceLrOnPlateau;
        private readonly int _cooldownForReduceLrOnPlateau;
        #endregion


        public ReduceLROnPlateau(double factorForReduceLrOnPlateau, int patienceForReduceLrOnPlateau, int cooldownForReduceLrOnPlateau)
        {
            FactorForReduceLrOnPlateau = factorForReduceLrOnPlateau;
            _patienceForReduceLrOnPlateau = patienceForReduceLrOnPlateau;
            _cooldownForReduceLrOnPlateau = cooldownForReduceLrOnPlateau;
        }


        public static int NbConsecutiveEpochsWithoutProgress(List<EpochData> epochData)
        {
            if (epochData.Count <= 1)
            {
                return 0;
            }
            var minLoss = epochData.Select(x => x.TrainingLoss).Min();
            for (int i = epochData.Count - 1; i >= 0; --i)
            {
                if (epochData[i].TrainingLoss <= minLoss+1e-8)
                {
                    return epochData.Count - 1 - i;
                }
            }
            return 0;
        }

        public static int NbConsecutiveEpochsWithSameMultiplicativeFactor(List<EpochData> epochData)
        {
            for (int i = epochData.Count - 2; i >= 0; --i)
            {
                if (Math.Abs(epochData[i].LearningRateMultiplicativeFactorFromReduceLrOnPlateau- epochData[i+1].LearningRateMultiplicativeFactorFromReduceLrOnPlateau)>1e-8)
                {
                    return epochData.Count - 1 - i;
                }
            }
            return epochData.Count;
        }

        public bool ShouldReduceLrOnPlateau(List<EpochData> previousEpochData)
        {
            var nbConsecutiveEpochWithoutProgress = NbConsecutiveEpochsWithoutProgress(previousEpochData);
            if (_patienceForReduceLrOnPlateau <= 0 || nbConsecutiveEpochWithoutProgress <= _patienceForReduceLrOnPlateau || FactorForReduceLrOnPlateau <= 0.0)
            {
                return false;
            }

            //we are about to reduce the learning rate. We must make sure that it has not been reduced recently (cooldown factor)
            var nbConsecutiveEpochsWithoutReducingLrOnPlateau = NbConsecutiveEpochsWithSameMultiplicativeFactor(previousEpochData);
            //TODO Add tests
            if (_cooldownForReduceLrOnPlateau >= 1 && nbConsecutiveEpochsWithoutReducingLrOnPlateau <= _cooldownForReduceLrOnPlateau)
            {
                return false;
            }

            return true;
        }
        public string Serialize()
        {
            return new Serializer()
                .Add(nameof(FactorForReduceLrOnPlateau), FactorForReduceLrOnPlateau)
                .Add(nameof(_patienceForReduceLrOnPlateau), _patienceForReduceLrOnPlateau)
                .Add(nameof(_cooldownForReduceLrOnPlateau), _cooldownForReduceLrOnPlateau)
                .ToString();
        }
        public static ReduceLROnPlateau ValueOf(IDictionary<string, object> serialized)
        {
            return new ReduceLROnPlateau(
                (double) serialized[nameof(FactorForReduceLrOnPlateau)],
                (int) serialized[nameof(_patienceForReduceLrOnPlateau)],
                (int) serialized[nameof(_cooldownForReduceLrOnPlateau)]
            );
        }

    }
}
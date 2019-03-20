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
        private const double MinDeltaForReduceLrOnPlateau = 0.0001;
        #endregion


        public ReduceLROnPlateau(double factorForReduceLrOnPlateau, int patienceForReduceLrOnPlateau, int cooldownForReduceLrOnPlateau)
        {
            FactorForReduceLrOnPlateau = factorForReduceLrOnPlateau;
            _patienceForReduceLrOnPlateau = patienceForReduceLrOnPlateau;
            _cooldownForReduceLrOnPlateau = cooldownForReduceLrOnPlateau;
        }


        public static int NbConsecutiveEpochsWithoutProgress(List<EpochData> epochData)
        {
            if (epochData.Count == 0)
            {
                return 0;
            }
            var maxAccuracy = epochData.Select(x => x.TrainingAccuracy).Max();
            for (int i = epochData.Count - 1; i >= 0; --i)
            {
                if (epochData[i].TrainingAccuracy >= (maxAccuracy - MinDeltaForReduceLrOnPlateau))
                {
                    return epochData.Count - 1 - i;
                }
            }
            return epochData.Count;
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
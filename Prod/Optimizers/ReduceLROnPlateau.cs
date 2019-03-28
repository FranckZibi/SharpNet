using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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


        /// <summary>
        /// return the distance (# epochs) between the last epoch and the best epoch found so far (in term of training loss)
        /// </summary>
        /// <param name="epochData"></param>
        /// <param name="maxNbConsecutiveEpochsToReport">stops as soon as the returned result is >= 'maxNbConsecutiveEpochsToReport'</param>
        /// <returns>distance (in number of epochs) between the last epoch and the best epoch found so far (in term of training loss)
        /// it will return 0 if the best epoch was the last processed, 1 if the best epoch was just before the last epoch, etc...</returns>
        public static int NbConsecutiveEpochsWithoutProgress(List<EpochData> epochData, int maxNbConsecutiveEpochsToReport = int.MaxValue)
        {
            Debug.Assert(maxNbConsecutiveEpochsToReport>=1);
            if (epochData.Count <= 1)
            {
                return 0;
            }
            var minLoss = epochData.Select(x => x.TrainingLoss).Min();
            int nbConsecutiveEpochsWithoutProgress = 0;
            for (int i = epochData.Count - 1; i >= 0; --i)
            {
                //if progress observed
                if (epochData[i].TrainingLoss <= minLoss+1e-8)
                {
                    break;
                }
                ++nbConsecutiveEpochsWithoutProgress;
                if (nbConsecutiveEpochsWithoutProgress >= maxNbConsecutiveEpochsToReport)
                {
                    break;
                }
            }
            return nbConsecutiveEpochsWithoutProgress;
        }

        public static int NbConsecutiveEpochsWithSameMultiplicativeFactor(List<EpochData> epochData)
        {
            for (int i = epochData.Count - 2; i >= 0; --i)
            {
                if (Math.Abs(epochData[i].LearningRateMultiplicativeFactorFromReduceLrOnPlateau- epochData[i+1].LearningRateMultiplicativeFactorFromReduceLrOnPlateau)>1e-30)
                {
                    return epochData.Count - 1 - i;
                }
            }
            return epochData.Count;
        }

        public bool ShouldReduceLrOnPlateau(List<EpochData> previousEpochData)
        {
            var nbConsecutiveEpochWithoutProgress = NbConsecutiveEpochsWithoutProgress(previousEpochData, _patienceForReduceLrOnPlateau+1);
            if (_patienceForReduceLrOnPlateau <= 0 || nbConsecutiveEpochWithoutProgress <= _patienceForReduceLrOnPlateau || FactorForReduceLrOnPlateau <= 0.0)
            {
                return false;
            }

            //we are about to reduce the learning rate. We must make sure that it has not been reduced recently (cooldown factor)
            if (_cooldownForReduceLrOnPlateau >= 1 && NbConsecutiveEpochsWithSameMultiplicativeFactor(previousEpochData) <= _cooldownForReduceLrOnPlateau)
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
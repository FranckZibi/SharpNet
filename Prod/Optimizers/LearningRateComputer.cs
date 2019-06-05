using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public class LearningRateComputer : ILearningRateComputer
    {
        private readonly ILearningRateScheduler _lrScheduler;
        private readonly ReduceLROnPlateau _reduceLrOnPlateauIfAny;
        private readonly double _minimumLearningRate;

        public LearningRateComputer(ILearningRateScheduler lrScheduler, ReduceLROnPlateau reduceLrOnPlateauIfAny, double minimumLearningRate)
        {
            _lrScheduler = lrScheduler;
            _reduceLrOnPlateauIfAny = reduceLrOnPlateauIfAny;
            _minimumLearningRate = minimumLearningRate;
        }
        public double LearningRate(int epoch, int blockIdInEpoch, int nbBlocksInEpoch, double learningRateMultiplicativeFactorFromReduceLrOnPlateau)
        {
            Debug.Assert(epoch>= 1);
            Debug.Assert(blockIdInEpoch >= 0);
            Debug.Assert(nbBlocksInEpoch >= 1);
            var learningRateFromScheduler = _lrScheduler.LearningRate(epoch, blockIdInEpoch, nbBlocksInEpoch);
            var learningRateWithPlateauReduction = learningRateFromScheduler * learningRateMultiplicativeFactorFromReduceLrOnPlateau;
            learningRateWithPlateauReduction = Math.Max(learningRateWithPlateauReduction, _minimumLearningRate);
            return learningRateWithPlateauReduction;
        }
        public bool ShouldReduceLrOnPlateau(List<EpochData> previousEpochsData)
        {
            return _reduceLrOnPlateauIfAny != null && _reduceLrOnPlateauIfAny.ShouldReduceLrOnPlateau(previousEpochsData);
        }
        public double MultiplicativeFactorFromReduceLrOnPlateau(List<EpochData> previousEpochsData)
        {
            var learningRateMultiplicativeFactorFromReduceLrOnPlateau = previousEpochsData.Count >= 1
                ? previousEpochsData.Last().LearningRateMultiplicativeFactorFromReduceLrOnPlateau
                : 1.0;
            if (ShouldReduceLrOnPlateau(previousEpochsData))
            {
                learningRateMultiplicativeFactorFromReduceLrOnPlateau *= _reduceLrOnPlateauIfAny.FactorForReduceLrOnPlateau;
            }
            return learningRateMultiplicativeFactorFromReduceLrOnPlateau;
        }

        public bool ShouldCreateSnapshotForEpoch(int epoch)
        {
            return _lrScheduler.ShouldCreateSnapshotForEpoch(epoch);
        }
    }
}
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    /// <summary>
    /// see https://www.jeremyjordan.me/nn-learning-rate/
    /// </summary>
    public class LearningRateFinder : ILearningRateComputer
    {
        #region private fields
        private readonly double _momentum;
        private readonly double _minLearningRate;
        /// <summary>
        ///between each block, we'll multiply the learning rate by 'multiplicativeCoeff'
        /// </summary>
        private readonly double _multiplicativeCoeff;
        private readonly List<double> _loss = new List<double>();
        private readonly List<double> _avgLosses = new List<double>();
        private readonly List<double> _smoothedLosses = new List<double>();
        private readonly int _nbBlocksPerEpoch;

        #endregion

        public LearningRateFinder(int miniBatchSize, int entireBatchSize, double momentum = 0.95, double minLearningRate = 1e-7, double maxLearningRate = 10.0)
        {
            _momentum = momentum;
            _minLearningRate = minLearningRate;
            _nbBlocksPerEpoch = (entireBatchSize + miniBatchSize - 1) / miniBatchSize;
            _multiplicativeCoeff = Math.Pow(maxLearningRate / minLearningRate, (1.0 / (_nbBlocksPerEpoch - 1)));
        }

        public double LearningRate(int epoch, int blockIdInEpoch, int nbBlocksInEpoch, double learningRateMultiplicativeFactorFromReduceLrOnPlateau)
        {
            Debug.Assert(epoch >= 1);
            Debug.Assert(blockIdInEpoch >= 0);
            Debug.Assert(nbBlocksInEpoch >= 1);
            return _minLearningRate * Math.Pow(_multiplicativeCoeff, blockIdInEpoch);
        }

        /// <summary>
        /// at the end of each block id, this method is called with the loss computed in the last block
        /// </summary>
        /// <param name="loss">loss associated with the last computed block</param>
        /// <returns>true if we should stop the computation (no need to look to other blocks)
        /// false if we should continue</returns>
        public bool AddLossForLastBlockId(double loss)
        {
            if (double.IsNaN(loss) ||double.IsInfinity(loss))
            {
                return true;
            }
            _loss.Add(loss);
            var prevAvgLoss = (_avgLosses.Count == 0 ? 0 : _avgLosses.Last());
            var avgLoss = _momentum * prevAvgLoss + (1 - _momentum) * loss;
            _avgLosses.Add(avgLoss);
            var smoothedLoss = avgLoss / (1.0-Math.Pow(_momentum, 1+_smoothedLosses.Count));
            _smoothedLosses.Add(smoothedLoss);
            return false;
        }
        public double BestLearningRate()
        {
            if (_multiplicativeCoeff <= 1.0)
            {
                return 0.0;
            }
            var nbBlocksBetweenAFactor10InLearningRate = (int)Math.Ceiling(Math.Log(10) / Math.Log(_multiplicativeCoeff));
            double maxDecreaseInLoss = double.MinValue;
            double bestLearningRate = double.NaN;
            for (int blockId = 10 + nbBlocksBetweenAFactor10InLearningRate;blockId < _smoothedLosses.Count - 5;++blockId)
            {
                var learningRateForBatchBlock = LearningRate(1, blockId, _nbBlocksPerEpoch, 1.0);
                //we measure the reduction of the loss between:
                //  the loss observed with the learning rate used at 'blockId' (=learningRateForBatchBlock)
                //  the loss observed with a learning rate 10x lower (=learningRateForBatchBlock/10) as observed at 'blockId- nbBlocksBetweenAFactor10InLearningRate'
                var observedDecreaseInLoss = _smoothedLosses[blockId- nbBlocksBetweenAFactor10InLearningRate] - _smoothedLosses[blockId];
                if (observedDecreaseInLoss > maxDecreaseInLoss)
                {
                    maxDecreaseInLoss = observedDecreaseInLoss;
                    bestLearningRate = learningRateForBatchBlock;
                }
            }
            return bestLearningRate;
        }
        public string AsCsv()
        {
            var sb = new StringBuilder();
            sb.Append("Sep=;" + Environment.NewLine);
            sb.Append("LearningRate;LearningRateLog;Loss" + Environment.NewLine);
            for (int blockId = 0; blockId < _loss.Count; ++blockId)
            {
                var learningRateForBatchBlock = LearningRate(1, blockId, _nbBlocksPerEpoch, 1.0);
                sb.Append(learningRateForBatchBlock.ToString(CultureInfo.InvariantCulture) + ";"+Math.Log10(learningRateForBatchBlock).ToString(CultureInfo.InvariantCulture) + ";" + _smoothedLosses[blockId] + Environment.NewLine);
            }
            return sb.ToString();
        }
        public bool ShouldReduceLrOnPlateau(List<EpochData> previousEpochsData) {return false;}
        public double MultiplicativeFactorFromReduceLrOnPlateau(List<EpochData> previousEpochsData) {return 1.0;}
        public bool ShouldCreateSnapshotForEpoch(int epoch)
        {
            return false;
        }
    }
}

using System.Diagnostics;

namespace SharpNet.Optimizers
{
    public class OneCycleLearningRateScheduler : ILearningRateScheduler
    {
        #region private fields
        private readonly double _maxLearningRate;
        private readonly int _div;
        private readonly double _pct;
        private readonly int _nbEpochs;
        #endregion

        private double MinLearningRate() {return _maxLearningRate/ _div;}
        private double MaxLearningRate() {return _maxLearningRate ;}
        private double LearningRateForLastEpoch() {return MinLearningRate()/100;}
        private double EpochForMaxLearningRate() {return 1.0+ _nbEpochs * (1.0 - _pct) / 2;}
        private double EpochForMinLearningRate() {return 1.0 + _nbEpochs * (1.0 - _pct);}

        /// <summary>
        /// see https://github.com/sgugger/Deep-Learning/blob/master/Cyclical%20LR%20and%20momentums.ipynb
        /// </summary>
        /// <param name="maxLearningRate"></param>
        /// <param name="div"> the amount to divide the passed learning rate to get the minimum learning rate. 
        /// E.g.: pick 1/10th of the maximum learning rate for the minimum learning rate.</param>
        /// <param name="pct">the part of the cycle (in percent) that will be devoted to the LR annealing after the triangular cycle. 
        /// E.g.: dedicate 13.68% of the cycle to the annealing at the end (that’s 13 epochs over 95).</param>
        /// <param name="nbEpochs">total number of epochs</param>
        public OneCycleLearningRateScheduler(double maxLearningRate, int div, double pct, int nbEpochs)
        {
            Debug.Assert(div >= 1);
            _maxLearningRate = maxLearningRate;
            _div = div;
            _pct = pct;
            _nbEpochs = nbEpochs;
        }

        public bool ShouldCreateSnapshotForEpoch(int epoch) { return epoch == _nbEpochs; }

        public double LearningRate(int epoch, int blockIdInEpoch, int nbBlocksInEpoch)
        {
            double currentEpoch = epoch + ((double) blockIdInEpoch) / nbBlocksInEpoch;
            if (currentEpoch <= EpochForMaxLearningRate())
            {
                //first part of the cycle: increasing the learning rate from 'MinLearningRate()' to 'MaxLearningRate()'
                return Utils.Interpolate(1.0, MinLearningRate(), EpochForMaxLearningRate(), MaxLearningRate(), currentEpoch);
            }
            if (currentEpoch <= EpochForMinLearningRate())
            {
                //second part of the cycle: decreasing the learning rate from 'MaxLearningRate()' to 'MinLearningRate()'
                return Utils.Interpolate(EpochForMaxLearningRate(), MaxLearningRate(), EpochForMinLearningRate(), MinLearningRate(), currentEpoch);
            }
            //last part: annihilating the learning rate from 'MinLearningRate()' to 'LearningRateForLastEpoch()'
            return Utils.Interpolate(EpochForMinLearningRate(), MinLearningRate(), _nbEpochs, LearningRateForLastEpoch(), currentEpoch);
        }
    }
}
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
        private double LearningRateForLastEpoch() {return MinLearningRate()/1000;}
        private double EpochForMaxLearningRate(int nbEpochs) {return 1.0+ nbEpochs * (1.0 - _pct) / 2;}
        private double EpochForMinLearningRate(int nbEpochs) {return 1.0 + nbEpochs * (1.0 - _pct);}

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
            _maxLearningRate = maxLearningRate;
            _div = div;
            _pct = pct;
            _nbEpochs = nbEpochs;
        }
        public double LearningRate(int epoch, int blockIdInEpoch, int nbBlocksInEpoch)
        {
            double currentEpoch = epoch + ((double) blockIdInEpoch) / nbBlocksInEpoch;
            if (currentEpoch <= EpochForMaxLearningRate(_nbEpochs))
            {
                return Utils.Interpolate(1.0, MinLearningRate(), EpochForMaxLearningRate(_nbEpochs), MaxLearningRate(), currentEpoch);
            }
            if (currentEpoch <= EpochForMinLearningRate(_nbEpochs))
            {
                return Utils.Interpolate(EpochForMaxLearningRate(_nbEpochs), MaxLearningRate(), EpochForMinLearningRate(_nbEpochs), MinLearningRate(), currentEpoch);
            }
            return Utils.Interpolate(EpochForMinLearningRate(_nbEpochs), MinLearningRate(), _nbEpochs, LearningRateForLastEpoch(), currentEpoch);
        }
    }
}
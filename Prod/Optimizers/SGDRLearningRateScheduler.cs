﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpNet.Optimizers
{
    /// <summary>
    /// Implementation of SGDR: stochastic gradient descent with warm restarts
    /// (see https://arxiv.org/pdf/1608.03983.pdf)
    /// </summary>
    public class SGDRLearningRateScheduler : ILearningRateScheduler
    {
        #region private fields
        private readonly double _maxLearningRate;
        private readonly List<Tuple<double, double>> _values = new List<Tuple<double, double>>();
        #endregion

        /// <summary>
        /// see https://arxiv.org/pdf/1608.03983.pdf
        /// </summary>
        /// <param name="maxLearningRate"></param>
        /// <param name="nbEpochsInFirstRun">Number of epochs in the first warm started run</param>
        /// <param name="nbEpochInNextRunMultiplier">factor to multiply the number of epochs in the previous run
        /// <param name="nbEpochs">total number of epochs</param>
        /// to get the number of epochs in the next run</param>
        public SGDRLearningRateScheduler(double maxLearningRate, int nbEpochsInFirstRun, int nbEpochInNextRunMultiplier, int nbEpochs)
        {
            Debug.Assert(nbEpochsInFirstRun>=1);
            Debug.Assert(nbEpochInNextRunMultiplier >= 1);
            int firstEpochInCurrentRun = 1;
            int nbEpochsInCurrentRun = nbEpochsInFirstRun;
            for (;;)
            {
                _values.Add(Tuple.Create((double)firstEpochInCurrentRun,0.0));
                var lastEpochInCurrentRun = firstEpochInCurrentRun+ nbEpochsInCurrentRun-1;
                var firstEpochInNextRun = firstEpochInCurrentRun+ nbEpochsInCurrentRun;
                var nbEpochsInNextRun = nbEpochsInCurrentRun*nbEpochInNextRunMultiplier;
                var lastEpochInNextRun = firstEpochInNextRun + nbEpochsInNextRun-1;
                //if it is not possible to finish entirely the next run
                if (lastEpochInNextRun > nbEpochs)
                {
                    //we'll increase the length (in epoch count) of the current run to make it finish at exactly the last epoch
                    //(no need to do a new run after it because it will have a smaller size)
                    lastEpochInCurrentRun = nbEpochs;
                }
                _values.Add(Tuple.Create(lastEpochInCurrentRun+1-1e-6, 1.0));
                if (lastEpochInCurrentRun >= nbEpochs)
                {
                    break;
                }
                firstEpochInCurrentRun = firstEpochInNextRun;
                nbEpochsInCurrentRun = nbEpochsInNextRun;
            }
            _maxLearningRate = maxLearningRate;
        }
        public double LearningRate(int epoch, int blockIdInEpoch, int nbBlocksInEpoch)
        {
            var currentEpoch = epoch + ((double)blockIdInEpoch) / nbBlocksInEpoch;
            var multiplier = Utils.Interpolate(_values, currentEpoch);
            var minLearningRate = 1e-6;
            var learningRate = minLearningRate + 0.5* (_maxLearningRate -minLearningRate) * (1.0 + Math.Cos(multiplier * Math.PI));
            return learningRate;
        }
    }
}
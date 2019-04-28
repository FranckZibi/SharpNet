using System;
using SharpNet.Optimizers;
using SharpNet.Pictures;

namespace SharpNet
{
    public interface IMetaParameters
    {
        string ExtraDescription { get; set; }
    }

    public class ResNetMetaParameters : IMetaParameters
    {
        public bool UseGPU { get; set; } = true;
        public bool UseDoublePrecision { get; set; } = false;
        public bool DivideBy10OnPlateau { get; set; } = true; // 'true' : validated on 19-apr-2019: +20 bps
        public bool UseAdam { get; set; } = false; // 'false' : validated on 19-apr-2019: +70 bps

        public bool UseNesterov { get; set; } = false;

        //for one cycle policy: by how much we have to divide the max learning rate to reach the min learning rate
        public int OneCycleDividerForMinLearningRate { get; set; } = 10;
        public double OneCyclePercentInAnnealing { get; set; } = 0.2;
        public bool OneCycleLearningRate { get; set; } = false;
        public double lambdaL2Regularization { get; set; } = 1e-4;
        public bool LinearLearningRate { get; set; } = false;
        public int NumEpochs { get; set; } = 160; //64k iterations
        public int BatchSize { get; set; } = 128;
        public double InitialLearningRate { get; set; } = 0.1;

        public int CutoutPatchlength { get; set; } = 16; // '16' : validated on 17-apr-2019: +70 bps

        //validated on 18-apr-2019: +300 bps (for both using WidthShiftRange & HeightShiftRange)
        public double WidthShiftRange { get; set; } = 0.1;
        public double HeightShiftRange { get; set; } = 0.1;
        public bool HorizontalFlip { get; set; } = true; // 'true' : validated on 18-apr-2019: +70 bps
        public bool VerticalFlip { get; set; } = false;

        public ImageDataGenerator.FillModeEnum FillMode { get; set; } =
            ImageDataGenerator.FillModeEnum.Reflect; //validated on 18-apr-2019: +50 bps

        public string ExtraDescription { get; set; } = "";


        public ILearningRateScheduler Cifar10LearningRateScheduler()
        {
            if (OneCycleLearningRate)
            {
                return new OneCycleLearningRateScheduler(InitialLearningRate, OneCycleDividerForMinLearningRate,
                    OneCyclePercentInAnnealing, NumEpochs);
            }

            if (LinearLearningRate)
            {
                return LearningRateScheduler.InterpolateByInterval(1, InitialLearningRate, 80, InitialLearningRate / 10,
                    120, InitialLearningRate / 100);
            }

            return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, 80, InitialLearningRate / 10, 120,
                InitialLearningRate / 100);

        }

        public ImageDataGenerator ResNetImageDataGenerator()
        {
            return new ImageDataGenerator(WidthShiftRange, HeightShiftRange, HorizontalFlip, VerticalFlip, FillMode,
                0.0, CutoutPatchlength);
        }

        public ReduceLROnPlateau Cifar10ReduceLROnPlateau()
        {
            if (OneCycleLearningRate)
            {
                return null;
            }

            var factorForReduceLrOnPlateau = DivideBy10OnPlateau ? 0.1 : Math.Sqrt(0.1);
            return new ReduceLROnPlateau(factorForReduceLrOnPlateau, 5, 5);
        }

        /*publicILearningRateScheduler UpdatedCifar10LearningRateScheduler()
        {
            var initialLearningRate = 0.1;
            return LearningRateScheduler.ConstantByInterval(1, initialLearningRate/10.0, 2, initialLearningRate, 80, initialLearningRate / 10, 120, initialLearningRate / 100, 160, initialLearningRate / 1000, 180, initialLearningRate / 2000);
        }*/
        public ILearningRateScheduler ResNet110LearningRateScheduler()
        {
            if (OneCycleLearningRate)
            {
                return new OneCycleLearningRateScheduler(InitialLearningRate, OneCycleDividerForMinLearningRate,
                    OneCyclePercentInAnnealing, NumEpochs);
            }

            return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate / 10, 2, InitialLearningRate, 80,
                InitialLearningRate / 10, 120, InitialLearningRate / 100);
        }
    }
}
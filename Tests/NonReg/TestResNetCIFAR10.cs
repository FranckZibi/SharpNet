using System.Diagnostics;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNet.Pictures;

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestResNetCIFAR10
    {
        [Test, Ignore]
        public void Test()
        {
            //int num_res_blocks = 3;
            int num_res_blocks = 3;
            int batchSize = 128;
            const int numEpochs = 200;

            //var learningRate = 0.1/batchSize;
            //var initialLearningRate = 1e-3;
            var initialLearningRate = 0.1;
            var learningRate = LearningRateScheduler.ConstantByInterval(1, initialLearningRate, 80, initialLearningRate/10, 120, initialLearningRate/100);
            //var learningRate = LearningRateScheduler.DivideByConstantEveryXEpoch(initialLearningRate, 10, 30, true);
            //var learningRate = LearningRateScheduler.Constant(0.001);

            var logFileName = Utils.ConcatenatePathWithFileName(@"c:\temp\ML\", "TestResNetCIFAR10" + "_" + Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            var logger = new Logger(logFileName, true);


            TestCIFAR10.Load(out CpuTensor<byte> xTrainingSet, out var yTrainingSet, out var xTestSet, out var yTestSet);

            TestCIFAR10.ToWorkingSet(xTrainingSet, yTrainingSet, out CpuTensor<float> X_train, out CpuTensor<float> Y_train);
            TestCIFAR10.ToWorkingSet(xTestSet, yTestSet, out CpuTensor<float> X_test, out CpuTensor<float> Y_test);

            //We remove the mean 
            var mean = X_train.Mean(); X_train.Add(-mean); X_test.Add(-mean);
            
            /*
            //bool useGpu = false;
            var nbRows = 1000;
            X_train = (CpuTensor<float>)X_train.ExtractSubTensor(0, nbRows);
            Y_train = (CpuTensor<float>)Y_train.ExtractSubTensor(0, nbRows);
            X_test = (CpuTensor<float>)X_test.ExtractSubTensor(0, System.Math.Min(10000,nbRows));
            Y_test = (CpuTensor<float>)Y_test.ExtractSubTensor(0, System.Math.Min(10000, nbRows));
            */

            var network = GetResNet(true, num_res_blocks, X_train.Shape, Y_train.Shape, logger);
            network.Fit(X_train, Y_train, learningRate, numEpochs, batchSize, X_test, Y_test);
        }

        //!D TODO this currently returns a plain network, not a ResNet
        private static Network GetResNet(bool useGPU, int num_res_blocks, int[] xShape, int[] yShape, Logger logger)
        {
            //return Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\Network_18736_102.txt");
            var activationFunction = cudnnActivationMode_t.CUDNN_ACTIVATION_RELU;

            var networkConfig = new NetworkConfig(useGPU) { Logger = logger, UseDoublePrecision = false, LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy };
            var imageDataGenerator = new ImageDataGenerator(0.1, 0.1, true, false, ImageDataGenerator.FillModeEnum.Nearest, 0.0);
            var network = new Network(networkConfig
                //.WithAdam(),
                .WithSGD(0.9, 0.0001, true),
                //.WithSGD(0.9, 0, true),
                imageDataGenerator
            );
            network.AddInput(xShape[1], xShape[2], xShape[3]);

            //!D double lambdaL2Regularization = 0.0;
            double lambdaL2Regularization = 1e-4;
            network.AddConvolution_BatchNorm_Activation(16, 3, 1, 1, lambdaL2Regularization, activationFunction);

            int stageC = 16; //number of channels for current stage
            for (int stageId = 0; stageId < 3; ++stageId)
            {
                for (int res_block = 0; res_block < num_res_blocks; res_block += 1)
                {
                    int stride = (res_block == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = network.Layers.Last().LayerIndex;
                    network.AddConvolution_BatchNorm_Activation(stageC, 3, stride, 1, lambdaL2Regularization, activationFunction);
                    network.AddConvolution_BatchNorm(stageC, 3, 1, 1, lambdaL2Regularization);
                    network.AddShortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, lambdaL2Regularization);
                    network.AddActivation(activationFunction);
                }
                stageC *= 2;
            }
            network.AddAvgPooling(8, 8);
            network.AddOutput(yShape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return network;
        }

    }
}


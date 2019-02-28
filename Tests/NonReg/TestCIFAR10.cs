using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.GPU;

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestCIFAR10
    {
        [Test, Ignore]
        public void Test()
        {
            var logFileName  = Utils.ConcatenatePathWithFileName(@"c:\temp\ML\", "TestCIFAR10" + "_" + Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            var logger = new Logger(logFileName, true);

            Load(out var xTrainingSet, out var yTrainingSet, out var xTestSet, out var yTestSet);
            ToWorkingSet(xTrainingSet, yTrainingSet, out CpuTensor<float> X_train, out CpuTensor<float> Y_train);
            ToWorkingSet(xTestSet, yTestSet, out CpuTensor<float> X_test, out CpuTensor<float> Y_test);

            /*
            bool useGpu = false;
            var nbRows = 10;
            X_train = (CpuTensor<float>)X_train.ExtractSubTensor(0, nbRows);
            Y_train = (CpuTensor<float>)Y_train.ExtractSubTensor(0, nbRows);
            X_test = (CpuTensor<float>)X_test.ExtractSubTensor(0, Math.Min(10000,nbRows));
            Y_test = (CpuTensor<float>)Y_test.ExtractSubTensor(0, Math.Min(10000, nbRows));
            */

            int batchSize = 32;
            const int numEpochs = 3;
            var learningRate = 0.001;
            var activationFunction = cudnnActivationMode_t.CUDNN_ACTIVATION_ELU;



            //var network = Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\Network_18196_15.txt");network.Config.Logger = logger;

            var networkConfig = new NetworkConfig(true) { Logger = logger, UseDoublePrecision = false, LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy};
            var network = new Network(networkConfig
                .WithAdam()
            );
            double lambdaL2Regularization = 0.0;
            network.AddInput(X_test.Shape[1], X_test.Shape[2], X_test.Shape[3])

                .AddConvolution_Activation(32, 3, 1, 1, lambdaL2Regularization, activationFunction)
                //model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
                //model.add(Activation('elu'))
                
                .AddBatchNorm()
                //#model.add(BatchNormalization())

                .AddConvolution_Activation(32, 3, 1, 1, lambdaL2Regularization, activationFunction)
                //model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
                //model.add(Activation('elu'))
                
                .AddBatchNorm()
                //#model.add(BatchNormalization())

                .AddMaxPooling(2,2)
                //model.add(MaxPooling2D(pool_size=(2,2)))
                .AddDropout(0.2)
                //model.add(Dropout(0.2))

                .AddConvolution_Activation(64, 3, 1, 1, lambdaL2Regularization, activationFunction)
                //model.add(Conv2D(64, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
                //model.add(Activation('elu'))
                //#.AddBatchNorm()
                //#model.add(BatchNormalization())

                .AddConvolution_Activation(64, 3, 1, 1, lambdaL2Regularization, activationFunction)
                //model.add(Conv2D(64, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
                //model.add(Activation('elu'))
                .AddBatchNorm()
                //#model.add(BatchNormalization())
                .AddMaxPooling(2, 2)
                //model.add(MaxPooling2D(pool_size=(2,2)))
                .AddDropout(0.3)
                //model.add(Dropout(0.3))

                .AddConvolution_Activation(128, 3, 1, 1, lambdaL2Regularization, activationFunction)
                //model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
                //model.add(Activation('elu'))
                .AddBatchNorm()
                //model.add(BatchNormalization())
                .AddConvolution_Activation(128, 3, 1, 1, lambdaL2Regularization, activationFunction)
                //model.add(Conv2D(128, (3, 3), padding = 'same', kernel_regularizer = regularizers.l2(weight_decay)))
                //model.add(Activation('elu'))
                .AddBatchNorm()
                //#model.add(BatchNormalization())

                .AddMaxPooling(2, 2)
                //model.add(MaxPooling2D(pool_size=(2,2)))
                .AddDropout(0.4)
                //model.add(Dropout(0.4))

                //model.add(Flatten())
                //model.add(Dense(num_classes, activation='softmax'))

                //.AddDense(100, activationFunction)
                //.AddDropout(0.5)

                .AddOutput(Y_train.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            network.Fit(X_train, Y_train, learningRate , numEpochs, batchSize, X_test, Y_test);
        }


        public static void Load(out CpuTensor<byte> xTrainingSet, out CpuTensor<byte> yTrainingSet, out CpuTensor<byte> xTestSet, out CpuTensor<byte> yTestSet)
        {
            var path = @"C:\Projects\SharpNet\Tests\Data\cifar-10-batches-bin\";
            xTrainingSet = new CpuTensor<byte>(new []{50000,3,32,32}, "xTrainingSet");
            yTrainingSet = new CpuTensor<byte>(new []{50000,1,1,1}, "yTrainingSet");
            xTestSet = new CpuTensor<byte>(new[] { 10000, 3, 32, 32 }, "xTestSet");
            yTestSet = new CpuTensor<byte>(new[] { 10000, 1, 1, 1 }, "yTestSet");
            for (int i = 0; i < 5; ++i)
            {
                LoadAt(Path.Combine(path, "data_batch_" + (i + 1) + ".bin"), xTrainingSet, yTrainingSet, 10000 * i);
            }
            LoadAt(Path.Combine(path, "test_batch.bin"), xTestSet, yTestSet, 0);
        }

        private static void LoadAt(string path, CpuTensor<byte> x, CpuTensor<byte> y, int indexFirst)
        {
            var b = File.ReadAllBytes(path);
            for (int count = 0; count < 10000; ++count)
            {
                int bIndex = count * (1 + 32 * 32 * 3);
                int xIndex = (count + indexFirst) * x.MultDim0;
                int yIndex = (count + indexFirst) * y.MultDim0;
                y[yIndex] = b[bIndex];
                for (int j = 0; j < 32 * 32 * 3; ++j)
                {
                    x[xIndex+j] = b[bIndex+1+j];
                }
            }
        }

        public static void ToWorkingSet(CpuTensor<byte> x, CpuTensor<byte> y, out CpuTensor<float> xWorkingSet, out  CpuTensor<float> yWorkingSet)
        {
            xWorkingSet = x.Select(b=>b/255.0f);
            yWorkingSet = y.ToCategorical(1.0f, out _);
        }

    }
}

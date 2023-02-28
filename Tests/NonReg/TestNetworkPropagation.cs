using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;

namespace SharpNetTests.NonReg
{
    /// <summary>
    /// Test of the forward and backward propagation of gradients in a Neural Network
    /// </summary>
    [TestFixture]
    public class TestNetworkPropagation
    {
        private const string X_1_1_1_1 = @"numpy.array([[[[1]]]], numpy.float)";
        private const string Y_1_2 = @"numpy.array([[1,0]], numpy.float)";
        public const string X_2_1_4_4 = @"numpy.array([[[[0.7262433,0.8173254,0.7680227,0.5581612],[0.2060332,0.5588848,0.9060271,0.4421779],[0.9775497,0.2737045,0.2919063,0.4673147],[0.6326591,0.4695119,0.9821513,0.03036699]]],[[[0.8623701,0.9953471,0.6771811,0.3145918],[0.8169079,0.8480518,0.9919022,0.0326252],[0.699942,0.5262842,0.9340187,0.6876203],[0.5468155,0.08110995,0.1871246,0.4533272]]]], numpy.float)";
        public const string Y_2_3 = @"numpy.array([[1,0,0], [0,0,1]], numpy.float)";
        public const string Y_3_3 = @"numpy.array([[1,0,0], [0,0,1], [0,1,0]], numpy.float)";
        private const string X_1_1_4_4 = @"numpy.array([[[[0.7262433,0.8173254,0.7680227,0.5581612],[0.2060332,0.5588848,0.9060271,0.4421779],[0.9775497,0.2737045,0.2919063,0.4673147],[0.6326591,0.4695119,0.9821513,0.03036699]]]], numpy.float)";
        public const string X_2_3_4_5 = "numpy.array([[[[0.67872983,0.95197606,0.8040681,0.17448357,-0.88190055],[0.17665438,1.2180812,-0.17346638,1.4326493,-0.67888665],[-0.62428117,-0.0980559,0.39797723,-0.09146436,1.4464538],[-1.4088991,1.0871105,1.4860412,0.53154343,-0.55622464]],[[0.9507237,1.0441554,1.4757066,-1.4021244,0.599826],[0.07885243,1.302056,0.56286085,0.1404463,-1.2566701],[-0.9386263,-0.14001845,-0.6084844,1.4656314,0.42809242],[0.7888908,-1.4088172,-0.3569865,-0.47057447,1.3723655]],[[0.0153876385,0.6479175,-1.1431268,-0.67964554,1.2212938],[0.8842968,-0.48851898,-0.12837364,-1.0595249,-0.83605576],[-0.26978016,0.6561805,0.35949084,-0.036095843,-0.9152597],[1.1345744,0.97626954,0.7061925,1.0746577,0.53924704]]],[[[0.3745984,-0.8447797,1.1860874,1.1893194,-1.2402604],[1.0157373,-0.9895715,0.43234587,0.98044324,-0.842437],[1.4999181,-0.05590306,0.45661572,0.65917796,0.92838776],[-1.1971939,0.68728215,0.65535206,-1.1234182,1.2155292]],[[-0.9326286,-0.069570385,0.122385725,-0.52349794,0.51311],[-0.094806775,1.0004907,0.9276114,0.880891,0.79351795],[1.3126212,-0.6150096,0.068355896,0.4901785,-0.5022329],[-0.63983274,0.9618302,-0.8324462,-0.9393852,-1.2944435]],[[0.3738683,0.5791351,0.39118314,1.2829121,-0.83597386],[1.2861229,-1.3004352,1.2003129,0.53551644,-1.2180659],[-1.0527077,-1.1790825,0.32961074,1.3591285,-0.028124375],[-1.0558312,0.53283465,0.20958523,-0.8237906,0.35454643]]]], numpy.float)";
        public const string X_3_4_5 = "numpy.array([[[0.67872983,0.95197606,0.8040681,0.17448357,-0.88190055],[0.17665438,1.2180812,-0.17346638,1.4326493,-0.67888665],[-0.62428117,-0.0980559,0.39797723,-0.09146436,1.4464538],[-1.4088991,1.0871105,1.4860412,0.53154343,-0.55622464]],[[0.9507237,1.0441554,1.4757066,-1.4021244,0.599826],[0.07885243,1.302056,0.56286085,0.1404463,-1.2566701],[-0.9386263,-0.14001845,-0.6084844,1.4656314,0.42809242],[0.7888908,-1.4088172,-0.3569865,-0.47057447,1.3723655]],[[0.0153876385,0.6479175,-1.1431268,-0.67964554,1.2212938],[0.8842968,-0.48851898,-0.12837364,-1.0595249,-0.83605576],[-0.26978016,0.6561805,0.35949084,-0.036095843,-0.9152597],[1.1345744,0.97626954,0.7061925,1.0746577,0.53924704]]], numpy.float)";
        public const string Y_2_2 = @"numpy.array([[1,0],[1,0]], numpy.float)";
        private const string Y_1_3 = @"numpy.array([[1,0,0]], numpy.float)";
        private const string W_N_1_4_4 = "[[0.22065729, -0.11788255, -0.4187895],[0.32060236, -0.44626778, 0.24227637],[-0.46897227, 0.5059137, 0.4339162],[-0.02144825, -0.04082066, -0.09005189],[0.28492624, -0.28046286, -0.18176123],[-0.1717251, -0.55430335, -0.28846815],[0.29476583, -0.3019745, 0.03277987],[0.41012663, 0.09135884, 0.2522431],[-0.40020466, -0.2832676, 0.2568243],[0.47819465, 0.06466031, 0.45569366],[0.4343483, -0.30980763, -0.01376414],[0.09202623, -0.02883267, 0.19485158],[-0.5382978, -0.5129023, 0.47553152],[0.15798962, 0.43635488, 0.4626748],[-0.47213712, 0.17086667, -0.03163177],[0.01544881, 0.26190037, 0.38539213]]";

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestSigmoidActivation_NCHW_1_1_1_1(List<int> resourceIds)
        {
            var X = FromNumpyArray(X_1_1_1_1);
            var Y = FromNumpyArray(Y_1_2);
            var network = GetNetwork(EvaluationMetricEnum.BinaryCrossentropy, resourceIds);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3]).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray("[[0.5553087, -0.2966646]]");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.635366380214691,0.426373064517975]]");
            TestLossAccuracy(network, X, Y, 0.504664778709412, 1.0);

            const double learningRate = 0.1;
            const int numEpochs = 10;
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.707915127277374,0.336254686117172]]");
            TestLossAccuracy(network, X, Y, 0.377643942832947, 1.0);
        }

        private static List<List<int>>  GetTestCases()
        {
            var result = new List<List<int>>();
            result.Add(new List<int> {-1}); //single CPU
            if (GPUWrapper.GetDeviceCount() >= 1)
            {
                result.Add(new List<int> { 0 }); // single GPU
            }
            result.Add(new List<int> { -1,-2 }); //multi CPU
            if (GPUWrapper.GetDeviceCount() >= 2)
            {
                result.Add(Enumerable.Range(0, GPUWrapper.GetDeviceCount()).ToList()); //multi GPU
            }
            return result;
        }


        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestSigmoidActivation_NCHW_1_1_4_4(List<int> resourceIds)
        {
            var X = FromNumpyArray(X_1_1_4_4);
            var Y = FromNumpyArray(Y_1_3);
            var network = GetNetwork(EvaluationMetricEnum.BinaryCrossentropy, resourceIds);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3]).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray(W_N_1_4_4);
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.41122958, 0.27045158, 0.7466786]]");
            TestLossAccuracy(network, X, Y, 0.8590097427368164, 0.0);

            const double learningRate = 0.1;
            const int numEpochs = 10;
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.67996985, 0.17650272, 0.4107801]]");
            TestLossAccuracy(network, X, Y, 0.369619220495224, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestSigmoidActivation_NCHW_2_1_4_4(List<int> resourceIds)
        {
            var X = FromNumpyArray(X_2_1_4_4);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.BinaryCrossentropy, resourceIds);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3]).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray(W_N_1_4_4);
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.41122958, 0.27045155, 0.7466786],[0.65826976, 0.14434774, 0.69001585]]");
            TestLossAccuracy(network, X, Y, 0.6962824463844299, 0.5);

            const double learningRate = 0.1;
            const int numEpochs = 10;
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.4153017,  0.19545524, 0.6464454],[0.6064757,  0.09901482, 0.62172073]]");
            TestLossAccuracy(network, X, Y, 0.6080149412155151, 0.5);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestSoftmaxActivation_NCHW_2_1_4_4(List<int> resourceIds)
        {
            var X = FromNumpyArray(X_2_1_4_4);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3]).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            var w = FromNumpyArray("[[ 0.02377093, -0.36406565,  0.20111328],[-0.02257341,  0.49298579,  0.3783552 ],[-0.33265597,  0.22183669,  0.4130335 ],[ 0.03862739,  0.45694906,-0.046529  ],[-0.5435763 ,  0.4115948 ,  0.5266854 ],[-0.04584688, -0.08123899,  0.43348545],[-0.23025852, -0.24818823, -0.31672138],[-0.13403434, -0.39957535,  0.34845835],[-0.11953372, -0.18876502, -0.19744089],[-0.5492821 ,  0.52302474,  0.3208636 ],[ 0.18945718, 0.04014206, -0.3605097 ],[-0.47365752, -0.26253745, -0.2964717 ],[-0.2434968 , -0.34853765, -0.23780361],[ 0.4313671 ,  0.5169173 , -0.43086883],[ 0.00898802,  0.24687833,  0.17265934],[ 0.02312517, -0.22023779,  0.3136925 ]]");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.11047623,0.41491902,0.47460473],[0.05679994,0.34877774,0.59442234]]");
            TestLossAccuracy(network, X, Y, 1.3615601062774658, 0.5);

            const double learningRate = 0.1;
            const int numEpochs = 10;
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.6009891,0.09348926,0.30552167],[0.2477788,0.08346885,0.6687523]]");
            TestLossAccuracy(network, X, Y, 0.45576000213623047, 1.0);
        }

        public static CpuTensor<float> FromNumpyArray(string s)
        {
            return (CpuTensor<float>)TensorExtensions.FromNumpyArray(s);
        }


        /// <returns></returns>
        /// <summary>
        /// Load a Convolution from a Keras/TensorFlow format:
        ///     shape for standard convolution:
        ///         (f2,f1, inputChannels, filtersCount)
        ///     shape for depthwise convolution
        ///         (f2,f1, inputChannels, depthMultiplier)
        /// into a SharpNet format:
        ///     shape for standard convolution:
        ///         (filtersCount=outputChannels, inputChannels, f1,f2)
        ///     shape for depthwise convolution
        ///         (depthMultiplier, inputChannels=outputChannels, f1,f2)
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>/// 
        public static CpuTensor<float> FromConvNumpyArray(string s)
        {
            var result = (CpuTensor<float>)TensorExtensions.FromNumpyArray(s);
            if (result.Shape.Length == 3)
            {
                return (CpuTensor<float>)result.ChangeAxis(new[] { 2, 1, 0 });
            }

            return (CpuTensor<float>)result.ChangeAxis(new[] { 3, 2, 0, 1 });
        }

   
        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestReluActivation_NCHW_2_1_4_4(List<int> resourceIds)
        {
            var X = FromNumpyArray(X_2_1_4_4);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.BinaryCrossentropy, resourceIds);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3]).Dense(3, 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray("[[ 0.22065729, -0.11788255, -0.4187895 ],[ 0.32060236, -0.44626778,  0.24227637],[-0.46897227,  0.5059137 ,  0.4339162 ],[-0.02144825, -0.04082066, -0.09005189],[ 0.28492624, -0.28046286, -0.18176123],[-0.1717251 , -0.55430335, -0.28846815],[ 0.29476583, -0.3019745 ,  0.03277987],[ 0.41012663,  0.09135884,  0.2522431 ],[-0.40020466, -0.2832676 ,  0.2568243 ],[ 0.47819465,  0.06466031,  0.45569366],[ 0.4343483 , -0.30980763, -0.01376414],[ 0.09202623, -0.02883267,  0.19485158],[-0.5382978 , -0.5129023 ,  0.47553152],[ 0.15798962,  0.43635488,  0.4626748 ],[-0.47213712,  0.17086667, -0.03163177],[ 0.01544881,  0.26190037,  0.38539213]]");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);
            w = FromNumpyArray("[[ 0.7206471 , -0.3155403 ,  0.16133356],[ 0.4253831 ,  0.71631813,  0.10403013],[ 0.4923072 ,  0.58519197,  0.364321  ]]");
            w.CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.6299896,  0.65307385, 0.5972025 ],[0.7039944 , 0.56498057, 0.5980379]]");
            TestLossAccuracy(network, X, Y, 0.8323098421096802, 0.0);

            const double learningRate = 0.1;
            const int numEpochs = 10;
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.5302857,  0.49406603, 0.5208457 ],[0.5566553, 0.4216699,  0.5151303]]");
            TestLossAccuracy(network, X, Y, 0.6792957186698914, 0.5);
        }


        /// <summary>
        /// TODO: fix bug when using several CPU to compute the embedding
        /// </summary>
        /// <param name="resourceIds"></param>
        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestEmbedding(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double momentum = 0.9;
            const int vocabularySize = 3;
            const int embeddingDim = 5;
            const int maxWordsBySentence = 4;

            var X = FromNumpyArray(@"numpy.array([[1, 2, 1, 1], [2, 2, 1, 1]], numpy.float)");
            var Y = FromNumpyArray(@"numpy.array([[1], [0]], numpy.float)");
            var network = GetNetwork(EvaluationMetricEnum.BinaryCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            Debug.Assert(network.Layers.Count == 0);
            network.Input(maxWordsBySentence, -1, -1)
                .Embedding(new [] { vocabularySize }, new[] { embeddingDim }, new[] { -1 }, 0.0)
                .Flatten()
                .Dense(1, 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            FromNumpyArray("[[-0.020802486687898636, -0.02934335544705391, 0.0035390742123126984, 0.006125748157501221, -0.008332550525665283], [0.0307827927172184, -0.0006774887442588806, 0.0498129241168499, 0.019673515111207962, -0.037462640553712845],[0.020981673151254654, 0.016241561621427536, 0.007225655019283295, -0.013524651527404785, -0.007948171347379684]]")
                .CopyTo(((EmbeddingLayer)network.Layers[1]).Weights);
            FromNumpyArray("[[0.05924016237258911], [-0.2979503273963928], [0.39012110233306885], [0.2964285612106323], [0.15513628721237183], [0.032458603382110596], [-0.5190843939781189], [0.3992980718612671], [-0.03236877918243408], [-0.12109190225601196], [0.4128159284591675], [0.14623379707336426], [-0.5325161814689636], [0.38246530294418335], [-0.4191945493221283], [0.4918263554573059], [-0.30854684114456177], [0.1737397313117981], [-0.40517792105674744], [-0.3750319480895996]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.514347792],[0.507476687]]");
            TestLossAccuracy(network, X, Y, 0.6865345239639282, 0.5);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.58809334],[0.322747618]]");
            TestLossAccuracy(network, X, Y, 0.46029043197631836, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestEmbedding_Adam_Sigmoid(List<int> resourceIds)
        {

            const int numEpochs = 5;
            const double learningRate = 0.01;
            //var momentum = 0.9;
            const int batchSize = 2;
            const int vocabularySize = 3;
            const int embeddingDim = 5;
            const int maxWordsBySentence = 4;

            var X = FromNumpyArray(@"numpy.array([[1, 1, 1, 2], [2, 2, 2, 2], [1, 2, 2, 2],[1, 1, 1, 1]], numpy.float)");
            var Y = FromNumpyArray(@"numpy.array([[1], [0], [0], [1]], numpy.float)");
            
            var network = GetNetwork(EvaluationMetricEnum.BinaryCrossentropy, resourceIds);
            network.Sample.WithAdam(0.9, 0.999, 1e-7);

            Debug.Assert(network.Layers.Count == 0);
            network.Input(maxWordsBySentence, -1, -1)
                .Embedding(new [] { vocabularySize }, new[] { embeddingDim }, new[] { -1 }, 0.0)
                .GlobalAvgPoolingOnHeight()
                .Dense(4, 0.0, false).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(1, 0.0, false).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            FromNumpyArray("[[-0.020802486687898636, -0.02934335544705391, 0.0035390742123126984, 0.006125748157501221, -0.008332550525665283], [0.0307827927172184, -0.0006774887442588806, 0.0498129241168499, 0.019673515111207962, -0.037462640553712845], [0.020981673151254654, 0.016241561621427536, 0.007225655019283295, -0.013524651527404785, -0.007948171347379684]]")
                .CopyTo(((EmbeddingLayer)network.Layers[1]).Weights);
            FromNumpyArray("[[0.09049081802368164, -0.45512667298316956, 0.5959198474884033, 0.4528021812438965], [0.2369745969772339, 0.04958134889602661, -0.7929145097732544, 0.6099379062652588], [-0.04944407939910889, -0.18497097492218018, 0.6305867433547974, 0.22337579727172852], [-0.813431978225708, 0.5842254161834717, -0.6403303146362305, 0.7512772083282471], [-0.47131311893463135, 0.26539182662963867, -0.6189195513725281, -0.5728708505630493]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);
            FromNumpyArray("[[-0.6677531003952026], [0.5261931419372559], [-0.026724934577941895], [0.8222856521606445]]")
                .CopyTo(((DenseLayer)network.Layers[5]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.5087772607803345], [0.49968811869621277], [0.5027181506156921], [0.5118059515953064]]", 1e-4);
            TestLossAccuracy(network, X, Y, 0.6841691732406616, 0.75);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            //predictions after training
            TestPredict(network, X, "[[0.5486243367195129], [0.45513594150543213], [0.4749670624732971], [0.5833647847175598]]", 1e-4);
            TestLossAccuracy(network, X, Y, 0.5976990461349487, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestLeakyReluActivation_NCHW_2_1_4_4(List<int> resourceIds)
        {
            const double learningRate = 0.1;
            const int numEpochs = 10;
            const double momentum = 0.9;

            var X = FromNumpyArray(X_2_1_4_4);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.BinaryCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Dense(3, 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU, Tensor.SingleFloat(0.1f)).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            FromNumpyArray("[[0.023770928382873535, -0.36406564712524414, 0.2011132836341858], [-0.022573411464691162, 0.4929857850074768, 0.3783552050590515], [-0.33265596628189087, 0.22183668613433838, 0.41303348541259766], [0.03862738609313965, 0.45694905519485474, -0.04652899503707886], [-0.5435763001441956, 0.41159480810165405, 0.5266854166984558], [-0.04584687948226929, -0.08123898506164551, 0.4334854483604431], [-0.2302585244178772, -0.24818822741508484, -0.3167213797569275], [-0.13403433561325073, -0.3995753526687622, 0.34845834970474243], [-0.1195337176322937, -0.1887650191783905, -0.19744089245796204], [-0.5492820739746094, 0.5230247378349304, 0.32086360454559326], [0.18945717811584473, 0.040142059326171875, -0.3605096936225891], [-0.4736575186252594, -0.2625374495983124, -0.2964716851711273], [-0.24349680542945862, -0.3485376536846161, -0.2378036081790924], [0.43136709928512573, 0.5169172883033752, -0.43086883425712585], [0.008988022804260254, 0.24687832593917847, 0.17265933752059937], [0.023125171661376953, -0.22023779153823853, 0.31369251012802124]]")
                .CopyTo(((DenseLayer)network.Layers[1]).Weights);
            FromNumpyArray("[[-0.3802216053009033, -0.8489081859588623, -0.08725166320800781], [-0.7802162170410156, -0.7194366455078125, -0.9523963928222656], [-0.11738991737365723, 0.7826731204986572, -0.5935578346252441]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.4370572865009308, 0.5525224208831787, 0.3667464256286621], [0.43723082542419434, 0.6376928091049194, 0.32504063844680786]]");
            TestLossAccuracy(network, X, Y, 0.8004429340362549, 0);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X,"[[0.3570849895477295, 0.1356194019317627, 0.339799165725708], [0.3754737973213196, 0.15078075230121613, 0.35905107855796814]]");
            TestLossAccuracy(network, X, Y, 0.5415375232696533, 0.5);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestConvolutionWithReluActivation_NCHW_2_1_4_4(List<int> resourceIds)
        {
            var X = FromNumpyArray(X_2_1_4_4);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.BinaryCrossentropy, resourceIds);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(3, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray("[[[[-0.10066611, -0.22584948,  0.1257661 ]],[[ 0.00622791, -0.02702722, -0.19816945]],[[-0.00094005, -0.12673107,  0.10199177]]],[[[-0.05160269,  0.36979204, -0.38235503]],[[-0.25580615, -0.23532738, -0.18486507]],[[-0.18581466, -0.03875312, -0.18673505]]],[[[-0.1438927 , -0.05969113,  0.22153592]],[[ 0.02154535,  0.143184  ,  0.2194677 ]],[[-0.17963122,  0.14435953,  0.18853426]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            w = FromNumpyArray("[[ 0.05013201,  0.0884136 ,  0.1288763 ],[ 0.10524932,  0.27004865, -0.15511033],[ 0.28799555, -0.11378004,  0.31027994],[-0.12980627, -0.26656348, -0.2889419 ],[ 0.10056138, -0.20606633,  0.11035499],[-0.19916984,  0.01184309, -0.02502242],[-0.00895432, -0.23922653, -0.14380434],[ 0.13250148,  0.12896249,  0.3411176 ],[-0.20010757, -0.07243675,  0.10569999],[-0.14625986, -0.2575507 , -0.2796294 ],[ 0.2984304 ,  0.12682551, -0.34131444],[ 0.33970162, -0.2596441 , -0.28711483],[ 0.2641308 ,  0.15033874, -0.17174129],[-0.31036156,  0.15232903, -0.2033331 ],[-0.0004667 ,  0.15065774,  0.12756902],[ 0.2866663 , -0.160675  , -0.12804145],[ 0.01153374, -0.11623923, -0.08252719],[ 0.12417665,  0.28663734, -0.12360954],[ 0.13087502,  0.15079209,  0.29951695],[-0.0907169 , -0.27126557,  0.00555232],[ 0.19179931, -0.2861278 ,  0.07780427],[-0.20458487, -0.27085418,  0.04733434],[-0.10611108, -0.09193736,  0.19488677],[ 0.13467175, -0.2872713 ,  0.2647117 ],[-0.24014097, -0.02662796,  0.22110483],[ 0.33133528, -0.18674679, -0.04942989],[ 0.07396188, -0.18812832, -0.14777936],[ 0.13951644, -0.29781634, -0.12320091],[-0.01970455, -0.22537778, -0.05007559],[-0.10169415, -0.3120061 ,  0.0934028 ],[-0.13796891, -0.31914735, -0.11247423],[ 0.20420077, -0.12212758, -0.30907962],[-0.25789154,  0.2055321 ,  0.11365542],[-0.10406806,  0.2673215 , -0.1856383 ],[ 0.05355045,  0.1597245 , -0.13172172],[ 0.14546981,  0.26738545,  0.02670237],[ 0.08399773, -0.12938716, -0.04259995],[-0.13436754,  0.25714287, -0.01506558],[-0.26373556,  0.31247166, -0.0151737 ],[-0.058229  ,  0.2936549 ,  0.2405878 ],[-0.29457894,  0.05585265, -0.33545914],[-0.12306491, -0.32960945, -0.01645941],[-0.04173017,  0.24279085,  0.21392396],[-0.20707619,  0.1420064 , -0.16330862],[-0.07069319,  0.312768  , -0.2855286 ],[ 0.07745105, -0.17894101,  0.3308623 ],[ 0.21007964, -0.25078928,  0.19156727],[ 0.02520046, -0.11668615,  0.3065426 ]]");
            w.CopyTo(network.Layers[2].Weights);

            //predictions before training
            TestPredict(network, X, "[[0.3302841,0.7452456,0.4357071],[0.2857407,0.7822333,0.4093774]]");
            TestLossAccuracy(network, X, Y, 0.966899474461873, 0.0);

            const double learningRate = 0.1;
            const int numEpochs = 10;
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.5584541,0.1351326,0.4495322],[0.4868227,0.1388872,0.4988887]]");
            TestLossAccuracy(network, X, Y, 0.472797473271688, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestBatchNormalization_NCHW_2_1_4_4(List<int> resourceIds)
        {
            if (resourceIds.Count >= 2)
            {
                //because of the non trainable parameters in batchNorm layer, results are not the same in single gpu vs multi gpu
                return;
            }

            var X = FromNumpyArray(X_2_1_4_4);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.BinaryCrossentropy, resourceIds);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Flatten()
                .BatchNorm(0.99, 1e-5, "BatchNorm").Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray("[[ 0.5398403 ,  0.3300526 ,  0.55018014],[ 0.14779323, -0.34529093, -0.16082811],[ 0.25194514, -0.5108937 ,  0.22403759],[-0.15200073,  0.0372324 , 0.38966674],[-0.06222537, -0.49777025,  0.17976868],[-0.30487314,  0.29002702, -0.1486885 ],[ 0.21023047, -0.5419708 , -0.44205534],[ 0.03658289, 0.5347499 ,  0.04729468],[-0.29706508, -0.5559816 ,  0.54104596],[-0.526604  ,  0.12949431, -0.07999322],[ 0.07322848,  0.3647598 ,  0.03496403],[-0.5040164 ,  0.03338426, -0.34131938],[ 0.3909973 ,  0.22031981,  0.2741294 ],[ 0.36716205, -0.21828368,  0.42880273],[-0.03759038, 0.17174226, -0.33242768],[-0.26423737, -0.43534094, -0.30766475]]");
            w.CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.5917179,0.3094307,0.6521813],[0.497325,0.2508813,0.5720119]]");
            TestLossAccuracy(network, X, Y, 0.581050515174866, 0.5);

            const double learningRate = 0.1;
            const int numEpochs = 10;
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.606776178,0.248665854,0.616904914],[0.43677336,0.206704035,0.77601999]]");
            TestLossAccuracy(network, X, Y, 0.46736355622609455d, 0.5);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestBatchNormalizationNchw2345(List<int> resourceIds)
        {
            if (resourceIds.Count >= 2)
            {
                //because of the non trainable parameters in batchNorm layer, results are not the same in single gpu vs multi gpu
                return;
            }

            const int numEpochs = 10;
            const double learningRate = 0.001;

            var X = FromNumpyArray(X_2_3_4_5);
            var Y = FromNumpyArray(Y_2_2);

            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);

            network.Sample.WithSGD(0.9, false);

            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 5, 1, ConvolutionLayer.PADDING_TYPE.SAME, 0.00, false)
                .BatchNorm(0.99, 0.001)
                .Flatten()
                .Dense(Y.Shape[1], 0.00, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            FromConvNumpyArray("[[[[-0.09115192294120789, -0.1285761296749115], [0.015507444739341736, 0.02684168517589569], [-0.03651140630245209, 0.13488344848155975]], [[-0.00296860933303833, 0.21826930344104767], [0.08620502054691315, -0.164153054356575], [0.09193708002567291, 0.07116694748401642]], [[0.031661227345466614, -0.059262052178382874], [-0.03482714295387268, 0.056988105177879333], [0.18132375180721283, 0.07083024084568024]], [[0.1461208015680313, -0.18230044841766357], [-0.0965045914053917, -0.21228709816932678], [0.09919191896915436, 0.11635322868824005]], [[0.07881362736225128, 0.01434066891670227], [0.11239881813526154, -0.1983097642660141], [-0.1970173716545105, 0.1103084534406662]]], [[[-0.14341005682945251, -0.08240586519241333], [-0.0914153903722763, -0.17504669725894928], [-0.1464957296848297, 0.11816133558750153]], [[0.03754298388957977, 0.21120603382587433], [0.18177057802677155, -0.1570143699645996], [-0.17633794248104095, 0.0465688556432724]], [[-0.1411276012659073, 0.007909983396530151], [0.21125487983226776, -0.14206631481647491], [-0.1990942358970642, 0.042740508913993835]], [[0.027585193514823914, 0.13367687165737152], [0.036774829030036926, -0.17390123009681702], [0.19494734704494476, -0.09526071697473526]], [[0.023158341646194458, -0.09662055224180222], [-0.18425174057483673, -0.04842127859592438], [-0.1884990632534027, -0.20071429014205933]]], [[[0.185743048787117, 0.1644103080034256], [-0.014039069414138794, -0.013181895017623901], [0.055976465344429016, 0.10087977349758148]], [[0.19063322246074677, -0.06231115758419037], [-0.12864384055137634, 0.045224860310554504], [-0.043046414852142334, 0.038624927401542664]], [[0.16180698573589325, 0.13950984179973602], [-0.215886652469635, -0.11348751187324524], [-0.13103963434696198, -0.20128756761550903]], [[0.20945246517658234, -0.1445169746875763], [0.08663348853588104, 0.10372905433177948], [0.1503768414258957, 0.10335837304592133]], [[0.1012752503156662, -0.13684993982315063], [0.10320423543453217, -0.13648200035095215], [0.0363188236951828, -0.20214590430259705]]], [[[0.07519437372684479, 0.03793402016162872], [-0.21869845688343048, -0.011379584670066833], [0.19861914217472076, 0.1021869033575058]], [[0.04015164077281952, -0.06955106556415558], [0.11807812750339508, -0.1930990219116211], [0.07289008796215057, -0.010597199201583862]], [[-0.18950974941253662, 0.20634375512599945], [-0.01657506823539734, 0.0003098100423812866], [-0.169576033949852, 0.10081981122493744]], [[0.11222796142101288, -0.029622197151184082], [-0.1983664482831955, 0.016455188393592834], [-0.1479426920413971, -0.045503586530685425]], [[0.02768459916114807, -0.17393001914024353], [0.19610513746738434, -0.08166548609733582], [-0.01459348201751709, 0.0388517826795578]]], [[[-0.1761961728334427, -0.08882971107959747], [-0.12129751592874527, 0.07365266978740692], [-0.09969687461853027, 0.17206580936908722]], [[0.11302091181278229, 0.060240671038627625], [0.10617290437221527, 0.08903200924396515], [-0.04626387357711792, 0.21839921176433563]], [[0.054210200905799866, -0.044452205300331116], [0.0944950133562088, -0.17584437131881714], [-0.05926632881164551, -0.0757722407579422]], [[-0.03572949767112732, 0.0447700172662735], [0.08026386797428131, 0.06289060413837433], [0.17721955478191376, -0.19903990626335144]], [[-0.016498178243637085, -0.01501627266407013], [-0.07061576843261719, -0.16451311111450195], [-0.10312797129154205, -0.001189500093460083]]]]").CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            FromNumpyArray("[[0.04188910126686096, -0.2106827050447464], [0.2758572995662689, 0.20960667729377747], [0.10969790816307068, 0.02295169234275818], [-0.36704808473587036, 0.2823463976383209], [-0.02288818359375, -0.08562490344047546], [0.2919049561023712, 0.10340291261672974], [-0.37654581665992737, 0.2704438269138336], [-0.2964152991771698, 0.3477737605571747], [-0.21817557513713837, 0.12285253405570984], [-0.2865040600299835, -0.26518765091896057], [0.3682301938533783, 0.26051589846611023], [-0.3306047320365906, 0.11873883008956909], [-0.029090523719787598, 0.3655906021595001], [0.29481104016304016, 0.2010950744152069], [0.18971768021583557, -0.15412551164627075], [-0.030991554260253906, 0.13511088490486145], [-0.18393947184085846, 0.11978361010551453], [0.21475091576576233, 0.36932632327079773], [-0.21457472443580627, -0.3646804392337799], [-0.000677645206451416, -0.30406829714775085], [0.35434862971305847, 0.26963743567466736], [0.27055367827415466, -0.02103760838508606], [0.01838022470474243, 0.1418454349040985], [0.06422442197799683, 0.16968736052513123], [-0.3381834924221039, -0.20134034752845764], [0.3691190183162689, -0.26925402879714966], [0.2939138114452362, 0.2029973566532135], [-0.18334481120109558, -0.3666633069515228], [0.22322401404380798, -0.004866063594818115], [0.15571215748786926, -0.021110326051712036], [0.1831832230091095, -0.1317398101091385], [-0.04628628492355347, -0.21135306358337402], [0.08837255835533142, -0.2636321187019348], [-0.1075061559677124, 0.36796334385871887], [0.2639167010784149, -0.34314271807670593], [-0.28097736835479736, -0.1483306586742401], [0.049575090408325195, -0.3180835247039795], [-0.1435403972864151, 0.2087484896183014], [0.05358928442001343, 0.19163057208061218], [0.19200566411018372, -0.16791696846485138]]").CopyTo(((DenseLayer)network.Layers[4]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.4918535053730011, 0.5081464648246765], [0.8676233887672424, 0.13237664103507996]]");
            TestLossAccuracy(network, X, Y, 0.42578595876693726, 0.5);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.7446084022521973, 0.25539156794548035], [0.8963165879249573, 0.1036834716796875]]");
            TestLossAccuracy(network, X, Y, 0.2021792083978653, 1.0);
        }



        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestLayerNormalizationNchw2345(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.001;

            var X = FromNumpyArray(X_2_3_4_5);
            var Y = FromNumpyArray(Y_2_2);

            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);

            network.Sample.WithSGD(0.9, false);

            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 5, 2, ConvolutionLayer.PADDING_TYPE.SAME, 0.00, false)
                .LayerNorm(1, 0.001)
                .Flatten()
                .Dense(Y.Shape[1], 0.00, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            TestNetworkPropagation.FromConvNumpyArray("[[[[-0.09115192294120789, -0.1285761296749115], [0.015507444739341736, 0.02684168517589569], [-0.03651140630245209, 0.13488344848155975]], [[-0.00296860933303833, 0.21826930344104767], [0.08620502054691315, -0.164153054356575], [0.09193708002567291, 0.07116694748401642]], [[0.031661227345466614, -0.059262052178382874], [-0.03482714295387268, 0.056988105177879333], [0.18132375180721283, 0.07083024084568024]], [[0.1461208015680313, -0.18230044841766357], [-0.0965045914053917, -0.21228709816932678], [0.09919191896915436, 0.11635322868824005]], [[0.07881362736225128, 0.01434066891670227], [0.11239881813526154, -0.1983097642660141], [-0.1970173716545105, 0.1103084534406662]]], [[[-0.14341005682945251, -0.08240586519241333], [-0.0914153903722763, -0.17504669725894928], [-0.1464957296848297, 0.11816133558750153]], [[0.03754298388957977, 0.21120603382587433], [0.18177057802677155, -0.1570143699645996], [-0.17633794248104095, 0.0465688556432724]], [[-0.1411276012659073, 0.007909983396530151], [0.21125487983226776, -0.14206631481647491], [-0.1990942358970642, 0.042740508913993835]], [[0.027585193514823914, 0.13367687165737152], [0.036774829030036926, -0.17390123009681702], [0.19494734704494476, -0.09526071697473526]], [[0.023158341646194458, -0.09662055224180222], [-0.18425174057483673, -0.04842127859592438], [-0.1884990632534027, -0.20071429014205933]]], [[[0.185743048787117, 0.1644103080034256], [-0.014039069414138794, -0.013181895017623901], [0.055976465344429016, 0.10087977349758148]], [[0.19063322246074677, -0.06231115758419037], [-0.12864384055137634, 0.045224860310554504], [-0.043046414852142334, 0.038624927401542664]], [[0.16180698573589325, 0.13950984179973602], [-0.215886652469635, -0.11348751187324524], [-0.13103963434696198, -0.20128756761550903]], [[0.20945246517658234, -0.1445169746875763], [0.08663348853588104, 0.10372905433177948], [0.1503768414258957, 0.10335837304592133]], [[0.1012752503156662, -0.13684993982315063], [0.10320423543453217, -0.13648200035095215], [0.0363188236951828, -0.20214590430259705]]], [[[0.07519437372684479, 0.03793402016162872], [-0.21869845688343048, -0.011379584670066833], [0.19861914217472076, 0.1021869033575058]], [[0.04015164077281952, -0.06955106556415558], [0.11807812750339508, -0.1930990219116211], [0.07289008796215057, -0.010597199201583862]], [[-0.18950974941253662, 0.20634375512599945], [-0.01657506823539734, 0.0003098100423812866], [-0.169576033949852, 0.10081981122493744]], [[0.11222796142101288, -0.029622197151184082], [-0.1983664482831955, 0.016455188393592834], [-0.1479426920413971, -0.045503586530685425]], [[0.02768459916114807, -0.17393001914024353], [0.19610513746738434, -0.08166548609733582], [-0.01459348201751709, 0.0388517826795578]]], [[[-0.1761961728334427, -0.08882971107959747], [-0.12129751592874527, 0.07365266978740692], [-0.09969687461853027, 0.17206580936908722]], [[0.11302091181278229, 0.060240671038627625], [0.10617290437221527, 0.08903200924396515], [-0.04626387357711792, 0.21839921176433563]], [[0.054210200905799866, -0.044452205300331116], [0.0944950133562088, -0.17584437131881714], [-0.05926632881164551, -0.0757722407579422]], [[-0.03572949767112732, 0.0447700172662735], [0.08026386797428131, 0.06289060413837433], [0.17721955478191376, -0.19903990626335144]], [[-0.016498178243637085, -0.01501627266407013], [-0.07061576843261719, -0.16451311111450195], [-0.10312797129154205, -0.001189500093460083]]]]").CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.07255405187606812, -0.3649131655693054], [0.47779881954193115, 0.36304938793182373], [0.19000232219696045, 0.039753496646881104], [-0.6357459425926208, 0.489038348197937], [-0.03964346647262573, -0.1483067274093628], [0.5055941343307495, 0.17909908294677734], [-0.65219646692276, 0.46842241287231445], [-0.5134063959121704, 0.602361798286438], [-0.3778911828994751, 0.21278679370880127], [-0.4962396025657654, -0.45931848883628845], [0.6377934217453003, 0.45122671127319336], [-0.5726242065429688, 0.20566171407699585]]").CopyTo(((DenseLayer)network.Layers[4]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.598140478,0.401859522],[0.107721858,0.892278075]]");
            TestLossAccuracy(network, X, Y, 1.3710662126541138, 0.5);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.887947023,0.112052985],[0.839851797,0.160148203]]");
            TestLossAccuracy(network, X, Y, 0.14668652415275574, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestResNet_Shortcut_Same_Dimension_NCHW_2_1_4_4(List<int> resourceIds)
        {
            var X = FromNumpyArray(X_2_1_4_4);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true)
                .AddLayer(2, 1).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            Tensor w = FromNumpyArray("[[[[-0.7714059]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            w = FromNumpyArray("[[[-1.0248963]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Weights);
            w = FromNumpyArray("[[-0.0961059 , -0.39184043, -0.2705268 ],[0.07956541, 0.22585392, -0.341879],[-0.27921697, -0.32074332, 0.23311937],[-0.5426143, -0.2890964, -0.18300149],[0.10221738, -0.05122566, 0.1905775],[0.5236904, 0.12203938, 0.30513626],[0.17442077, -0.38318935, -0.10136446],[-0.3381198, 0.28183156, 0.46150166],[0.27193302, -0.16640529, -0.41912097],[0.25417566, 0.06102628, -0.52639526],[-0.14935666, -0.5422965, -0.03686011],[-0.3144787, -0.02274132, -0.23660958],[-0.3006308, -0.26082158, 0.16282296],[-0.35234135, 0.07790905, 0.10894704],[-0.30488306, -0.17647654, 0.30045635],[0.48848134, 0.53268725, -0.46586674]]");
            w.CopyTo(((DenseLayer)network.Layers[4]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.3337501,0.3300667,0.3361832],[0.3392599,0.3294313,0.3313088]]");
            TestLossAccuracy(network, X, Y, 1.10103356838226, 0.0);

            const double learningRate = 0.1;
            const int numEpochs = 10;
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.3923742,0.2287465,0.3788793],[0.3707,0.2305237,0.3987763]]");
            TestLossAccuracy(network, X, Y, 0.927446961402893, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestResNet_Shortcut_Different_Dimension_With_Conv_1x1_to_change_Dimension_NCHW_2_1_4_4(List<int> resourceIds)
        {
            var X = FromNumpyArray(X_2_1_4_4);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true); //left
            network.Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true, 1); //right (identity shortcut)
            network.AddLayer(3, 2).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)
                ;
            Tensor w = FromNumpyArray("[[[[-0.7714059]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            w = FromNumpyArray("[[[-1.0248963]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Weights);
            w = FromNumpyArray("[[[1.4231325]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[3]).Weights);
            w = FromNumpyArray("[[ 0.32856905,  0.1894297 ,  0.4066078 ],[-0.43956745, 0.52707547, -0.20674482],[0.31645727, -0.31735897, -0.38774815],[-0.15041429, 0.02662414, -0.3353554],[0.22785252, 0.538137, 0.03771406],[-0.35584196, -0.04203749, 0.46805507],[0.22338176, -0.34921265, 0.51070255],[-0.05367857, 0.31961358, -0.46928698],[-0.20997655, 0.03387326, 0.39165902],[-0.28344244, 0.3322929, 0.17337584],[0.01335454, 0.37127644, -0.52875155],[0.09800142, 0.21306825, 0.31867707],[0.35722166, 0.34757876, 0.0046258],[-0.12657085, 0.43093973, -0.27573565],[-0.41127366, 0.11429685, 0.06350583],[-0.09927812, -0.04027134, 0.16407043]]");
            w.CopyTo(((DenseLayer)network.Layers[5]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.4210416,0.2635172,0.3154411],[0.4182904,0.2633894,0.3183202]]");
            TestLossAccuracy(network, X, Y, 1.004860520362856, 0.5);

            const double learningRate = 0.1;
            const int numEpochs = 10;
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.5254775,0.02025996,0.4542625],[0.4535444,0.01977866,0.526677]]");
            TestLossAccuracy(network, X, Y, 0.642307877540588, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestL2Regularization_ConvolutionLayer_SGDVanilla_NCHW_2_1_4_4(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.12;
            const double lambdaL2Regularization = 0.05;
            var X = FromNumpyArray(X_2_1_4_4);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds)
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            Tensor w = FromNumpyArray("[[[[-0.7714059]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            w = FromNumpyArray("[[-0.10847557,  0.00658482,  0.41918087],[ 0.5224567 ,  0.42545766, -0.31801027],[-0.28710383, -0.31568986,  0.02822173],[-0.4120677 ,  0.21388823, -0.22343507],[-0.00887001,  0.42890936,  0.00528443],[ 0.14757729, -0.45275694, -0.36124444],[-0.5223615 ,  0.06962186,  0.44158655],[-0.44399977,  0.25540823, -0.35566014],[ 0.31000054,  0.03869426,  0.37737155],[-0.28234982,  0.43704945, -0.08071807],[-0.41145545,  0.41357315,  0.5401688 ],[-0.40983498, -0.47532582, -0.2516185 ],[-0.02894175,  0.07887733, -0.33317018],[ 0.07574445,  0.37989277, -0.47620153],[-0.5085196 ,  0.04452544, -0.4278263 ],[ 0.42463195,  0.26129186, -0.37209088]]");
            w.CopyTo(network.Layers[2].Weights);

            //predictions before training
            TestPredict(network, X, "[[0.553912222385406,0.135723426938057,0.310364335775375],[0.57789421081543,0.159836992621422,0.26226881146431]]");
            TestLossAccuracy(network, X, Y, null  /*0.9943205118179321*/, 0.5);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.56008917093277,0.0195358134806156,0.420375019311905],[0.398750722408295,0.0234715715050697,0.577777683734894]]");
            TestLossAccuracy(network, X, Y, null /*0.5985526442527771*/, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestL2Regularization_DenseLayer_SGDVanilla_NCHW_2_1_4_4(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.12;
            const double lambdaL2Regularization = 0.05;
            var X = FromNumpyArray(X_2_1_4_4);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds)
                .Input(X.Shape[1], X.Shape[2], X.Shape[3]).Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            Tensor w = FromNumpyArray("[[-0.3793878 ,  0.13005257, -0.48190022],[-0.5270703 , -0.5069973 , -0.45630288],[-0.08369148, -0.24146178, -0.09606424],[-0.0498544 , -0.4154459 , -0.3665961 ],[-0.3581952 , -0.3345901 ,  0.48476475],[ 0.320306  ,  0.301827  , -0.48490363],[ 0.33425486, -0.42483532,  0.20156533],[ 0.0346387 ,  0.34260863,  0.45479387],[-0.28320554,  0.27089173, -0.5511215 ],[-0.09140414, -0.2540371 , -0.38209555],[ 0.30901152, -0.22211927, -0.07776272],[-0.01273596, -0.43774882,  0.319129  ],[-0.26144847,  0.45303112, -0.5552845 ],[ 0.0012697 , -0.24624684, -0.01347905],[ 0.18339497, -0.46073103,  0.54499584],[-0.32917506,  0.03634387, -0.5220559 ]]");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.475850999355316,0.251384913921356,0.27276411652565],[0.506687998771667,0.285933136940002,0.207378879189491]]");
            TestLossAccuracy(network, X, Y, null  /*1.4464839696884155*/, 0.5);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.56395947933197,0.0651973560452461,0.370843172073364],[0.344296395778656,0.0696434527635574,0.586060106754303]]");
            TestLossAccuracy(network, X, Y, null /*0.8020790815353394*/, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestL2Regularization_DenseLayer_SGDMomentum_NCHW_2_1_4_4(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.12;
            const double lambdaL2Regularization = 0.5;
            var X = FromNumpyArray(X_2_1_4_4);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(0.9, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3]).Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            Tensor w = FromNumpyArray("[[-0.3793878 ,  0.13005257, -0.48190022],[-0.5270703 , -0.5069973 , -0.45630288],[-0.08369148, -0.24146178, -0.09606424],[-0.0498544 , -0.4154459 , -0.3665961 ],[-0.3581952 , -0.3345901 ,  0.48476475],[ 0.320306  ,  0.301827  , -0.48490363],[ 0.33425486, -0.42483532,  0.20156533],[ 0.0346387 ,  0.34260863,  0.45479387],[-0.28320554,  0.27089173, -0.5511215 ],[-0.09140414, -0.2540371 , -0.38209555],[ 0.30901152, -0.22211927, -0.07776272],[-0.01273596, -0.43774882,  0.319129  ],[-0.26144847,  0.45303112, -0.5552845 ],[ 0.0012697 , -0.24624684, -0.01347905],[ 0.18339497, -0.46073103,  0.54499584],[-0.32917506,  0.03634387, -0.5220559 ]]");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.475850999355316,0.251384913921356,0.27276411652565],[0.506687998771667,0.285933136940002,0.207378879189491]]");
            TestLossAccuracy(network, X, Y, null  /* 4.0434770584106445 */, 0.5);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.635447382926941,0.101806730031967,0.262745916843414],[0.348914474248886,0.0936608985066414,0.557424664497375]]");
            TestLossAccuracy(network, X, Y, null /*1.5627011060714722*/, 1.0);
        }

        //[Test, TestCaseSource(nameof(GetTestCases))]
        //public void TestL2Regularization_DenseLayer_Adam_NCHW_2_1_4_4(List<int> resourceIds)
        //{
        //    const int numEpochs = 10;
        //    const double learningRate = 0.12;
        //    const double lambdaL2Regularization = 0.05;
        //    var X = FromNumpyArray(X_2_1_4_4);
        //    var Y = FromNumpyArray(Y_2_3);
        //    var network = GetNetwork(LossFunctionEnum.CategoricalCrossentropy, resourceIds);
        //    network.Config.WithAdam(0.9, 0.999);
        //    network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
        //        .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
        //    Tensor w = FromNumpyArray("[[-0.3793878 ,  0.13005257, -0.48190022],[-0.5270703 , -0.5069973 , -0.45630288],[-0.08369148, -0.24146178, -0.09606424],[-0.0498544 , -0.4154459 , -0.3665961 ],[-0.3581952 , -0.3345901 ,  0.48476475],[ 0.320306  ,  0.301827  , -0.48490363],[ 0.33425486, -0.42483532,  0.20156533],[ 0.0346387 ,  0.34260863,  0.45479387],[-0.28320554,  0.27089173, -0.5511215 ],[-0.09140414, -0.2540371 , -0.38209555],[ 0.30901152, -0.22211927, -0.07776272],[-0.01273596, -0.43774882,  0.319129  ],[-0.26144847,  0.45303112, -0.5552845 ],[ 0.0012697 , -0.24624684, -0.01347905],[ 0.18339497, -0.46073103,  0.54499584],[-0.32917506,  0.03634387, -0.5220559 ]]");
        //    w.CopyTo(((DenseLayer)network.Layers[1]).Weights);
        //    //predictions before training
        //    TestPredict(network, X, "[[0.475850999355316,0.251384913921356,0.27276411652565],[0.506687998771667,0.285933136940002,0.207378879189491]]");
        //    TestLossAccuracy(network, X, Y, null  /*  1.4464839696884155 */, 0.5);

        //    var batchSize = X.Shape[0];
        //    TestNetwork.Fit(network, X, Y, learningRate* batchSize, numEpochs, batchSize);

        //    //predictions after training
        //    TestPredict(network, X, "[[0.894426345825195,0.00220351060852408,0.103370226919651],[0.0549939684569836,0.00156258791685104,0.943443357944489]]");
        //    TestLossAccuracy(network, X, Y, null /*0.4707931876182556*/, 1.0);
        //}

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestConcatenate_NCHW_1_1_1_1(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_1_1_1_1);
            var Y = FromNumpyArray(Y_1_2);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .ConcatenateLayer(1, 2)
                .Flatten().Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            var w = FromNumpyArray("[[[[-0.7714059]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            w = FromNumpyArray("[[[[-1.0248963]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Weights);
            w = FromNumpyArray("[[-0.2765898 ,  0.26417303],[0.8351141, -1.053136]]");
            w.CopyTo(((DenseLayer)network.Layers[5]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.8710213, 0.12897871]]");
            TestLossAccuracy(network, X, Y, 0.1380888819694519, 1.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[9.9985039e-01, 1.4956875e-04]]");
            TestLossAccuracy(network, X, Y, 0.0001495592441642657, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestMultiply_NCHW_1_1_1_1(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(@"numpy.array([[[[1]]]]], numpy.float)");
            var Y = FromNumpyArray(@"numpy.array([[1,0]], numpy.float)");
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true, 0)
                .MultiplyLayer(1, 2)
                .Flatten().Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            var w = FromNumpyArray("[[[[-0.7714059]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            w = FromNumpyArray("[[[[-1.0248963]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Weights);
            w = FromNumpyArray("[[-0.24186122,	-0.9861101]]");
            w.CopyTo(((DenseLayer)network.Layers[5]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.6430006, 0.35699943]]");
            TestLossAccuracy(network, X, Y, 0.44160962104797363, 1.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.80545473, 0.19454528]]");
            TestLossAccuracy(network, X, Y, 0.21634827554225922, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestMultiply_NCHW_1_2_1_1_same_dimension(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(@"[[[[1]],[[0]]]]");
            var Y = FromNumpyArray(@"[[1,0]]");
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true, 0)
                .MultiplyLayer(1, 2)
                .Flatten().Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            var w = FromNumpyArray("[[[[-0.54546636]],[[-0.12539268]]],[[[-0.72356474]],[[0.26959312]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            w = FromNumpyArray("[[[[-0.7247112]],[[-0.49400187]]],[[[-0.39867145]],[[0.04389346]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Weights);
            w = FromNumpyArray("[[-0.209458,-0.8539965],[-0.5895995,0.17340887]]");
            w.CopyTo(((DenseLayer)network.Layers[5]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.50867134,0.4913287]]");
            TestLossAccuracy(network, X, Y, 0.6759531497955322, 1.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.691145,0.30885503]]");
            TestLossAccuracy(network, X, Y, 0.3694056272506714, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestMultiply_NCHW_1_2_1_1_different_dimension(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(@"[[[[1]],[[0]]]]");
            var Y = FromNumpyArray(@"[[1,0]]");
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true, 0)
                .MultiplyLayer(1, 2)
                .Flatten().Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            var w = FromNumpyArray("[[[[-0.54546636, -0.12539268],[-0.72356474, 0.26959312]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            w = FromNumpyArray("[[[[-0.8368243],[-0.4603461]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Weights);
            w = FromNumpyArray("[[-0.209458  , -0.8539965 ],[-0.5895995, 0.17340887]]");
            w.CopyTo(((DenseLayer)network.Layers[5]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.45814985,0.54185015]]");
            TestLossAccuracy(network, X, Y, 0.7805589437484741, 0.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.67311347,0.3268865]]");
            TestLossAccuracy(network, X, Y, 0.3958413600921631, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void TestMultiply_NCHW_2_3_4_5_different_dimension(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_2_3_4_5);
            var Y = FromNumpyArray(Y_2_2);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true)
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true)
                .GlobalAvgPooling()
                .MultiplyLayer(1, 3)
                .Flatten().Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            FromConvNumpyArray("[[[[-0.4878799319267273, -0.6471760272979736], [-0.11215460300445557, 0.24113142490386963], [-0.5400518774986267, -0.8205036520957947]]]]").CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            FromConvNumpyArray("[[[[-0.7247111797332764, -0.3986714482307434], [-0.4940018653869629, 0.04389345645904541]]]]").CopyTo(((ConvolutionLayer)network.Layers[2]).Weights);
            FromNumpyArray("[[-0.029460519552230835, 0.1628669798374176], [-0.28001704812049866, -0.23855498433113098], [0.07715305685997009, 0.11627233028411865], [0.32925912737846375, 0.011087954044342041], [0.12424156069755554, -0.05900973081588745], [-0.2703372836112976, 0.12233385443687439], [-0.08240920305252075, 0.006095200777053833], [-0.023135006427764893, 0.08786126971244812], [-0.2075882852077484, -0.3384675085544586], [0.10181871056556702, -0.08105111122131348], [0.04287368059158325, -0.014433145523071289], [-0.050517499446868896, 0.19285127520561218], [0.16756221652030945, -0.06256869435310364], [-0.1878374218940735, -0.17477598786354065], [0.3118181526660919, 0.36103251576423645], [0.16790542006492615, 0.27620890736579895], [0.21295377612113953, -0.15440134704113007], [0.03934970498085022, -0.35186851024627686], [-0.19449061155319214, -0.2855254113674164], [-0.08950188755989075, 0.2891680896282196], [-0.37375181913375854, 0.18617329001426697], [0.07124421000480652, 0.28268447518348694], [0.041756272315979004, 0.13584479689598083], [0.12497344613075256, 0.151188462972641], [0.3146173655986786, -0.22298070788383484], [-0.22048203647136688, -0.30460700392723083], [0.12072917819023132, -0.2646358907222748], [-0.15740737318992615, 0.17554828524589539], [0.13976749777793884, -0.357845664024353], [-0.365357369184494, -0.15716126561164856], [0.14519938826560974, 0.22951403260231018], [0.03488221764564514, 0.1870688498020172], [0.28289076685905457, 0.14199396967887878], [0.31583401560783386, 0.08595579862594604], [0.005727171897888184, 0.2800586521625519], [0.013508498668670654, 0.3192369043827057], [-0.14768590033054352, -0.05077126622200012], [-0.28260645270347595, -0.3034713864326477], [-0.05905658006668091, -0.3151003122329712], [-0.12471392750740051, -0.2689373791217804]]").CopyTo(((DenseLayer)network.Layers[6]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.48352786898612976, 0.5164721012115479], [0.510468602180481, 0.48953139781951904]]");
            TestLossAccuracy(network, X, Y, 0.6995362043380737, 0.5);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.6233826875686646, 0.37661728262901306], [0.6570475101470947, 0.3429524898529053]]");
            TestLossAccuracy(network, X, Y, 0.4462968111038208, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void Test_DepthwiseConvolution(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray("[[[[0.0,0.1],[0.2,0.3]],[[0.4,0.5],[0.6,0.7]],[[0.8,0.9],[0.95,1.0]]]]");
            var Y = FromNumpyArray(@"[[1,0]]");
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .DepthwiseConvolution(3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 1, lambdaL2Regularization, true)
                .Flatten().Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            var w = FromConvNumpyArray("[[[[-1.8182212e-01],[-2.4118826e-01],[-4.1797549e-02]],[[ 8.9864373e-02],[-2.0126545e-01],[-3.0578366e-01]],[[ 3.1133693e-01],[ 2.8887171e-01],[-1.8024862e-01]]],[[[ 2.5361568e-01],[ 2.5364757e-04],[ 3.7667376e-01]],[[ 2.5444084e-01],[-1.6983339e-01],[ 3.4809512e-01]],[[ 1.6018957e-01],[-1.4399531e-01],[ 4.0002191e-01]]],[[[ 6.4367443e-02],[ 3.8731194e-01],[-3.4315154e-01]],[[-1.1633310e-01],[ 1.9787598e-01],[ 3.6543274e-01]],[[-1.5561658e-01],[ 2.6152426e-01],[-1.7792954e-01]]]]");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            w = FromNumpyArray("[[-0.2895949 , -0.1177094 ],[ 0.23789662,  0.3767377 ],[ 0.41499782,  0.2744642 ], [ 0.00143611,  0.4848951 ], [-0.49549383, -0.20193446], [-0.6493831 ,  0.06595588], [-0.28986678,  0.05255955], [-0.27410832, -0.3834508 ], [ 0.05772114, -0.02405858], [ 0.18792003, -0.33073252], [-0.2710427 ,  0.09345931], [ 0.12703902, -0.14722306]]");
            w.CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.529547, 0.470453]]");
            TestLossAccuracy(network, X, Y, 0.635733425617218, 1.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.8828165, 0.11718353]]");
            TestLossAccuracy(network, X, Y, 0.12463792413473129, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void Test_Conv1D_Causal(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_3_4_5);
            var Y = FromNumpyArray(Y_3_3);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], -1)
                .Conv1D(2, 3, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Conv1D(2, 3, 2, ConvolutionLayer.PADDING_TYPE.CAUSAL, lambdaL2Regularization, true)
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true)
                .Flatten().Dense(Y.Shape[1], lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            FromConvNumpyArray("[[[0.09934896230697632, -0.11215364933013916], [0.3982505798339844, 0.342079758644104], [-0.06867659091949463, -0.46536481380462646], [0.2547714114189148, -0.08702009916305542]], [[-0.5021747350692749, -0.1221388578414917], [-0.3608691096305847, 0.3861338496208191], [0.10946327447891235, -0.052802085876464844], [-0.016413629055023193, 0.3857215642929077]], [[0.4184006452560425, -0.2657143771648407], [0.296006977558136, -0.28657031059265137], [-0.016508877277374268, -0.2890245020389557], [0.1388271450996399, 0.02789127826690674]]]").CopyTo(network.Layers[1].Weights);
            FromConvNumpyArray("[[[0.39741700887680054, 0.5679424405097961], [0.103904128074646, 0.46203213930130005]], [[0.5664966702461243, -0.5104600191116333], [-0.4302336871623993, 0.2359222173690796]], [[0.1441558599472046, -0.3472554683685303], [0.3229832053184509, -0.13790547847747803]]]").CopyTo(network.Layers[2].Weights);
            FromConvNumpyArray("[[[-1.0770841836929321, 0.557166576385498], [0.405431866645813, -0.2015085220336914]]]").CopyTo(network.Layers[3].Weights);
            FromNumpyArray("[[0.38363194465637207, 0.2582963705062866, 0.15701913833618164], [0.5796942710876465, -0.42992860078811646, 0.28377270698547363], [-0.34947991371154785, 0.8033483028411865, -0.22690773010253906], [0.8054455518722534, 0.22870910167694092, -0.36302077770233154]]").CopyTo(network.Layers[5].Weights);

            //predictions before training
            TestPredict(network, X, "[[0.2877206802368164, 0.4445759654045105, 0.2677032947540283], [0.3576244115829468, 0.2031501680612564, 0.4392254054546356], [0.273379385471344, 0.4651012122631073, 0.2615194320678711]]");
            TestLossAccuracy(network, X, Y, 0.9446693062782288, 2.0/3);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.36478546261787415, 0.2846057415008545, 0.35060879588127136], [0.2515701353549957, 0.058832425624132156, 0.6895974278450012], [0.20741572976112366, 0.6975075602531433, 0.09507670253515244]]");
            TestLossAccuracy(network, X, Y, 0.5801116824150085, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void Test_Convolution_With_Asymmetric_Padding(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_2_3_4_5);
            var Y = FromNumpyArray(Y_2_2);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 3, 2, ConvolutionLayer.PADDING_TYPE.SAME, 0.00, false)
                .Flatten().Dense(Y.Shape[1], 0.00, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            FromConvNumpyArray("[[[[-1.8182212e-01],[-2.4118826e-01],[-4.1797549e-02]],[[ 8.9864373e-02],[-2.0126545e-01],[-3.0578366e-01]],[[ 3.1133693e-01],[ 2.8887171e-01],[-1.8024862e-01]]],[[[ 2.5361568e-01],[ 2.5364757e-04],[ 3.7667376e-01]],[[ 2.5444084e-01],[-1.6983339e-01],[ 3.4809512e-01]],[[ 1.6018957e-01],[-1.4399531e-01],[ 4.0002191e-01]]],[[[ 6.4367443e-02],[ 3.8731194e-01],[-3.4315154e-01]],[[-1.1633310e-01],[ 1.9787598e-01],[ 3.6543274e-01]],[[-1.5561658e-01],[ 2.6152426e-01],[-1.7792954e-01]]]]").CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            FromNumpyArray("[[0.28597003,  0.4547698],[-0.4759524,   0.7416852],[-0.85341465, -0.40610817],[-0.14299387, -0.7595345],[0.08673978, -0.36584526],[0.855876, -0.65572554]]").CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.95572853, 0.04427144],[0.6475099, 0.35249016]]");
            TestLossAccuracy(network, X, Y, 0.23995131254196167, 1.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.98308647, 0.01691353],[0.961277, 0.03872294]]");
            TestLossAccuracy(network, X, Y, 0.028275400400161743, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void Test_Convolution_With_Asymmetric_Padding_V2(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_2_3_4_5);
            var Y = FromNumpyArray(Y_2_2);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Convolution(1, 3, 2, ConvolutionLayer.PADDING_TYPE.SAME, 0.00, false)
                .Flatten().Dense(Y.Shape[1], 0.00, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            FromConvNumpyArray("[[[[ 0.34814847],[-0.12127715],[ 0.22354162]],[[-0.06822091],[ 0.14409536],[ 0.1851415 ]],[[ 0.34887362],[-0.20124988],[-0.40070006]]],[[[0.00579837],[ 0.10344726],[-0.2527819 ]],[[ 0.39281756],[-0.04241154],[-0.27652574]],[[ 0.00179639],[-0.14511377],[-0.05352649]]],[[[ 0.2575059 ],[ 0.1235916 ],[-0.26898897]],[[ 0.01372808],[-0.22314253],[ 0.3652693 ]],[[ 0.4061154 ],[0.04825488],[ 0.4062844 ]]]]").CopyTo(((ConvolutionLayer)network.Layers[2]).Weights);
            FromNumpyArray("[[ 0.1670317  , 0.6739995 ], [ 0.02281129 , 0.36221045], [ 0.7731834  , 0.41565734], [-0.2101801  , 0.21554786], [-0.1369173  , 0.44515973], [-0.18519688 , 0.5447338 ]]").CopyTo(((DenseLayer)network.Layers[4]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.69206095, 0.30793908], [0.45456633, 0.54543364]]");
            TestLossAccuracy(network, X, Y, 0.5782463550567627, 0.5);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.97237974, 0.02762033], [0.9205178, 0.0794822 ]]");
            TestLossAccuracy(network, X, Y, 0.05541396513581276, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void Test_UpSampling2D(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_2_3_4_5);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(4, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true)
                .UpSampling2D(3, 2, UpSampling2DLayer.InterpolationEnum.Nearest)
                .Convolution(1, 3, 2, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            FromConvNumpyArray("[[[[-0.41233378648757935, -0.5469635725021362, -0.09478795528411865, 0.20379328727722168], [-0.45642712712287903, -0.6934521198272705, 0.7060458660125732, 0.6550993919372559], [-0.40876543521881104, 0.5751461982727051, 0.0005752444267272949, 0.8542157411575317]]]]").CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            FromConvNumpyArray("[[[[-0.1615283042192459], [-0.0656551718711853], [0.1326923966407776], [0.21013426780700684]], [[0.23147475719451904], [0.15308880805969238], [0.0008010268211364746], [0.2704615592956543]], [[-0.2763732671737671], [-0.11263367533683777], [-0.3622085750102997], [0.03678843379020691]]], [[[-0.1616799682378769], [0.029316306114196777], [-0.15289030969142914], [-0.21387864649295807]], [[0.032195329666137695], [-0.013419240713119507], [0.10481679439544678], [-0.18447379767894745]], [[-0.15118040144443512], [0.052129119634628296], [0.07085898518562317], [-0.08211708068847656]]], [[[-0.02411407232284546], [0.17931300401687622], [-0.2963199317455292], [-0.019487440586090088]], [[-0.2584547698497772], [0.23713970184326172], [-0.351848304271698], [0.3424469232559204]], [[0.22793227434158325], [0.13822901248931885], [-0.12481275200843811], [-0.32772859930992126]]]]").CopyTo(((ConvolutionLayer)network.Layers[3]).Weights);
            FromNumpyArray("[[0.07366013526916504, 0.3170207142829895, -0.1550242304801941], [0.420951247215271, -0.4191424548625946, 0.3381590247154236], [0.11008310317993164, 0.0986890196800232, 0.31357908248901367], [0.41440945863723755, 0.30317842960357666, 0.3536931872367859], [-0.010290741920471191, -0.21904385089874268, -0.020769357681274414], [-0.2869524359703064, -0.3439455032348633, 0.2285328507423401], [-0.022606879472732544, -0.1754196584224701, -0.12093043327331543], [-0.19505150616168976, 0.32367968559265137, 0.27787232398986816], [0.1375676393508911, -0.1417226493358612, 0.33683180809020996], [-0.36117273569107056, 0.001855224370956421, 0.24049299955368042], [-0.02008679509162903, 0.22243833541870117, -0.27483871579170227], [-0.20811842381954193, -0.17607355117797852, -0.1847764253616333], [-0.41185829043388367, 0.14473176002502441, 0.10743755102157593], [0.3232056498527527, -0.2687329947948456, 0.041926443576812744], [-0.07551324367523193, 0.23673099279403687, -0.4212562143802643], [-0.32285287976264954, -0.20976179838180542, 0.35986894369125366], [-0.42236655950546265, 0.06221747398376465, 0.19280701875686646], [-0.1036037802696228, 0.22280341386795044, 0.2663360834121704], [-0.278300404548645, 0.3701552152633667, -0.3987610638141632], [-0.2845539450645447, 0.08112376928329468, -0.06442150473594666], [0.13321810960769653, 0.39671868085861206, -0.34261322021484375], [-0.23947212100028992, -0.10445082187652588, -0.36301395297050476], [0.20646917819976807, 0.11567127704620361, 0.15597444772720337], [-0.3057088851928711, 0.39422833919525146, -0.23814217746257782], [0.1633470058441162, 0.12872058153152466, 0.2478216290473938], [-0.3868710696697235, -0.335817813873291, 0.42601829767227173], [-0.3151834011077881, 0.30162113904953003, -0.06157597899436951], [-0.19710223376750946, 0.0573333203792572, 0.2074006199836731], [-0.28093406558036804, 0.2030026912689209, 0.4050601124763489], [0.29869991540908813, -0.31979823112487793, 0.41144388914108276]]").CopyTo(((DenseLayer)network.Layers[4]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.06410794705152512, 0.28633567690849304, 0.6495563387870789], [0.4432253837585449, 0.45047569274902344, 0.10629893094301224]]");
            TestLossAccuracy(network, X, Y, 2.4943435192108154, 0.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.994227409362793, 0.0028337084222584963, 0.002938868710771203], [0.003187840338796377, 2.7520829462446272e-05, 0.9967846870422363]]");
            TestLossAccuracy(network, X, Y, 0.004504904616624117, 1.0);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void Test_ZeroPadding2D(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_2_3_4_5);
            var Y = FromNumpyArray(Y_2_3);
            var network = GetNetwork(EvaluationMetricEnum.CategoricalCrossentropy, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(4, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true)
                .ZeroPadding2D(1, 2, 3, 4)
                .Convolution(1, 3, 2, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true).Dense(Y.Shape[1], 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            FromConvNumpyArray("[[[[-0.41233378648757935, -0.5469635725021362, -0.09478795528411865, 0.20379328727722168], [-0.45642712712287903, -0.6934521198272705, 0.7060458660125732, 0.6550993919372559], [-0.40876543521881104, 0.5751461982727051, 0.0005752444267272949, 0.8542157411575317]]]]").CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            FromConvNumpyArray("[[[[0.07042673230171204], [0.284183144569397], [0.009618103504180908], [0.1527213454246521]], [[0.3260027766227722], [0.17525649070739746], [-0.08861970901489258], [0.09088295698165894]], [[-0.05772939324378967], [0.18769580125808716], [-0.07808583974838257], [0.22967994213104248]]], [[[-0.03302586078643799], [-0.2509212791919708], [-0.1510678380727768], [0.09804627299308777]], [[-0.14208482205867767], [0.15808266401290894], [0.20872968435287476], [-0.264028936624527]], [[-0.34229689836502075], [0.2561035752296448], [0.03571978211402893], [0.24708086252212524]]], [[[-0.07826903462409973], [0.28786909580230713], [0.2017209529876709], [0.17617851495742798]], [[0.15275120735168457], [0.1589658260345459], [-0.2798265814781189], [0.011122971773147583]], [[-0.20089077949523926], [0.19293898344039917], [0.1550755500793457], [0.26180779933929443]]]]").CopyTo(((ConvolutionLayer)network.Layers[3]).Weights);
            FromNumpyArray("[[-0.10645946860313416, 0.10168024897575378, 0.3214356005191803], [-0.4053522050380707, 0.1360771358013153, -0.10558605194091797], [0.22095760703086853, 0.02734297513961792, -0.2764899730682373], [-0.027258336544036865, 0.08908268809318542, -0.4332699477672577], [0.3699513375759125, -0.4085657000541687, -0.43480184674263], [0.3350076377391815, 0.047171443700790405, -0.457603394985199], [-0.06026741862297058, 0.22628697752952576, -0.3810364007949829], [0.03480765223503113, -0.4253525137901306, -0.07680919766426086], [0.4418511688709259, -0.11238834261894226, -0.2000998556613922], [-0.330613374710083, 0.29349616169929504, -0.1369912028312683], [0.12755998969078064, -0.11974358558654785, 0.422267347574234], [-0.1738777756690979, 0.24821648001670837, -0.3069990873336792], [-0.3616308569908142, 0.44209930300712585, -0.4273287057876587], [0.23651161789894104, 0.015389323234558105, -0.20962989330291748], [-0.4005410671234131, -0.2227368801832199, 0.3084500730037689], [0.26213350892066956, 0.019401252269744873, -0.15848103165626526], [-0.26640957593917847, -0.30630677938461304, -0.15309825539588928], [-0.0580502450466156, -0.3467845916748047, -0.08887636661529541], [-0.1492699682712555, 0.4617249071598053, 0.2631932199001312], [-0.14096277952194214, -0.22659775614738464, 0.18531039357185364], [-0.3825145661830902, -0.11876800656318665, -0.05306357145309448], [0.23907557129859924, 0.4140254557132721, 0.12409594655036926], [-0.36942005157470703, 0.09518781304359436, 0.23360726237297058], [0.36897769570350647, 0.379914253950119, -0.26894521713256836]]").CopyTo(((DenseLayer)network.Layers[4]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.1705085188150406, 0.26617881655693054, 0.5633126497268677], [0.1805817037820816, 0.6831672191619873, 0.13625110685825348]]");
            TestLossAccuracy(network, X, Y, 1.881112813949585, 0.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.8653897643089294, 0.040067702531814575, 0.09454257786273956], [0.054335951805114746, 0.10663946717977524, 0.8390246033668518]]");
            TestLossAccuracy(network, X, Y, 0.160045325756073, 1.0);
        }
        
        [Test, TestCaseSource(nameof(GetTestCases))]
        public void Test_HuberLoss(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double momentum = 0.9;
            var X = FromNumpyArray(@"numpy.array([[0,1,2],[3,4,5]], numpy.float)");
            var Y = FromNumpyArray(@"numpy.array([[0],[5]], numpy.float)");

            var network = GetNetwork(EvaluationMetricEnum.Huber, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network
                .Input(X.Shape[1], 1, -1)
                .Dense(3, 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(1, 0.0, false);

            FromNumpyArray("[[0.17207741737365723, -0.19425582885742188, 0.6897902488708496], [0.5924994945526123, -0.11895132064819336, -0.8060355186462402], [0.44127702713012695, -0.15072321891784668, -0.8697922229766846]]")
                .CopyTo(((DenseLayer)network.Layers[1]).Weights);
            FromNumpyArray("[[0.6883463859558105], [0.9837051630020142], [0.17996716499328613]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[1.0153478384017944], [3.5054831504821777]]", 1e-3);
            TestLossAccuracy(network, X, Y, 0.7549323439598083, null, 1e-3);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[-0.5761867165565491], [7.070157527923584]]", 1e-3);
            TestLossAccuracy(network, X, Y, 0.8678827881813049, null, 1e-3);
        }

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void Test_MseLoss(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_2_3_4_5);
            var Y = FromNumpyArray(Y_2_3);

            var network = GetNetwork(EvaluationMetricEnum.Mse, resourceIds);
            network.Sample.WithSGD(momentum, false);
            network.Sample.Metrics = new List<EvaluationMetricEnum> { network.Sample.LossFunction, EvaluationMetricEnum.Mae};
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Dense(3, 0.0, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(3, 0.0, false);

            FromNumpyArray("[[0.14902335405349731, -0.16823047399520874, 0.5973758101463318], [0.513119637966156, -0.10301488637924194, -0.6980472207069397], [0.3821571469306946, -0.13053011894226074, -0.7532621622085571], [-0.18320828676223755, -0.5413036346435547, 0.579200804233551], [0.16419488191604614, -0.07920318841934204, -0.024620473384857178]]")
                .CopyTo(network.Layers[1].Weights);
            FromNumpyArray("[[0.22044727206230164, 0.3150377571582794, 0.05763563513755798], [0.2562893331050873, 0.31423577666282654, -0.2831522822380066], [-0.23865070939064026, 0.13086608052253723, 0.07996329665184021], [-0.19262267649173737, 0.17915889620780945, -0.07649621367454529], [0.17553207278251648, -0.006345480680465698, -0.2592658996582031], [-0.3532370626926422, 0.13268747925758362, 0.14748194813728333], [0.26681867241859436, 0.3054209053516388, -0.3524059057235718], [-0.0743943452835083, 0.26640889048576355, 0.07575491070747375], [-0.08206719160079956, 0.1284121572971344, 0.07956790924072266], [0.2841203510761261, 0.012592524290084839, 0.15496674180030823], [-0.17980638146400452, -0.2484310269355774, 0.04503124952316284], [-0.2442535012960434, 0.24186745285987854, 0.12160143256187439], [-0.27119147777557373, -0.3446856439113617, -0.30974677205085754], [-0.3750991225242615, 0.23870810866355896, -0.28597140312194824], [0.15718379616737366, -0.0744141936302185, 0.09016254544258118], [-0.2230645716190338, 0.31610485911369324, -0.18952983617782593], [-0.37350326776504517, -0.3442955017089844, 0.3457939922809601], [0.209515780210495, 0.05385535955429077, 0.107828289270401], [0.27271541953086853, 0.13648775219917297, 0.30335161089897156], [0.19686904549598694, -0.3310122787952423, 0.27363476157188416], [-0.010377317667007446, 0.33995702862739563, 0.004759669303894043], [0.0042154788970947266, -0.11687412858009338, -0.06438267230987549], [0.371217280626297, 0.04651784896850586, -0.1674891859292984], [0.3474888503551483, 0.28037092089653015, -0.04222455620765686], [0.0916936993598938, 0.2884680926799774, 0.0825285017490387], [0.30377236008644104, -0.11384806036949158, -0.3356134295463562], [-0.37017637491226196, -0.10759112238883972, 0.0320684015750885], [-0.029354870319366455, 0.2205376923084259, 0.10602417588233948], [-0.049274712800979614, 0.3876277506351471, -0.28544583916664124], [-0.37114545702934265, -0.233595073223114, -0.23805370926856995], [0.045278966426849365, 0.16116967797279358, -0.01359209418296814], [0.28720858693122864, -0.10025259852409363, -0.09117457270622253], [-0.22608521580696106, -0.06644889712333679, 0.20117756724357605], [-0.0758948028087616, -0.3437873423099518, 0.3798452317714691], [-0.15284530818462372, -0.3742479383945465, 0.13099047541618347], [-0.14322248101234436, -0.18771331012248993, 0.3071592152118683]]")
                .CopyTo(network.Layers[3].Weights);

            //predictions before training
            TestPredict(network, X, "[[0.104760080575943, 0.6928589344024658, -0.7177746891975403], [-1.4105323553085327, 0.6341972947120667, 0.09186349809169769]]", 1e-3);
            //TestLossAccuracy(network, X, Y, 0.8355380296707153, null, 1e-3);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[1.02695631980896, 0.20715948939323425, 0.22072364389896393], [0.012165275402367115, 0.01876860111951828, 0.38967499136924744]]", 1e-3);
            //TestLossAccuracy(network, X, Y, 0.8678827881813049, null, 1e-3);
        }

        #region Recurrent Layer
        private const string X_RNN_4_2_1 = @"numpy.array([ [[1.0],[2.0]] , [[2.0],[3.0]] , [[3.0],[4.0]]  , [[4.0],[5.0]] ], numpy.float)";
        private const string Y_RNN_4_2_1 = @"numpy.array([ [[2.0],[3.0]] , [[3.0],[4.0]] , [[4.0],[5.0]] , [[5.0],[6.0]] ], numpy.float)";
        private const string Y_RNN_4_1_1 = @"numpy.array([ [[3.0]] , [[4.0]] , [[5.0]] , [[6.0]] ], numpy.float)";

        [Test, TestCaseSource(nameof(GetTestCases))]
        public void Test_SimpleRNN_returnSequence_false(List<int> resourceIds)
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            var X = FromNumpyArray(X_RNN_4_2_1);
            var Y = FromNumpyArray(Y_RNN_4_1_1);

            int batchSize = X.Shape[0];
            int timeSteps = X.Shape[1];
            int inputSize = X.Shape[2];

            var network = GetNetwork(EvaluationMetricEnum.Huber, resourceIds);
            network.Sample.WithSGD(0.9, false);
            network
                .Input(timeSteps, inputSize, -1)
                .SimpleRNN(3, false, false)
                .Dense(1, 0.0, true);

            network.Layers[1].Weights.ZeroMemory();
            FromNumpyArray("[[-1.0770841836929321], [0.557166576385498], [0.405431866645813]]")
                .CopyTo(network.Layers[2].Weights);

            //predictions before training
            TestPredict(network, X, "[[0],[0],[0],[0]]", 1e-4);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            //predictions after training
            TestPredict(network, X, "[[5.32697773],[5.32737398],[5.3273859],[5.3273859]]", 1e-4);
            network.Dispose();
        }

        [TestCase(true, false, "[[-1.243709683418274], [0.6433604955673218]]", "[[[0],[0]],[[0],[0]],[[0],[0]],[[0],[0]]]", "[[[5.14900208],[5.16034889]],[[5.16030025],[5.16046333]],[[5.16046238],[5.16046524]],[[5.16046524],[5.16046524]]]")]
        [TestCase(true, true, "[[0.15931272506713867], [-0.179845929145813], [0.6386216878890991], [0.5485479831695557]]", "[[[0],[0]],[[0],[0]],[[0],[0]],[[0],[0]]]", "[[[4.57469749],[4.6931181]],[[4.69146442],[4.71216202]],[[4.71178246],[4.71545982]],[[4.71539021],[4.71604347]]]")]
        [TestCase(false, false, "[[0.18850135803222656], [-0.2127966284751892]]", "[[0],[0],[0],[0]]", "[[6.66841745],[6.68074799],[6.68167067],[6.68174171]]")]
        [TestCase(false, true, "[[0.18850135803222656], [-0.2127966284751892], [0.7556273937225342], [0.6490507125854492]]", "[[0],[0],[0],[0]]", "[[5.12246323],[5.13376093],[5.13461781],[5.13470268]]")]
        public void Test_SimpleRNN(bool returnSequences, bool isBidirectional, string denseWeight, string expectedPredictionBeforeTraining, string expectedPredictionAfterTraining)
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            var X = FromNumpyArray(X_RNN_4_2_1);
            var Y = FromNumpyArray(returnSequences ? Y_RNN_4_2_1 : Y_RNN_4_1_1);

            int batchSize = X.Shape[0];
            int timeSteps = X.Shape[1];
            int inputSize = X.Shape[2];

            var network = GetNetwork(EvaluationMetricEnum.Huber, new List<int> {0});
            network.Sample.WithSGD(0.9, false);
            network
                .Input(timeSteps, inputSize, -1)
                .SimpleRNN(2, returnSequences, isBidirectional)
                .Dense(1, 0.0, true);

            network.Layers[1].Weights.ZeroMemory();
            FromNumpyArray(denseWeight).CopyTo(network.Layers[2].Weights);

            //predictions before training
            TestPredict(network, X, expectedPredictionBeforeTraining, 1e-4);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            //predictions after training
            TestPredict(network, X, expectedPredictionAfterTraining, 1e-4);
            network.Dispose();
        }

        [TestCase(true, false, "[[-1.243709683418274], [0.6433604955673218]]", "[[[0],[0]],[[0],[0]],[[0],[0]],[[0],[0]]]", "[[[4.6592536],[5.60682297]],[[5.1264925],[5.90583038]],[[5.36573982],[6.05742359]],[[5.49268246],[6.14186144]]]")]
        [TestCase(true, true, "[[0.18850135803222656], [-0.2127966284751892], [0.7556273937225342], [0.6490507125854492]]", "[[[0],[0]],[[0],[0]],[[0],[0]],[[0],[0]]]", "[[[4.9929328],[5.11667824]],[[5.49777794],[5.44859505]],[[5.80846691],[5.65499783]],[[6.00010109],[5.78587103]]]")]
        [TestCase(false, false, "[[0.18850135803222656], [-0.2127966284751892]]", "[[0],[0],[0],[0]]", "[[4.71483612],[4.91614676],[5.0574832],[5.16643476]]")]
        [TestCase(false, true, "[[0.18850135803222656], [-0.2127966284751892], [0.7556273937225342], [0.6490507125854492]]", "[[0],[0],[0],[0]]", "[[5.67572021],[6.24725914],[6.59858942],[6.81990719]]")]
        public void Test_LSTM(bool returnSequences, bool isBidirectional, string denseWeight, string expectedPredictionBeforeTraining, string expectedPredictionAfterTraining)
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            var X = FromNumpyArray(X_RNN_4_2_1);
            var Y = FromNumpyArray(returnSequences ? Y_RNN_4_2_1:Y_RNN_4_1_1);

            int batchSize = X.Shape[0];
            int timeSteps = X.Shape[1];
            int inputSize = X.Shape[2];

            var network = GetNetwork(EvaluationMetricEnum.Huber, new List<int> {0});
            network.Sample.WithSGD(0.9, false);
            network
                .Input(timeSteps, inputSize, -1)
                .LSTM(2, returnSequences, isBidirectional, 1, 0.0, false)
                .Dense(1, 0.0, true);

            network.Layers[1].Weights.ZeroMemory();
            FromNumpyArray(denseWeight).CopyTo(network.Layers[2].Weights);

            //predictions before training
            TestPredict(network, X, expectedPredictionBeforeTraining, 1e-4);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            //predictions after training
            TestPredict(network, X, expectedPredictionAfterTraining, 1e-4);

            network.Dispose();
        }


        [TestCase(true, false, "[[-1.243709683418274], [0.6433604955673218]]", "[[[0],[0]],[[0],[0]],[[0],[0]],[[0],[0]]]", "[[[4.97974014],[5.57354736]],[[5.3242569],[5.62948704]],[[5.48532438],[5.6469841]],[[5.56586361],[5.65275764]]]")]
        [TestCase(true, true, "[[0.18850135803222656], [-0.2127966284751892], [0.7556273937225342], [0.6490507125854492]]", "[[[0],[0]],[[0],[0]],[[0],[0]],[[0],[0]]]", "[[[5.11490726],[5.21613836]],[[5.28695011],[5.44156599]],[[5.33742332],[5.54083538]],[[5.3631835],[5.5902195]]]")]
        [TestCase(false, false, "[[0.18850135803222656], [-0.2127966284751892]]", "[[0],[0],[0],[0]]", "[[6.24894905],[6.42779493],[6.49549866],[6.5253849]]")]
        [TestCase(false, true, "[[0.18850135803222656], [-0.2127966284751892], [0.7556273937225342], [0.6490507125854492]]", "[[0],[0],[0],[0]]", "[[5.61915636],[5.76153946],[5.80879307],[5.83384371]]")]
        public void Test_GRU(bool returnSequences, bool isBidirectional, string denseWeight, string expectedPredictionBeforeTraining, string expectedPredictionAfterTraining)
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            var X = FromNumpyArray(X_RNN_4_2_1);
            var Y = FromNumpyArray(returnSequences ? Y_RNN_4_2_1 : Y_RNN_4_1_1);

            int batchSize = X.Shape[0];
            int timeSteps = X.Shape[1];
            int inputSize = X.Shape[2];

            var network = GetNetwork(EvaluationMetricEnum.Huber, new List<int> {0});
            network.Sample.WithSGD(0.9, false);
            network
                .Input(timeSteps, inputSize, -1)
                .GRU(2, returnSequences, isBidirectional, 1, 0.0, false)
                .Dense(1, 0.0, true);

            network.Layers[1].Weights.ZeroMemory();
            FromNumpyArray(denseWeight).CopyTo(network.Layers[2].Weights);

            //predictions before training
            TestPredict(network, X, expectedPredictionBeforeTraining, 1e-4);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            //predictions after training
            TestPredict(network, X, expectedPredictionAfterTraining, 1e-4);

            network.Dispose();
        }
        #endregion

        private static Network GetNetwork(EvaluationMetricEnum evaluationMetric, List<int> resourceIds)
        {
            var evaluationMetrics = new List<EvaluationMetricEnum>{ evaluationMetric};
            if (evaluationMetric is EvaluationMetricEnum.BinaryCrossentropy or EvaluationMetricEnum.CategoricalCrossentropy)
            {
                evaluationMetrics.Add(EvaluationMetricEnum.Accuracy);
            }
            return TestNetwork.NewForTests(
                new NetworkSample{ 
                    LossFunction = evaluationMetric, 
                    Metrics = evaluationMetrics,
                    RandomizeOrder = false, 
                    ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM, CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow, 
                    ResourceIds = resourceIds.ToList()
                },
                NetworkSample.DefaultWorkingDirectory,
                "ModelName"
                );
        }
        private static void TestPredict(Network network, Tensor X, string expectedPredictionAsString, double epsilon = 1e-6)
        {
            var observedPrediction = network.Predict(X, false);
            var expectedPrediction = FromNumpyArray(expectedPredictionAsString);
            Assert.IsTrue(TensorExtensions.SameFloatContent(observedPrediction, expectedPrediction, epsilon), "expecting: " + Environment.NewLine + expectedPrediction.ToNumpy()+Environment.NewLine+ " but was:" + Environment.NewLine + observedPrediction.ToNumpy());
        }
        private static void TestLossAccuracy(Network network, CpuTensor<float> X, CpuTensor<float> Y_expected, double? expectedLoss, double? expectedAccuracy, double epsilon = 1e-5)
        {
            var batchSize = X.Shape[0];
            using var dataSet = new InMemoryDataSet(X, Y_expected);
            var observedMetrics = network.ComputeMetricsForTestDataSet(batchSize, dataSet);
            if (expectedLoss.HasValue)
            { 
                Assert.AreEqual(expectedLoss.Value, observedMetrics[network.Sample.LossFunction], epsilon, "expected loss: " + expectedLoss.Value + " but was: " + observedMetrics[network.Sample.LossFunction]);
            }
            if (expectedAccuracy.HasValue)
            {
                Assert.AreEqual(expectedAccuracy.Value, observedMetrics[EvaluationMetricEnum.Accuracy], epsilon, "expected accuracy: " + expectedAccuracy.Value + " but was: " + observedMetrics[EvaluationMetricEnum.Accuracy]);
            }
        }
    }
}

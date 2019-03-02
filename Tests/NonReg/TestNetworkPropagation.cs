using System;
using System.Diagnostics;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNetTests.Data;
// ReSharper disable UnusedMember.Local

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestNetworkPropagation
    {
        public const string X_1_1_1_1 = @"numpy.array([[[[0.7262433]]]], numpy.float)";
        public const string Y_1_1_1_1 = @"numpy.array([[1,0]], numpy.float)";
        public const string X_2_1_4_4 = @"numpy.array([[[[0.7262433,0.8173254,0.7680227,0.5581612],[0.2060332,0.5588848,0.9060271,0.4421779],[0.9775497,0.2737045,0.2919063,0.4673147],[0.6326591,0.4695119,0.9821513,0.03036699]]],[[[0.8623701,0.9953471,0.6771811,0.3145918],[0.8169079,0.8480518,0.9919022,0.0326252],[0.699942,0.5262842,0.9340187,0.6876203],[0.5468155,0.08110995,0.1871246,0.4533272]]]], numpy.float)";
        public const string Y_2_1_4_4 = @"numpy.array([[1,0,0], [0,0,1]], numpy.float)";
        public const string X_1_1_4_4 = @"numpy.array([[[[0.7262433,0.8173254,0.7680227,0.5581612],[0.2060332,0.5588848,0.9060271,0.4421779],[0.9775497,0.2737045,0.2919063,0.4673147],[0.6326591,0.4695119,0.9821513,0.03036699]]]], numpy.float)";
        public const string Y_1_1_4_4 = @"numpy.array([[1,0,0]], numpy.float)";
        public const string X_1_1_2_2 = @"numpy.array([[[[0.7262433, 0.8173254],[0.2060332, 0.5588848]]]], numpy.float)";
        public const string Y_1_1_2_2 = @"numpy.array([[1,0,0]], numpy.float)";
        private const string X_1_1_2_1 = @"numpy.array([[[[0.7262433],[0.2060332]]]], numpy.float)";
        private const string Y_1_1_2_1 = @"numpy.array([[1,0,0]], numpy.float)";
        private const string W_N_1_4_4 = "[[0.22065729, -0.11788255, -0.4187895],[0.32060236, -0.44626778, 0.24227637],[-0.46897227, 0.5059137, 0.4339162],[-0.02144825, -0.04082066, -0.09005189],[0.28492624, -0.28046286, -0.18176123],[-0.1717251, -0.55430335, -0.28846815],[0.29476583, -0.3019745, 0.03277987],[0.41012663, 0.09135884, 0.2522431],[-0.40020466, -0.2832676, 0.2568243],[0.47819465, 0.06466031, 0.45569366],[0.4343483, -0.30980763, -0.01376414],[0.09202623, -0.02883267, 0.19485158],[-0.5382978, -0.5129023, 0.47553152],[0.15798962, 0.43635488, 0.4626748],[-0.47213712, 0.17086667, -0.03163177],[0.01544881, 0.26190037, 0.38539213]]";

        [Test]
        public void TestSigmoidActivation_NCHW_1_1_1_1()
        {
            var X = FromNumpyArray(X_1_1_1_1, "X");
            var Y = FromNumpyArray(Y_1_1_1_1, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.BinaryCrossentropy);
            network
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray("[[0.5553087, -0.2966646]]", "Weights");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.5994776,0.4463447]]");
            TestLossAccuracy(network, X, Y, 0.551454782485962, 1.0);

            var learningRate = 0.1;
            var numEpochs = 10;
            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.66498965, 0.3707094]]");
            TestLossAccuracy(network, X, Y, 0.4355729818344116, 1.0);
        }

        [Test]
        public void TestSigmoidActivation_NCHW_1_1_4_4()
        {
            var X = FromNumpyArray(X_1_1_4_4, "X");
            var Y = FromNumpyArray(Y_1_1_4_4, "Y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.BinaryCrossentropy);
            network
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray(W_N_1_4_4, "Weights");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.41122958, 0.27045158, 0.7466786]]");
            TestLossAccuracy(network, X, Y, 0.8590097427368164, 0.0);

            var learningRate = 0.1;
            var numEpochs = 10;
            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.67996985, 0.17650272, 0.4107801]]");
            TestLossAccuracy(network, X, Y, 0.369619220495224, 1.0);
        }

        [Test]
        public void TestSigmoidActivation_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.BinaryCrossentropy);
            network
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray(W_N_1_4_4, "Weights");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.41122958, 0.27045155, 0.7466786],[0.65826976, 0.14434774, 0.69001585]]");
            TestLossAccuracy(network, X, Y, 0.6962824463844299, 0.5);

            var learningRate = 0.1;
            var numEpochs = 10;
            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.4153017,  0.19545524, 0.6464454],[0.6064757,  0.09901482, 0.62172073]]");
            TestLossAccuracy(network, X, Y, 0.6080149412155151, 0.5);
        }

        [Test]
        public void TestSoftmaxActivation_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "Y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            var w = FromNumpyArray("[[ 0.02377093, -0.36406565,  0.20111328],[-0.02257341,  0.49298579,  0.3783552 ],[-0.33265597,  0.22183669,  0.4130335 ],[ 0.03862739,  0.45694906,-0.046529  ],[-0.5435763 ,  0.4115948 ,  0.5266854 ],[-0.04584688, -0.08123899,  0.43348545],[-0.23025852, -0.24818823, -0.31672138],[-0.13403434, -0.39957535,  0.34845835],[-0.11953372, -0.18876502, -0.19744089],[-0.5492821 ,  0.52302474,  0.3208636 ],[ 0.18945718, 0.04014206, -0.3605097 ],[-0.47365752, -0.26253745, -0.2964717 ],[-0.2434968 , -0.34853765, -0.23780361],[ 0.4313671 ,  0.5169173 , -0.43086883],[ 0.00898802,  0.24687833,  0.17265934],[ 0.02312517, -0.22023779,  0.3136925 ]]", "Weights");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.11047623,0.41491902,0.47460473],[0.05679994,0.34877774,0.59442234]]");
            TestLossAccuracy(network, X, Y, 1.3615601062774658, 0.5);

            var learningRate = 0.1;
            var numEpochs = 10;
            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.6009891,0.09348926,0.30552167],[0.2477788,0.08346885,0.6687523]]");
            TestLossAccuracy(network, X, Y, 0.45576000213623047, 1.0);
        }

        private static CpuTensor<float> FromNumpyArray(string s, string description)
        {
            return (CpuTensor<float>)TensorExtensions.FromNumpyArray(s, description);
        }


        [Test]
        public void TestReluActivation_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.BinaryCrossentropy);
            network
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .AddDense_Activation(3, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray("[[ 0.22065729, -0.11788255, -0.4187895 ],[ 0.32060236, -0.44626778,  0.24227637],[-0.46897227,  0.5059137 ,  0.4339162 ],[-0.02144825, -0.04082066, -0.09005189],[ 0.28492624, -0.28046286, -0.18176123],[-0.1717251 , -0.55430335, -0.28846815],[ 0.29476583, -0.3019745 ,  0.03277987],[ 0.41012663,  0.09135884,  0.2522431 ],[-0.40020466, -0.2832676 ,  0.2568243 ],[ 0.47819465,  0.06466031,  0.45569366],[ 0.4343483 , -0.30980763, -0.01376414],[ 0.09202623, -0.02883267,  0.19485158],[-0.5382978 , -0.5129023 ,  0.47553152],[ 0.15798962,  0.43635488,  0.4626748 ],[-0.47213712,  0.17086667, -0.03163177],[ 0.01544881,  0.26190037,  0.38539213]]", "W_Layer1");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);
            w = FromNumpyArray("[[ 0.7206471 , -0.3155403 ,  0.16133356],[ 0.4253831 ,  0.71631813,  0.10403013],[ 0.4923072 ,  0.58519197,  0.364321  ]]", "W_Layer2");
            w.CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.6299896,  0.65307385, 0.5972025 ],[0.7039944 , 0.56498057, 0.5980379]]");
            TestLossAccuracy(network, X, Y, 0.8323098421096802, 0.0);

            var learningRate = 0.1;
            var numEpochs = 10;
            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.5302857,  0.49406603, 0.5208457 ],[0.5566553, 0.4216699,  0.5151303]]");
            TestLossAccuracy(network, X, Y, 0.6792957186698914, 0.5);
        }

        [Test]
        public void TestConvolutionWithReluActivation_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.BinaryCrossentropy);
            network
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .AddConvolution(3, 3, 1, 1, 0.0)
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray("[[[[-0.10066611, -0.22584948,  0.1257661 ]],[[ 0.00622791, -0.02702722, -0.19816945]],[[-0.00094005, -0.12673107,  0.10199177]]],[[[-0.05160269,  0.36979204, -0.38235503]],[[-0.25580615, -0.23532738, -0.18486507]],[[-0.18581466, -0.03875312, -0.18673505]]],[[[-0.1438927 , -0.05969113,  0.22153592]],[[ 0.02154535,  0.143184  ,  0.2194677 ]],[[-0.17963122,  0.14435953,  0.18853426]]]]", "Convolution");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[ 0.05013201,  0.0884136 ,  0.1288763 ],[ 0.10524932,  0.27004865, -0.15511033],[ 0.28799555, -0.11378004,  0.31027994],[-0.12980627, -0.26656348, -0.2889419 ],[ 0.10056138, -0.20606633,  0.11035499],[-0.19916984,  0.01184309, -0.02502242],[-0.00895432, -0.23922653, -0.14380434],[ 0.13250148,  0.12896249,  0.3411176 ],[-0.20010757, -0.07243675,  0.10569999],[-0.14625986, -0.2575507 , -0.2796294 ],[ 0.2984304 ,  0.12682551, -0.34131444],[ 0.33970162, -0.2596441 , -0.28711483],[ 0.2641308 ,  0.15033874, -0.17174129],[-0.31036156,  0.15232903, -0.2033331 ],[-0.0004667 ,  0.15065774,  0.12756902],[ 0.2866663 , -0.160675  , -0.12804145],[ 0.01153374, -0.11623923, -0.08252719],[ 0.12417665,  0.28663734, -0.12360954],[ 0.13087502,  0.15079209,  0.29951695],[-0.0907169 , -0.27126557,  0.00555232],[ 0.19179931, -0.2861278 ,  0.07780427],[-0.20458487, -0.27085418,  0.04733434],[-0.10611108, -0.09193736,  0.19488677],[ 0.13467175, -0.2872713 ,  0.2647117 ],[-0.24014097, -0.02662796,  0.22110483],[ 0.33133528, -0.18674679, -0.04942989],[ 0.07396188, -0.18812832, -0.14777936],[ 0.13951644, -0.29781634, -0.12320091],[-0.01970455, -0.22537778, -0.05007559],[-0.10169415, -0.3120061 ,  0.0934028 ],[-0.13796891, -0.31914735, -0.11247423],[ 0.20420077, -0.12212758, -0.30907962],[-0.25789154,  0.2055321 ,  0.11365542],[-0.10406806,  0.2673215 , -0.1856383 ],[ 0.05355045,  0.1597245 , -0.13172172],[ 0.14546981,  0.26738545,  0.02670237],[ 0.08399773, -0.12938716, -0.04259995],[-0.13436754,  0.25714287, -0.01506558],[-0.26373556,  0.31247166, -0.0151737 ],[-0.058229  ,  0.2936549 ,  0.2405878 ],[-0.29457894,  0.05585265, -0.33545914],[-0.12306491, -0.32960945, -0.01645941],[-0.04173017,  0.24279085,  0.21392396],[-0.20707619,  0.1420064 , -0.16330862],[-0.07069319,  0.312768  , -0.2855286 ],[ 0.07745105, -0.17894101,  0.3308623 ],[ 0.21007964, -0.25078928,  0.19156727],[ 0.02520046, -0.11668615,  0.3065426 ]]", "W_Layer2");
            w.CopyTo(((DenseLayer)network.Layers[2]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.3302841,0.7452456,0.4357071],[0.2857407,0.7822333,0.4093774]]");
            TestLossAccuracy(network, X, Y, 0.966899474461873, 0.0);

            var learningRate = 0.1;
            var numEpochs = 10;
            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.5584541,0.1351326,0.4495322],[0.4868227,0.1388872,0.4988887]]");
            TestLossAccuracy(network, X, Y, 0.472797473271688, 1.0);
        }

        [Test]
        public void TestBatchNormalisation_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.BinaryCrossentropy);
            network
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .Flatten()
                .AddBatchNorm(0.99, 1e-5)
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray("[[ 0.5398403 ,  0.3300526 ,  0.55018014],[ 0.14779323, -0.34529093, -0.16082811],[ 0.25194514, -0.5108937 ,  0.22403759],[-0.15200073,  0.0372324 , 0.38966674],[-0.06222537, -0.49777025,  0.17976868],[-0.30487314,  0.29002702, -0.1486885 ],[ 0.21023047, -0.5419708 , -0.44205534],[ 0.03658289, 0.5347499 ,  0.04729468],[-0.29706508, -0.5559816 ,  0.54104596],[-0.526604  ,  0.12949431, -0.07999322],[ 0.07322848,  0.3647598 ,  0.03496403],[-0.5040164 ,  0.03338426, -0.34131938],[ 0.3909973 ,  0.22031981,  0.2741294 ],[ 0.36716205, -0.21828368,  0.42880273],[-0.03759038, 0.17174226, -0.33242768],[-0.26423737, -0.43534094, -0.30766475]]", "Weights");
            w.CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.5917179,0.3094307,0.6521813],[0.497325,0.2508813,0.5720119]]");
            TestLossAccuracy(network, X, Y, 0.581050515174866, 0.5);

            var learningRate = 0.1;
            var numEpochs = 10;
            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.8123572,0.3967234,0.3261538],[0.184946,0.3705481,0.6772146]]");
            TestLossAccuracy(network, X, Y, 0.360853592554728, 1.0);
        }

        [Test]
        public void TestResNet_Shortcut_Same_Dimension_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .AddConvolution(1, 1, 1, 0, 0.0)
                .AddConvolution(1, 1, 1, 0, 0.0)
                .AddSumLayer(1, 2)
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            Tensor w = FromNumpyArray("[[[[-0.7714059]]]]", "Convolution0");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[[-1.0248963]]", "Convolution1");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Convolution);
            w = FromNumpyArray("[[-0.0961059 , -0.39184043, -0.2705268 ],[0.07956541, 0.22585392, -0.341879],[-0.27921697, -0.32074332, 0.23311937],[-0.5426143, -0.2890964, -0.18300149],[0.10221738, -0.05122566, 0.1905775],[0.5236904, 0.12203938, 0.30513626],[0.17442077, -0.38318935, -0.10136446],[-0.3381198, 0.28183156, 0.46150166],[0.27193302, -0.16640529, -0.41912097],[0.25417566, 0.06102628, -0.52639526],[-0.14935666, -0.5422965, -0.03686011],[-0.3144787, -0.02274132, -0.23660958],[-0.3006308, -0.26082158, 0.16282296],[-0.35234135, 0.07790905, 0.10894704],[-0.30488306, -0.17647654, 0.30045635],[0.48848134, 0.53268725, -0.46586674]]", "WeightsDense");
            w.CopyTo(((DenseLayer)network.Layers[4]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.3337501,0.3300667,0.3361832],[0.3392599,0.3294313,0.3313088]]");
            TestLossAccuracy(network, X, Y, 1.10103356838226, 0.0);

            var learningRate = 0.1;
            var numEpochs = 10;
            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.3923742,0.2287465,0.3788793],[0.3707,0.2305237,0.3987763]]");
            TestLossAccuracy(network, X, Y, 0.927446961402893, 1.0);
        }

        [Test]
        public void TestResNet_Shortcut_Different_Dimension_With_Conv_1x1_to_change_Diemnsion_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .AddConvolution(1, 1, 1, 0, 0.0)
                .AddConvolution(1, 1, 1, 0, 0.0); //left
            network.AddConvolution(1, 1, 1, 0, 0.0, 1); //right (identity shortcut)
            network.AddSumLayer(3, 2)
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)
                ;
            Tensor w = FromNumpyArray("[[[[-0.7714059]]]]", "Convolution0");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[[-1.0248963]]", "Convolution1");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Convolution);
            w = FromNumpyArray("[[[1.4231325]]", "Convolution1");
            w.CopyTo(((ConvolutionLayer)network.Layers[3]).Convolution);
            w = FromNumpyArray("[[ 0.32856905,  0.1894297 ,  0.4066078 ],[-0.43956745, 0.52707547, -0.20674482],[0.31645727, -0.31735897, -0.38774815],[-0.15041429, 0.02662414, -0.3353554],[0.22785252, 0.538137, 0.03771406],[-0.35584196, -0.04203749, 0.46805507],[0.22338176, -0.34921265, 0.51070255],[-0.05367857, 0.31961358, -0.46928698],[-0.20997655, 0.03387326, 0.39165902],[-0.28344244, 0.3322929, 0.17337584],[0.01335454, 0.37127644, -0.52875155],[0.09800142, 0.21306825, 0.31867707],[0.35722166, 0.34757876, 0.0046258],[-0.12657085, 0.43093973, -0.27573565],[-0.41127366, 0.11429685, 0.06350583],[-0.09927812, -0.04027134, 0.16407043]]", "WeightsDense");
            w.CopyTo(((DenseLayer)network.Layers[5]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.4210416,0.2635172,0.3154411],[0.4182904,0.2633894,0.3183202]]");
            TestLossAccuracy(network, X, Y, 1.004860520362856, 0.5);

            var learningRate = 0.1;
            var numEpochs = 10;
            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.5254775,0.02025996,0.4542625],[0.4535444,0.01977866,0.526677]]");
            TestLossAccuracy(network, X, Y, 0.642307877540588, 1.0);
        }
        [Test, Ignore]
        public void TestParallelRunWithTensorFlow()
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;

            var logFileName = Utils.ConcatenatePathWithFileName(@"c:\temp\ML\", "NetworkPropagation" + "_" + Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            var logger = new Logger(logFileName, true);

            //2_1_4_4
            var X_train = FromNumpyArray(X_2_1_4_4, "X_train"); var Y_train = FromNumpyArray(Y_2_1_4_4, "Y_train");
            //1_1_4_4
            //var X_train = FromNumpyArray(X_1_1_4_4, "X_train");var Y_train = FromNumpyArray(Y_1_1_4_4, "Y_train");
            //1_1_2_2
            //var X_train = FromNumpyArray(X_1_1_2_2, "X_train");var Y_train = FromNumpyArray(Y_1_1_2_2, "Y_train");
            //1_1_2_1
            //var X_train = FromNumpyArray(X_1_1_2_1, "X_train"); var Y_train = FromNumpyArray(Y_1_1_2_1, "Y_train");
            //1_1_1_1
            //var X_train = FromNumpyArray("[[[[1.0]]]]", "X_train");var Y_train = FromNumpyArray("[[1, 0]]", "Y_train");

            int batchSize = X_train.Shape[0];
            var network = new Network(new NetworkConfig(false) { Logger = logger, UseDoublePrecision = false, LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy }
                //.WithAdam(beta1,beta2)
                );
            network
                .AddInput(X_train.Shape[1], X_train.Shape[2], X_train.Shape[3])
                .AddConvolution(1, 1, 1, 0,0.0)
                .AddConvolution(1, 1, 1, 0, 0.0); //left

            network.AddConvolution(1, 1, 1, 0, 0.0, 1); //right (idnetity shortcut)

            network.AddSumLayer(3, 2)
                .Flatten()
                //.AddConvolution_Activation(1, 1, 1, 0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                //.AddPooling(2, 2)
                //.AddDense(1, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                //.AddBatchNorm()
                //.AddConvolution(3, 3, 1, 1)
                //.AddPooling(2, 2)
                //.AddDense(2, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)5
                .AddOutput(Y_train.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)
                ;

            logger.Info(network.Summary() + Environment.NewLine);

            Tensor w = FromNumpyArray("[[[[-0.7714059]]]]", "Convolution0");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[[-1.0248963]]", "Convolution1");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Convolution);
            w = FromNumpyArray("[[[1.4231325]]", "Convolution1");
            w.CopyTo(((ConvolutionLayer)network.Layers[3]).Convolution);
            w = FromNumpyArray("[[ 0.32856905,  0.1894297 ,  0.4066078 ],[-0.43956745, 0.52707547, -0.20674482],[0.31645727, -0.31735897, -0.38774815],[-0.15041429, 0.02662414, -0.3353554],[0.22785252, 0.538137, 0.03771406],[-0.35584196, -0.04203749, 0.46805507],[0.22338176, -0.34921265, 0.51070255],[-0.05367857, 0.31961358, -0.46928698],[-0.20997655, 0.03387326, 0.39165902],[-0.28344244, 0.3322929, 0.17337584],[0.01335454, 0.37127644, -0.52875155],[0.09800142, 0.21306825, 0.31867707],[0.35722166, 0.34757876, 0.0046258],[-0.12657085, 0.43093973, -0.27573565],[-0.41127366, 0.11429685, 0.06350583],[-0.09927812, -0.04027134, 0.16407043]]", "WeightsDense");
            w.CopyTo(((DenseLayer)network.Layers[6]).Weights);

            var predict_before = network.Predict(X_train, false).ToNumpy();
            network.LogContent();
            var lossAccuracyBefore = network.ComputeLossAndAccuracy(batchSize, X_train, Y_train);

            logger.Info("-");
            logger.Info("--------------------------------------------------------------------");
            logger.Info("-");

            network.Fit(X_train, Y_train, learningRate, numEpochs, batchSize);
            network.LogContent();

            var predict_after = network.Predict(X_train, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeLossAndAccuracy(batchSize, X_train, Y_train);

            logger.Info("C# numEpochs= " + numEpochs);
            logger.Info("C# learningRate= " + learningRate);
            logger.Info("C# prediction_before= " + predict_before);
            logger.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            logger.Info("C# prediction_after= " + predict_after);
            logger.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
        }

        private static Network GetNetwork(NetworkConfig.LossFunctionEnum lossFunction)
        {
            return new Network(new NetworkConfig(false) { Logger = Logger.NullLogger, UseDoublePrecision = false, LossFunction = lossFunction, RandomizeOrder = false});
        }
        private static void TestPredict(Network network, Tensor X, string expectedPredictionAsString)
        {
            var observedPrediction = network.Predict(X, false);
            var expectedPrediction = FromNumpyArray(expectedPredictionAsString, "expected");
            Assert.IsTrue(TestTensor.SameContent(observedPrediction, expectedPrediction, 1e-6), "expecting: " + Environment.NewLine + expectedPrediction.ToNumpy()+Environment.NewLine+ " but was:" + Environment.NewLine + observedPrediction.ToNumpy());
        }
        private static void TestLossAccuracy(Network network, CpuTensor<float> X, CpuTensor<float> Y_expected, double expectedLoss, double expectedAccuracy)
        {
            var batchSize = X.Shape[0];
            var observedLossAccuracy = network.ComputeLossAndAccuracy(batchSize, X, Y_expected);
            Assert.AreEqual(expectedLoss, observedLossAccuracy.Item1, 1e-6, "expeted loss: "+expectedLoss+" but was: "+observedLossAccuracy.Item1);
            Assert.AreEqual(expectedAccuracy, observedLossAccuracy.Item2, 1e-6, "expeted accuracy: " + expectedAccuracy + " but was: " + observedLossAccuracy.Item2);
        }
    }
}

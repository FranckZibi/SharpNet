using System;
using System.Diagnostics;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;
using SharpNet.Pictures;
using SharpNetTests.Data;
// ReSharper disable UnusedMember.Local

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestNetworkPropagation
    {
        private const string X_1_1_1_1 = @"numpy.array([[[[1]]]], numpy.float)";
        private const string Y_1_1_1_1 = @"numpy.array([[1,0]], numpy.float)";
        public const string X_2_1_4_4 = @"numpy.array([[[[0.7262433,0.8173254,0.7680227,0.5581612],[0.2060332,0.5588848,0.9060271,0.4421779],[0.9775497,0.2737045,0.2919063,0.4673147],[0.6326591,0.4695119,0.9821513,0.03036699]]],[[[0.8623701,0.9953471,0.6771811,0.3145918],[0.8169079,0.8480518,0.9919022,0.0326252],[0.699942,0.5262842,0.9340187,0.6876203],[0.5468155,0.08110995,0.1871246,0.4533272]]]], numpy.float)";
        public const string Y_2_1_4_4 = @"numpy.array([[1,0,0], [0,0,1]], numpy.float)";
        private const string X_1_1_4_4 = @"numpy.array([[[[0.7262433,0.8173254,0.7680227,0.5581612],[0.2060332,0.5588848,0.9060271,0.4421779],[0.9775497,0.2737045,0.2919063,0.4673147],[0.6326591,0.4695119,0.9821513,0.03036699]]]], numpy.float)";
        private const string Y_1_1_4_4 = @"numpy.array([[1,0,0]], numpy.float)";
        //public const string X_1_1_2_2 = @"numpy.array([[[[0.7262433, 0.8173254],[0.2060332, 0.5588848]]]], numpy.float)";
        //public const string Y_1_1_2_2 = @"numpy.array([[1,0,0]], numpy.float)";
        private const string X_1_1_2_1 = @"numpy.array([[[[0.7262433],[0.2060332]]]], numpy.float)";
        private const string Y_1_1_2_1 = @"numpy.array([[1,0]], numpy.float)";
        private const string W_N_1_4_4 = "[[0.22065729, -0.11788255, -0.4187895],[0.32060236, -0.44626778, 0.24227637],[-0.46897227, 0.5059137, 0.4339162],[-0.02144825, -0.04082066, -0.09005189],[0.28492624, -0.28046286, -0.18176123],[-0.1717251, -0.55430335, -0.28846815],[0.29476583, -0.3019745, 0.03277987],[0.41012663, 0.09135884, 0.2522431],[-0.40020466, -0.2832676, 0.2568243],[0.47819465, 0.06466031, 0.45569366],[0.4343483, -0.30980763, -0.01376414],[0.09202623, -0.02883267, 0.19485158],[-0.5382978, -0.5129023, 0.47553152],[0.15798962, 0.43635488, 0.4626748],[-0.47213712, 0.17086667, -0.03163177],[0.01544881, 0.26190037, 0.38539213]]";

        [Test]
        public void TestSigmoidActivation_NCHW_1_1_1_1()
        {
            var X = FromNumpyArray(X_1_1_1_1, "X");
            var Y = FromNumpyArray(Y_1_1_1_1, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.BinaryCrossentropy);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray("[[0.5553087, -0.2966646]]", "Weights");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.635366380214691,0.426373064517975]]");
            TestLossAccuracy(network, X, Y, 0.504664778709412, 1.0);

            var learningRate = 0.1;
            var numEpochs = 10;
            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.707915127277374,0.336254686117172]]");
            TestLossAccuracy(network, X, Y, 0.377643942832947, 1.0);
        }

        [Test]
        public void TestSigmoidActivation_NCHW_1_1_4_4()
        {
            var X = FromNumpyArray(X_1_1_4_4, "X");
            var Y = FromNumpyArray(Y_1_1_4_4, "Y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.BinaryCrossentropy);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

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
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

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
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

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
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Dense_Activation(3, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

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
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(3, 3, 1, 1, 0.0, true)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

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
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Flatten()
                .BatchNorm(0.99, 1e-5)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

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
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, 0, 0.0, true)
                .Convolution(1, 1, 1, 0, 0.0, true)
                .AddLayer(1, 2)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

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
        public void TestResNet_Shortcut_Different_Dimension_With_Conv_1x1_to_change_Dimension_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, 0, 0.0, true)
                .Convolution(1, 1, 1, 0, 0.0, true); //left
            network.Convolution(1, 1, 1, 0, 0.0, true, 1); //right (identity shortcut)
            network.AddLayer(3, 2)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)
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

        [Test]
        public void TestL2Regularization_ConvolutionLayer_SGDVanilla_NCHW_2_1_4_4()
        {
            const int numEpochs = 10;
            const double learningRate = 0.12;
            const double lambdaL2Regularization = 0.05;
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, 0, lambdaL2Regularization, true)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            Tensor w = FromNumpyArray("[[[[-0.7714059]]]]", "Convolution0");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[-0.10847557,  0.00658482,  0.41918087],[ 0.5224567 ,  0.42545766, -0.31801027],[-0.28710383, -0.31568986,  0.02822173],[-0.4120677 ,  0.21388823, -0.22343507],[-0.00887001,  0.42890936,  0.00528443],[ 0.14757729, -0.45275694, -0.36124444],[-0.5223615 ,  0.06962186,  0.44158655],[-0.44399977,  0.25540823, -0.35566014],[ 0.31000054,  0.03869426,  0.37737155],[-0.28234982,  0.43704945, -0.08071807],[-0.41145545,  0.41357315,  0.5401688 ],[-0.40983498, -0.47532582, -0.2516185 ],[-0.02894175,  0.07887733, -0.33317018],[ 0.07574445,  0.37989277, -0.47620153],[-0.5085196 ,  0.04452544, -0.4278263 ],[ 0.42463195,  0.26129186, -0.37209088]]", "WeightsDense");
            w.CopyTo(((DenseLayer)network.Layers[2]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.553912222385406,0.135723426938057,0.310364335775375],[0.57789421081543,0.159836992621422,0.26226881146431]]");
            TestLossAccuracy(network, X, Y, null  /*0.9943205118179321*/, 0.5);

            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.56008917093277,0.0195358134806156,0.420375019311905],[0.398750722408295,0.0234715715050697,0.577777683734894]]");
            TestLossAccuracy(network, X, Y, null /*0.5985526442527771*/, 1.0);
        }

        [Test]
        public void TestL2Regularization_DenseLayer_SGDVanilla_NCHW_2_1_4_4()
        {
            const int numEpochs = 10;
            const double learningRate = 0.12;
            const double lambdaL2Regularization = 0.05;
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            Tensor w = FromNumpyArray("[[-0.3793878 ,  0.13005257, -0.48190022],[-0.5270703 , -0.5069973 , -0.45630288],[-0.08369148, -0.24146178, -0.09606424],[-0.0498544 , -0.4154459 , -0.3665961 ],[-0.3581952 , -0.3345901 ,  0.48476475],[ 0.320306  ,  0.301827  , -0.48490363],[ 0.33425486, -0.42483532,  0.20156533],[ 0.0346387 ,  0.34260863,  0.45479387],[-0.28320554,  0.27089173, -0.5511215 ],[-0.09140414, -0.2540371 , -0.38209555],[ 0.30901152, -0.22211927, -0.07776272],[-0.01273596, -0.43774882,  0.319129  ],[-0.26144847,  0.45303112, -0.5552845 ],[ 0.0012697 , -0.24624684, -0.01347905],[ 0.18339497, -0.46073103,  0.54499584],[-0.32917506,  0.03634387, -0.5220559 ]]", "Dense0");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.475850999355316,0.251384913921356,0.27276411652565],[0.506687998771667,0.285933136940002,0.207378879189491]]");
            TestLossAccuracy(network, X, Y, null  /*1.4464839696884155*/, 0.5);

            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.56395947933197,0.0651973560452461,0.370843172073364],[0.344296395778656,0.0696434527635574,0.586060106754303]]");
            TestLossAccuracy(network, X, Y, null /*0.8020790815353394*/, 1.0);
        }

        [Test]
        public void TestL2Regularization_DenseLayer_SGDMomentum_NCHW_2_1_4_4()
        {
            const int numEpochs = 10;
            const double learningRate = 0.12;
            const double lambdaL2Regularization = 0.5;
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithSGD(0.9, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            Tensor w = FromNumpyArray("[[-0.3793878 ,  0.13005257, -0.48190022],[-0.5270703 , -0.5069973 , -0.45630288],[-0.08369148, -0.24146178, -0.09606424],[-0.0498544 , -0.4154459 , -0.3665961 ],[-0.3581952 , -0.3345901 ,  0.48476475],[ 0.320306  ,  0.301827  , -0.48490363],[ 0.33425486, -0.42483532,  0.20156533],[ 0.0346387 ,  0.34260863,  0.45479387],[-0.28320554,  0.27089173, -0.5511215 ],[-0.09140414, -0.2540371 , -0.38209555],[ 0.30901152, -0.22211927, -0.07776272],[-0.01273596, -0.43774882,  0.319129  ],[-0.26144847,  0.45303112, -0.5552845 ],[ 0.0012697 , -0.24624684, -0.01347905],[ 0.18339497, -0.46073103,  0.54499584],[-0.32917506,  0.03634387, -0.5220559 ]]", "Dense0");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.475850999355316,0.251384913921356,0.27276411652565],[0.506687998771667,0.285933136940002,0.207378879189491]]");
            TestLossAccuracy(network, X, Y, null  /* 4.0434770584106445 */, 0.5);

            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.635447382926941,0.101806730031967,0.262745916843414],[0.348914474248886,0.0936608985066414,0.557424664497375]]");
            TestLossAccuracy(network, X, Y, null /*1.5627011060714722*/, 1.0);
        }


        [Test]
        public void TestL2Regularization_DenseLayer_Adam_NCHW_2_1_4_4()
        {
            const int numEpochs = 10;
            const double learningRate = 0.12;
            const double lambdaL2Regularization = 0.05;
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_1_4_4, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithAdam(0.9, 0.999);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            Tensor w = FromNumpyArray("[[-0.3793878 ,  0.13005257, -0.48190022],[-0.5270703 , -0.5069973 , -0.45630288],[-0.08369148, -0.24146178, -0.09606424],[-0.0498544 , -0.4154459 , -0.3665961 ],[-0.3581952 , -0.3345901 ,  0.48476475],[ 0.320306  ,  0.301827  , -0.48490363],[ 0.33425486, -0.42483532,  0.20156533],[ 0.0346387 ,  0.34260863,  0.45479387],[-0.28320554,  0.27089173, -0.5511215 ],[-0.09140414, -0.2540371 , -0.38209555],[ 0.30901152, -0.22211927, -0.07776272],[-0.01273596, -0.43774882,  0.319129  ],[-0.26144847,  0.45303112, -0.5552845 ],[ 0.0012697 , -0.24624684, -0.01347905],[ 0.18339497, -0.46073103,  0.54499584],[-0.32917506,  0.03634387, -0.5220559 ]]", "Dense0");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.475850999355316,0.251384913921356,0.27276411652565],[0.506687998771667,0.285933136940002,0.207378879189491]]");
            TestLossAccuracy(network, X, Y, null  /*  1.4464839696884155 */, 0.5);

            var batchSize = X.Shape[0];
            network.Fit(X, Y, learningRate* batchSize, numEpochs, batchSize);

            //predictions after training
            TestPredict(network, X, "[[0.894426345825195,0.00220351060852408,0.103370226919651],[0.0549939684569836,0.00156258791685104,0.943443357944489]]");
            TestLossAccuracy(network, X, Y, null /*0.4707931876182556*/, 1.0);
        }

        //TODO add Concatenate test with bigger input 'x'
        [Test]
        public void TestConcatenate_NCHW_1_1_1_1()
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_1_1_1_1, "X");
            var Y = FromNumpyArray(Y_1_1_1_1, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, 0, lambdaL2Regularization, true)
                .Convolution(1, 1, 1, 0, lambdaL2Regularization, true)
                .ConcatenateLayer(1, 2)
                .Flatten()
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            var w = FromNumpyArray("[[[[-0.7714059]]]]", "Convolution1");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[[[-1.0248963]]]]", "Convolution2");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Convolution);
            w = FromNumpyArray("[[-0.2765898 ,  0.26417303],[0.8351141, -1.053136]]", "Dense");
            w.CopyTo(((DenseLayer)network.Layers[5]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.8710213, 0.12897871]]");
            TestLossAccuracy(network, X, Y, 0.1380888819694519, 1.0);
            network.Fit(X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[9.9985039e-01, 1.4956875e-04]]");
            TestLossAccuracy(network, X, Y, 0.0001495592441642657, 1.0);
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow()
        {
            const int numEpochs = 1;
            const double learningRate = 1.0;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var logFileName = Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory, "NetworkPropagation" + "_" + Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            var logger = new Logger(logFileName, true);

            //2_1_4_4
            //var X = FromNumpyArray(X_2_1_4_4, "X_train"); var Y = FromNumpyArray(Y_2_1_4_4, "Y_train");
            //1_1_4_4
            //var X = FromNumpyArray(X_1_1_4_4, "X_train");var Y = FromNumpyArray(Y_1_1_4_4, "Y_train");
            //1_1_2_2
            //var X = FromNumpyArray(X_1_1_2_2, "X_train");var Y = FromNumpyArray(Y_1_1_2_2, "Y_train");
            //1_1_2_1
            var X = FromNumpyArray(X_1_1_2_1, "X_train"); var Y = FromNumpyArray(Y_1_1_2_1, "Y_train");
            //1_1_1_1
            //var X = FromNumpyArray("[[[[1.0]]]]", "X_train");var Y = FromNumpyArray("[[1, 0]]", "Y_train");

            int batchSize = X.Shape[0];
            var gpuDeviceId = -1;
            var network = new Network(new NetworkConfig{ Logger = logger, UseDoublePrecision = false, LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy, RandomizeOrder = false }
                       //.WithAdam(0.9,0.999)
                       .WithSGD(momentum, false)
                        ,ImageDataGenerator.NoDataAugmentation, 
                        gpuDeviceId
                );
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, 0, lambdaL2Regularization, true)
                .Convolution(1, 1, 1, 0, lambdaL2Regularization, true)
                .ConcatenateLayer(1,2)
                .Flatten();
//            network.Dense(2, lambdaL2Regularization);

            /*
            network.AddConvolution(1, 1, 1, 0, lambdaL2Regularization); //left
            network.AddConvolution(1, 1, 1, 0, lambdaL2Regularization, 1); //right (identity shortcut)
            network.AddSumLayer(3, 2);
            */

            //.AddConvolution_Activation(1, 1, 1, 0)
            //.AddActivation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
            //.AddPooling(2, 2)
            //.AddDense(1, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
            //.AddBatchNorm()
            //.AddConvolution(3, 3, 1, 1)
            //.AddPooling(2, 2)
            //.AddDense(2, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)5
            network.Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);


            logger.Info(network.Summary() + Environment.NewLine);

            //Tensor w = FromNumpyArray("[[-0.3793878, 0.13005257, -0.48190022],[-0.5270703, -0.5069973, -0.45630288],[-0.08369148, -0.24146178, -0.09606424],[-0.0498544, -0.4154459, -0.3665961],[-0.3581952, -0.3345901, 0.48476475],[0.320306, 0.301827, -0.48490363],[0.33425486, -0.42483532, 0.20156533],[0.0346387, 0.34260863, 0.45479387],[-0.28320554, 0.27089173, -0.5511215],[-0.09140414, -0.2540371, -0.38209555],[0.30901152, -0.22211927, -0.07776272],[-0.01273596, -0.43774882, 0.319129],[-0.26144847, 0.45303112, -0.5552845],[0.0012697, -0.24624684, -0.01347905],[0.18339497, -0.46073103, 0.54499584],[-0.32917506, 0.03634387, -0.5220559]]", "Dense0");
            //w.CopyTo(((DenseLayer)network.Layers[2]).Weights);
            Tensor w = FromNumpyArray("[[[[-0.7714059]]]]", "Convolution1");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[[[-1.0248963]]]]", "Convolution2");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Convolution);
            //w = FromNumpyArray("[[-0.0935044 ,  0.08930674,  0.2823201 ],[-0.35602492,  0.11951789, -0.09273729],[ 0.1940693 ,  0.02401561, -0.24284391],[-0.02394128,  0.07824224, -0.38054535],[ 0.32493195, -0.35884738, -0.38189083],[ 0.29424062,  0.04143113, -0.40191767],[-0.05293348,  0.19875017, -0.3346681 ],[ 0.03057194, -0.3735914 , -0.0674623 ],[ 0.3880823 , -0.09871182, -0.17574972],[-0.29038104,  0.25778064, -0.12032074],[ 0.11203721, -0.10517201,  0.3708817 ],[-0.1527186 ,  0.21801105, -0.26964042],[-0.31762403,  0.38830027, -0.3753271 ],[ 0.2077305 ,  0.0135166 , -0.18412004],[-0.35179925, -0.19563204,  0.27091482],[ 0.23023447,  0.01704031, -0.13919547],[-0.23399019, -0.26903233, -0.13446775],[-0.05098614, -0.3045844 , -0.07806098],[-0.13110533,  0.40553764,  0.23116526],[-0.12380904, -0.1990231 ,  0.16275999],[-0.3359664 , -0.10431516, -0.04660624],[ 0.20998248,  0.36364272,  0.10899469],[-0.32446536,  0.08360443,  0.2051796 ],[ 0.3240768 ,  0.33368257, -0.2362173 ],[-0.10315731, -0.37046236, -0.05063343],[-0.07400975,  0.33962646,  0.08680078],[-0.24701846,  0.38117084, -0.40346375],[-0.22841114, -0.2256576 ,  0.01416683],[-0.14235109, -0.1011408 ,  0.06660545],[-0.386455  , -0.29496735, -0.00108448],[ 0.36282876,  0.07580084, -0.21415901],[ 0.3094258 ,  0.23421249, -0.1189253 ]]", "Dense");
            w = FromNumpyArray("[[-0.22583461,  0.21569633],[ 0.68186784, -0.8598819 ],[ 0.28866315, -0.22398186],[ 0.46872187,  0.05800319]]", "Dense");
            //w = FromNumpyArray("[[-0.2765898 ,  0.26417303],[ 0.8351141 , -1.053136  ]]", "Dense");

                
            //w = FromNumpyArray("[[-0.4380005 ,  0.87558043]],", "Dense");
            w.CopyTo(((DenseLayer)network.Layers[5]).Weights);
            //w = FromNumpyArray("[[ 0.32856905,  0.1894297 ,  0.4066078 ],[-0.43956745, 0.52707547, -0.20674482],[0.31645727, -0.31735897, -0.38774815],[-0.15041429, 0.02662414, -0.3353554],[0.22785252, 0.538137, 0.03771406],[-0.35584196, -0.04203749, 0.46805507],[0.22338176, -0.34921265, 0.51070255],[-0.05367857, 0.31961358, -0.46928698],[-0.20997655, 0.03387326, 0.39165902],[-0.28344244, 0.3322929, 0.17337584],[0.01335454, 0.37127644, -0.52875155],[0.09800142, 0.21306825, 0.31867707],[0.35722166, 0.34757876, 0.0046258],[-0.12657085, 0.43093973, -0.27573565],[-0.41127366, 0.11429685, 0.06350583],[-0.09927812, -0.04027134, 0.16407043]]", "WeightsDense");
            //w = FromNumpyArray("[[0.8268806 , 0.47672093]]", "WeightsDense");
            //w.CopyTo(((DenseLayer)network.Layers[6]).Weights);
            //w = FromNumpyArray("[[-0.32775342,  0.7416243 , -0.8244499 ],[-0.85675025, 0.4522856, -0.45435268],[-0.48980364, 0.28126568, -0.28207463],[0.395963, 0.501062, 0.76570255],[0.24764937, 0.12065065, -0.66572535]]", "WeightsDense");
            //w.CopyTo(((DenseLayer)network.Layers[3]).Weights);

            var predict_before = network.Predict(X, false).ToNumpy();
            network.LogContent();
            var lossAccuracyBefore = network.ComputeLossAndAccuracy(batchSize, X, Y);

            logger.Info("-");
            logger.Info("--------------------------------------------------------------------");
            logger.Info("-");

            network.Fit(X, Y, learningRate, numEpochs, batchSize);
            network.LogContent();

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeLossAndAccuracy(batchSize, X, Y);

            logger.Info("C# numEpochs= " + numEpochs);
            logger.Info("C# learningRate= " + learningRate);
            logger.Info("C# l2regularizer= " + lambdaL2Regularization);
            logger.Info("C# momentum= " + momentum);
            logger.Info("C# prediction_before= " + predict_before);
            logger.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            logger.Info("C# prediction_after= " + predict_after);
            logger.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
        }

        private static Network GetNetwork(NetworkConfig.LossFunctionEnum lossFunction)
        {
            var gpuDeviceId = -1;
            return new Network(new NetworkConfig{ Logger = Logger.NullLogger, UseDoublePrecision = false, LossFunction = lossFunction, RandomizeOrder = false, ForceTensorflowCompatibilityMode = true}, ImageDataGenerator.NoDataAugmentation, gpuDeviceId);
        }
        private static void TestPredict(Network network, Tensor X, string expectedPredictionAsString)
        {
            var observedPrediction = network.Predict(X, false);
            var expectedPrediction = FromNumpyArray(expectedPredictionAsString, "expected");
            Assert.IsTrue(TestTensor.SameContent(observedPrediction, expectedPrediction, 1e-6), "expecting: " + Environment.NewLine + expectedPrediction.ToNumpy()+Environment.NewLine+ " but was:" + Environment.NewLine + observedPrediction.ToNumpy());
        }
        private static void TestLossAccuracy(Network network, CpuTensor<float> X, CpuTensor<float> Y_expected, double? expectedLoss, double? expectedAccuracy)
        {
            var batchSize = X.Shape[0];
            var observedLossAccuracy = network.ComputeLossAndAccuracy(batchSize, X, Y_expected);
            if (expectedLoss.HasValue)
            { 
                Assert.AreEqual(expectedLoss.Value, observedLossAccuracy.Item1, 1e-6, "expected loss: " + expectedLoss.Value + " but was: " + observedLossAccuracy.Item1);
            }
            if (expectedAccuracy.HasValue)
            {
                Assert.AreEqual(expectedAccuracy.Value, observedLossAccuracy.Item2, 1e-6, "expected accuracy: " + expectedAccuracy.Value + " but was: " + observedLossAccuracy.Item2);
            }
        }
    }
}

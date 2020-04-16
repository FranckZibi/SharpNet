using System;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNetTests.Data;

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
        private const string X_1_1_4_4 = @"numpy.array([[[[0.7262433,0.8173254,0.7680227,0.5581612],[0.2060332,0.5588848,0.9060271,0.4421779],[0.9775497,0.2737045,0.2919063,0.4673147],[0.6326591,0.4695119,0.9821513,0.03036699]]]], numpy.float)";
        public const string X_2_3_4_5 = "numpy.array([[[[0.67872983,0.95197606,0.8040681,0.17448357,-0.88190055],[0.17665438,1.2180812,-0.17346638,1.4326493,-0.67888665],[-0.62428117,-0.0980559,0.39797723,-0.09146436,1.4464538],[-1.4088991,1.0871105,1.4860412,0.53154343,-0.55622464]],[[0.9507237,1.0441554,1.4757066,-1.4021244,0.599826],[0.07885243,1.302056,0.56286085,0.1404463,-1.2566701],[-0.9386263,-0.14001845,-0.6084844,1.4656314,0.42809242],[0.7888908,-1.4088172,-0.3569865,-0.47057447,1.3723655]],[[0.0153876385,0.6479175,-1.1431268,-0.67964554,1.2212938],[0.8842968,-0.48851898,-0.12837364,-1.0595249,-0.83605576],[-0.26978016,0.6561805,0.35949084,-0.036095843,-0.9152597],[1.1345744,0.97626954,0.7061925,1.0746577,0.53924704]]],[[[0.3745984,-0.8447797,1.1860874,1.1893194,-1.2402604],[1.0157373,-0.9895715,0.43234587,0.98044324,-0.842437],[1.4999181,-0.05590306,0.45661572,0.65917796,0.92838776],[-1.1971939,0.68728215,0.65535206,-1.1234182,1.2155292]],[[-0.9326286,-0.069570385,0.122385725,-0.52349794,0.51311],[-0.094806775,1.0004907,0.9276114,0.880891,0.79351795],[1.3126212,-0.6150096,0.068355896,0.4901785,-0.5022329],[-0.63983274,0.9618302,-0.8324462,-0.9393852,-1.2944435]],[[0.3738683,0.5791351,0.39118314,1.2829121,-0.83597386],[1.2861229,-1.3004352,1.2003129,0.53551644,-1.2180659],[-1.0527077,-1.1790825,0.32961074,1.3591285,-0.028124375],[-1.0558312,0.53283465,0.20958523,-0.8237906,0.35454643]]]], numpy.float)";
        public const string Y_2_2 = @"numpy.array([[1,0],[1,0]], numpy.float)";
        private const string Y_1_3 = @"numpy.array([[1,0,0]], numpy.float)";
        private const string W_N_1_4_4 = "[[0.22065729, -0.11788255, -0.4187895],[0.32060236, -0.44626778, 0.24227637],[-0.46897227, 0.5059137, 0.4339162],[-0.02144825, -0.04082066, -0.09005189],[0.28492624, -0.28046286, -0.18176123],[-0.1717251, -0.55430335, -0.28846815],[0.29476583, -0.3019745, 0.03277987],[0.41012663, 0.09135884, 0.2522431],[-0.40020466, -0.2832676, 0.2568243],[0.47819465, 0.06466031, 0.45569366],[0.4343483, -0.30980763, -0.01376414],[0.09202623, -0.02883267, 0.19485158],[-0.5382978, -0.5129023, 0.47553152],[0.15798962, 0.43635488, 0.4626748],[-0.47213712, 0.17086667, -0.03163177],[0.01544881, 0.26190037, 0.38539213]]";
        
        [Test]
        public void TestSigmoidActivation_NCHW_1_1_1_1()
        {
            var X = FromNumpyArray(X_1_1_1_1, "X");
            var Y = FromNumpyArray(Y_1_2, "y");
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
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.707915127277374,0.336254686117172]]");
            TestLossAccuracy(network, X, Y, 0.377643942832947, 1.0);
        }

        [Test]
        public void TestSigmoidActivation_NCHW_1_1_4_4()
        {
            var X = FromNumpyArray(X_1_1_4_4, "X");
            var Y = FromNumpyArray(Y_1_3, "Y");
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
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.67996985, 0.17650272, 0.4107801]]");
            TestLossAccuracy(network, X, Y, 0.369619220495224, 1.0);
        }

        [Test]
        public void TestSigmoidActivation_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_3, "y");
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
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.4153017,  0.19545524, 0.6464454],[0.6064757,  0.09901482, 0.62172073]]");
            TestLossAccuracy(network, X, Y, 0.6080149412155151, 0.5);
        }

        [Test]
        public void TestSoftmaxActivation_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_3, "Y");
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
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.6009891,0.09348926,0.30552167],[0.2477788,0.08346885,0.6687523]]");
            TestLossAccuracy(network, X, Y, 0.45576000213623047, 1.0);
        }

        public static CpuTensor<float> FromNumpyArray(string s, string description)
        {
            return (CpuTensor<float>)TensorExtensions.FromNumpyArray(s, description);
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
        /// <param name="description"></param>
        /// <returns></returns>/// 
        public static CpuTensor<float> FromConvNumpyArray(string s, string description)
        {
            var result = (CpuTensor<float>)TensorExtensions.FromNumpyArray(s, description);
            return (CpuTensor<float>)result.ChangeAxis(new[] { 3, 2, 0, 1 });
        }

        [Test]
        public void TestReluActivation_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_3, "y");
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
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.5302857,  0.49406603, 0.5208457 ],[0.5566553, 0.4216699,  0.5151303]]");
            TestLossAccuracy(network, X, Y, 0.6792957186698914, 0.5);
        }

        [Test]
        public void TestConvolutionWithReluActivation_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_3, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.BinaryCrossentropy);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(3, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true)
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
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.5584541,0.1351326,0.4495322],[0.4868227,0.1388872,0.4988887]]");
            TestLossAccuracy(network, X, Y, 0.472797473271688, 1.0);
        }

        [Test, Explicit]
        public void TestSimpleRNN()
        {
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.BinaryCrossentropy);
            network
                .Input(4, 3, 1)
                .SimpleRnnLayer(3, 5, 2, false);

            var layer = (SimpleRnnLayer) network.Layers[1];
            TensorExtensions.FromNumpyArray("array([[-0.64691669, 0.90148689, 2.52832571, -0.24863478, 0.04366899],[-0.22631424,  1.33145711, -0.28730786,  0.68006984, -0.3198016 ],[-1.27255876,  0.31354772,  0.50318481,  1.29322588, -0.11044703],[-0.61736206,  0.5627611 ,  0.24073709,  0.28066508, -0.0731127 ],[ 1.16033857,  0.36949272,  1.90465871,  1.1110567 ,  0.6590498 ]])", nameof(layer.Weights_aa))
                .Transpose().CopyTo(layer.Weights_aa);
            TensorExtensions.FromNumpyArray("array([[-1.62743834,  0.60231928,  0.4202822 ], [ 0.81095167,  1.04444209, -0.40087819], [ 0.82400562, -0.56230543,  1.95487808], [-1.33195167, -1.76068856, -1.65072127], [-0.89055558, -1.1191154 ,  1.9560789 ]])", nameof(layer.Weights_ax))
                .Transpose().CopyTo(layer.Weights_ax);
            TensorExtensions.FromNumpyArray("array([[-0.3264995 , -1.34267579,  1.11438298, -0.58652394, -1.23685338], [ 0.87583893,  0.62336218, -0.43495668,  1.40754   ,  0.12910158]])", nameof(layer.Weights_ay))
                .Transpose().CopyTo(layer.Weights_ay);
            TensorExtensions.FromNumpyArray("array([[ 1.6169496 ], [ 0.50274088], [ 1.55880554], [ 0.1094027 ], [-1.2197444 ]])", nameof(layer.Bias_a))
                .Transpose().CopyTo(layer.Bias_a);
            TensorExtensions.FromNumpyArray("array([[ 2.44936865], [-0.54577417]])", nameof(layer.Bias_y))
                .Transpose().CopyTo(layer.Bias_y);
            var X = ((CpuTensor<float>)TensorExtensions.FromNumpyArray("array([[[ 1.62434536, -0.61175641, -0.52817175, -1.07296862],[ 0.86540763, -2.3015387 ,  1.74481176, -0.7612069 ],[ 0.3190391 , -0.24937038,  1.46210794, -2.06014071],[-0.3224172 , -0.38405435,  1.13376944, -1.09989127],[-0.17242821, -0.87785842,  0.04221375,  0.58281521],[-1.10061918,  1.14472371,  0.90159072,  0.50249434],[ 0.90085595, -0.68372786, -0.12289023, -0.93576943],[-0.26788808,  0.53035547, -0.69166075, -0.39675353],[-0.6871727 , -0.84520564, -0.67124613, -0.0126646 ],[-1.11731035,  0.2344157 ,  1.65980218,  0.74204416]],[[-0.19183555, -0.88762896, -0.74715829,  1.6924546 ],[ 0.05080775, -0.63699565,  0.19091548,  2.10025514],[ 0.12015895,  0.61720311,  0.30017032, -0.35224985],[-1.1425182 , -0.34934272, -0.20889423,  0.58662319],[ 0.83898341,  0.93110208,  0.28558733,  0.88514116],[-0.75439794,  1.25286816,  0.51292982, -0.29809284],[ 0.48851815, -0.07557171,  1.13162939,  1.51981682],[ 2.18557541, -1.39649634, -1.44411381, -0.50446586],[ 0.16003707,  0.87616892,  0.31563495, -2.02220122],[-0.30620401,  0.82797464,  0.23009474,  0.76201118]],[[-0.22232814, -0.20075807,  0.18656139,  0.41005165],[ 0.19829972,  0.11900865, -0.67066229,  0.37756379],[ 0.12182127,  1.12948391,  1.19891788,  0.18515642],[-0.37528495, -0.63873041,  0.42349435,  0.07734007],[-0.34385368,  0.04359686, -0.62000084,  0.69803203],[-0.44712856,  1.2245077 ,  0.40349164,  0.59357852],[-1.09491185,  0.16938243,  0.74055645, -0.9537006 ],[-0.26621851,  0.03261455, -1.37311732,  0.31515939],[ 0.84616065, -0.85951594,  0.35054598, -1.31228341],[-0.03869551, -1.61577235,  1.12141771,  0.40890054]]])", "x_HNC")).From_HNC_to_NCH();
            TestPredict(network, X, "[[0.930321455001831,0.0696785822510719],[0.988288283348084,0.0117117157205939],[0.255148202180862,0.744851768016815],[0.256481111049652,0.743518948554993],[0.996308386325836,0.00369161483831704],[0.922233164310455,0.0777667835354805],[0.332797795534134,0.667202234268188],[0.970049381256104,0.0299505963921547],[0.959831774234772,0.0401682630181313],[0.990113973617554,0.00988596677780151]]");
        }

        [Test]
        public void TestBatchNormalization_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_3, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.BinaryCrossentropy);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Flatten()
                .BatchNorm(0.99, 1e-5, "BatchNorm")
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var w = FromNumpyArray("[[ 0.5398403 ,  0.3300526 ,  0.55018014],[ 0.14779323, -0.34529093, -0.16082811],[ 0.25194514, -0.5108937 ,  0.22403759],[-0.15200073,  0.0372324 , 0.38966674],[-0.06222537, -0.49777025,  0.17976868],[-0.30487314,  0.29002702, -0.1486885 ],[ 0.21023047, -0.5419708 , -0.44205534],[ 0.03658289, 0.5347499 ,  0.04729468],[-0.29706508, -0.5559816 ,  0.54104596],[-0.526604  ,  0.12949431, -0.07999322],[ 0.07322848,  0.3647598 ,  0.03496403],[-0.5040164 ,  0.03338426, -0.34131938],[ 0.3909973 ,  0.22031981,  0.2741294 ],[ 0.36716205, -0.21828368,  0.42880273],[-0.03759038, 0.17174226, -0.33242768],[-0.26423737, -0.43534094, -0.30766475]]", "Weights");
            w.CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.5917179,0.3094307,0.6521813],[0.497325,0.2508813,0.5720119]]");
            TestLossAccuracy(network, X, Y, 0.581050515174866, 0.5);

            var learningRate = 0.1;
            var numEpochs = 10;
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.606776178,0.248665854,0.616904914],[0.43677336,0.206704035,0.77601999]]");
            TestLossAccuracy(network, X, Y, 0.46736355622609455d, 0.5);
        }


        [Test]
        public void TestBatchNormalization_NCHW_2_3_4_5()
        {
            var numEpochs = 10;
            var learningRate = 0.001;

            var X = FromNumpyArray(X_2_3_4_5, "x");
            var Y = FromNumpyArray(Y_2_2, "y");

            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);

            network.Config.WithSGD(0.9, false);

            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 5, 1, ConvolutionLayer.PADDING_TYPE.SAME, 0.00, false)
                .BatchNorm(0.99, 0.001)
                .Flatten()
                .Dense(Y.Shape[1], 0.00)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            FromConvNumpyArray("[[[[-0.09757597744464874, -0.12943518161773682], [-0.022430911660194397, 0.04822628200054169], [-0.10801036655902863, -0.16410072147846222]], [[0.16708092391490936, 0.15502481162548065], [-0.09673155099153519, 0.13610444962978363], [0.00013612210750579834, 0.2021443396806717]], [[0.1365472823381424, -0.09114216268062592], [0.18680743873119354, 0.08596672117710114], [-0.07727599143981934, 0.21467427909374237]], [[0.034543201327323914, 0.20785339176654816], [-0.1841544210910797, -0.062430888414382935], [0.10619138181209564, 0.1961117833852768]], [[-0.08351261913776398, 0.1403486281633377], [-0.09548699855804443, -0.10027485340833664], [-0.054827675223350525, -0.016819223761558533]]], [[[0.0862240344285965, 0.048551395535469055], [0.17063312232494354, -0.07318499684333801], [-0.21814675629138947, 0.09352590143680573]], [[0.11956654489040375, 0.1883874386548996], [0.06825126707553864, -0.2133733630180359], [0.15443511307239532, 0.2016986757516861]], [[0.16470329463481903, 0.13042022287845612], [-0.026639223098754883, 0.04722429811954498], [-0.030448973178863525, -0.14375779032707214]], [[0.15818192064762115, 0.05497334897518158], [0.17179779708385468, -0.0395486056804657], [-0.21229001879692078, 0.1377326399087906]], [[-0.21306011080741882, -0.04352074861526489], [0.03249470889568329, 0.07501937448978424], [-0.01312224566936493, -0.16665296256542206]]], [[[-0.12589189410209656, 0.19939927756786346], [0.16578726470470428, -0.053515225648880005], [-0.1981232911348343, 0.027802646160125732]], [[-0.15982685983181, 0.18705810606479645], [0.061477020382881165, -0.01974363625049591], [0.19126175343990326, 0.07413525879383087]], [[0.11118577420711517, -0.09440584480762482], [-0.06360596418380737, 0.10632188618183136], [-0.1649407148361206, 0.12002821266651154]], [[-0.2131941020488739, 0.15114004909992218], [0.11694170534610748, 0.2034386545419693], [0.12807761132717133, 0.21837098896503448]], [[-0.2125106006860733, 0.1541440635919571], [-0.14631718397140503, 0.16986380517482758], [-0.17452779412269592, -0.08766274154186249]]], [[[-0.04625248908996582, 0.040817275643348694], [0.030222848057746887, 0.15138696134090424], [-0.051083579659461975, -0.01798097789287567]], [[-0.15583977103233337, 0.19559283554553986], [0.0463034063577652, 0.18598677217960358], [0.051012441515922546, -0.17517994344234467]], [[0.0043562352657318115, -0.023692652583122253], [-0.15677320957183838, -0.12245085835456848], [0.00019080936908721924, 0.016388848423957825]], [[-0.15243855118751526, -0.09888984262943268], [-0.1595269739627838, 0.031626179814338684], [0.07695196568965912, 0.14576272666454315]], [[-0.07309457659721375, 0.12547363340854645], [-0.07126447558403015, 0.13446728885173798], [-0.008639395236968994, 0.13147230446338654]]], [[[0.13057009875774384, 0.03413178026676178], [0.21185363829135895, -0.009817391633987427], [-0.13669520616531372, 0.20451919734477997]], [[-0.16998356580734253, 0.049057021737098694], [0.04599924385547638, -0.13267986476421356], [-0.02885471284389496, -0.2000201940536499]], [[-0.1669563502073288, 0.1097056120634079], [0.2145453542470932, 0.10467793047428131], [-0.07123272120952606, 0.09202773869037628]], [[-0.16574031114578247, -0.20113876461982727], [-0.07456432282924652, -0.11747951060533524], [0.07458071410655975, 0.014269053936004639]], [[0.1817319244146347, -0.08313819766044617], [-0.0808459222316742, -0.19869333505630493], [0.12290696799755096, -0.21802327036857605]]]]", "conv").CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            FromNumpyArray("[[-0.04360523819923401, -0.1915765255689621], [-0.11051398515701294, 0.16789337992668152], [0.22713038325309753, 0.004034578800201416], [0.004247337579727173, -0.10147285461425781], [0.3033495843410492, 0.058746516704559326], [-0.12963755428791046, 0.307249516248703], [0.2897814214229584, 0.267313688993454], [0.3706692159175873, 0.2286134660243988], [-0.24051594734191895, 0.1306774914264679], [-0.0883188545703888, -0.37693652510643005], [-0.011823832988739014, -0.1450120359659195], [0.004867136478424072, 0.07239404320716858], [0.11218675971031189, -0.04805883765220642], [0.27923837304115295, 0.11633884906768799], [-0.018213331699371338, 0.17975720763206482], [0.15217456221580505, 0.3249054253101349], [0.1539529263973236, 0.24937674403190613], [0.3424430787563324, 0.23421409726142883], [0.33520612120628357, -0.04466903209686279], [-0.1680470108985901, -0.04061859846115112], [0.3229789435863495, 0.10781127214431763], [0.22962215542793274, -0.021088778972625732], [-0.23311462998390198, -0.3508523106575012], [0.09771737456321716, -0.2954067587852478], [0.06159600615501404, 0.02508920431137085], [-0.1509655863046646, -0.1505151093006134], [0.29415473341941833, 0.2609025537967682], [-0.24004726111888885, -0.029030412435531616], [0.25685539841651917, 0.032179176807403564], [0.11718210577964783, -0.3672073185443878], [0.2595576345920563, -0.26348304748535156], [-0.05063113570213318, 0.06681564450263977], [-0.35544872283935547, -0.3494228422641754], [-0.24743857979774475, 0.15849164128303528], [0.2864411771297455, 0.12451973557472229], [0.116861492395401, 0.3019762933254242], [0.3041217625141144, 0.2814690172672272], [-0.16239939630031586, -0.34676212072372437], [0.19004026055335999, 0.25934621691703796], [0.06760847568511963, -0.09577983617782593]]", "dense").CopyTo(((DenseLayer)network.Layers[4]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.5588785409927368, 0.44112151861190796], [0.4466683566570282, 0.5533316731452942]]");
            TestLossAccuracy(network, X, Y, 0.6938810348510742, 0.5);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.8257191777229309, 0.17428089678287506], [0.8386151790618896, 0.16138479113578796]]");
            TestLossAccuracy(network, X, Y, 0.18375201523303986, 1.0);
        }

        [Test]
        public void TestResNet_Shortcut_Same_Dimension_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_3, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true)
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
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

            //predictions after training
            TestPredict(network, X, "[[0.3923742,0.2287465,0.3788793],[0.3707,0.2305237,0.3987763]]");
            TestLossAccuracy(network, X, Y, 0.927446961402893, 1.0);
        }

        [Test]
        public void TestResNet_Shortcut_Different_Dimension_With_Conv_1x1_to_change_Dimension_NCHW_2_1_4_4()
        {
            var X = FromNumpyArray(X_2_1_4_4, "X");
            var Y = FromNumpyArray(Y_2_3, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true); //left
            network.Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true, 1); //right (identity shortcut)
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
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

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
            var Y = FromNumpyArray(Y_2_3, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            Tensor w = FromNumpyArray("[[[[-0.7714059]]]]", "Convolution0");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[-0.10847557,  0.00658482,  0.41918087],[ 0.5224567 ,  0.42545766, -0.31801027],[-0.28710383, -0.31568986,  0.02822173],[-0.4120677 ,  0.21388823, -0.22343507],[-0.00887001,  0.42890936,  0.00528443],[ 0.14757729, -0.45275694, -0.36124444],[-0.5223615 ,  0.06962186,  0.44158655],[-0.44399977,  0.25540823, -0.35566014],[ 0.31000054,  0.03869426,  0.37737155],[-0.28234982,  0.43704945, -0.08071807],[-0.41145545,  0.41357315,  0.5401688 ],[-0.40983498, -0.47532582, -0.2516185 ],[-0.02894175,  0.07887733, -0.33317018],[ 0.07574445,  0.37989277, -0.47620153],[-0.5085196 ,  0.04452544, -0.4278263 ],[ 0.42463195,  0.26129186, -0.37209088]]", "WeightsDense");
            w.CopyTo(((DenseLayer)network.Layers[2]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.553912222385406,0.135723426938057,0.310364335775375],[0.57789421081543,0.159836992621422,0.26226881146431]]");
            TestLossAccuracy(network, X, Y, null  /*0.9943205118179321*/, 0.5);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

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
            var Y = FromNumpyArray(Y_2_3, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            Tensor w = FromNumpyArray("[[-0.3793878 ,  0.13005257, -0.48190022],[-0.5270703 , -0.5069973 , -0.45630288],[-0.08369148, -0.24146178, -0.09606424],[-0.0498544 , -0.4154459 , -0.3665961 ],[-0.3581952 , -0.3345901 ,  0.48476475],[ 0.320306  ,  0.301827  , -0.48490363],[ 0.33425486, -0.42483532,  0.20156533],[ 0.0346387 ,  0.34260863,  0.45479387],[-0.28320554,  0.27089173, -0.5511215 ],[-0.09140414, -0.2540371 , -0.38209555],[ 0.30901152, -0.22211927, -0.07776272],[-0.01273596, -0.43774882,  0.319129  ],[-0.26144847,  0.45303112, -0.5552845 ],[ 0.0012697 , -0.24624684, -0.01347905],[ 0.18339497, -0.46073103,  0.54499584],[-0.32917506,  0.03634387, -0.5220559 ]]", "Dense0");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.475850999355316,0.251384913921356,0.27276411652565],[0.506687998771667,0.285933136940002,0.207378879189491]]");
            TestLossAccuracy(network, X, Y, null  /*1.4464839696884155*/, 0.5);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

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
            var Y = FromNumpyArray(Y_2_3, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithSGD(0.9, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            Tensor w = FromNumpyArray("[[-0.3793878 ,  0.13005257, -0.48190022],[-0.5270703 , -0.5069973 , -0.45630288],[-0.08369148, -0.24146178, -0.09606424],[-0.0498544 , -0.4154459 , -0.3665961 ],[-0.3581952 , -0.3345901 ,  0.48476475],[ 0.320306  ,  0.301827  , -0.48490363],[ 0.33425486, -0.42483532,  0.20156533],[ 0.0346387 ,  0.34260863,  0.45479387],[-0.28320554,  0.27089173, -0.5511215 ],[-0.09140414, -0.2540371 , -0.38209555],[ 0.30901152, -0.22211927, -0.07776272],[-0.01273596, -0.43774882,  0.319129  ],[-0.26144847,  0.45303112, -0.5552845 ],[ 0.0012697 , -0.24624684, -0.01347905],[ 0.18339497, -0.46073103,  0.54499584],[-0.32917506,  0.03634387, -0.5220559 ]]", "Dense0");
            w.CopyTo(((DenseLayer)network.Layers[1]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.475850999355316,0.251384913921356,0.27276411652565],[0.506687998771667,0.285933136940002,0.207378879189491]]");
            TestLossAccuracy(network, X, Y, null  /* 4.0434770584106445 */, 0.5);

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);

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
            var Y = FromNumpyArray(Y_2_3, "y");
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
            TestNetwork.Fit(network, X, Y, learningRate* batchSize, numEpochs, batchSize);

            //predictions after training
            TestPredict(network, X, "[[0.894426345825195,0.00220351060852408,0.103370226919651],[0.0549939684569836,0.00156258791685104,0.943443357944489]]");
            TestLossAccuracy(network, X, Y, null /*0.4707931876182556*/, 1.0);
        }

        [Test]
        public void TestConcatenate_NCHW_1_1_1_1()
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_1_1_1_1, "X");
            var Y = FromNumpyArray(Y_1_2, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
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
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[9.9985039e-01, 1.4956875e-04]]");
            TestLossAccuracy(network, X, Y, 0.0001495592441642657, 1.0);
        }

        [Test]
        public void TestMultiply_NCHW_1_1_1_1()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(@"numpy.array([[[[1]]]]], numpy.float)", "X_train");
            var Y = FromNumpyArray(@"numpy.array([[1,0]], numpy.float)", "Y_train");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true, 0)
                .MultiplyLayer(2, 1)
                .Flatten()
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            var w = FromNumpyArray("[[[[-0.7714059]]]]", "conv2D");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[[[-1.0248963]]]]", "conv2D_1");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Convolution);
            w = FromNumpyArray("[[-0.24186122,	-0.9861101]]", "dense");
            w.CopyTo(((DenseLayer)network.Layers[5]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.6430006, 0.35699943]]");
            TestLossAccuracy(network, X, Y, 0.44160962104797363, 1.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.80545473, 0.19454528]]");
            TestLossAccuracy(network, X, Y, 0.21634827554225922, 1.0);
        }

        [Test]
        public void TestMultiply_NCHW_1_2_1_1_same_dimension()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(@"[[[[1]],[[0]]]]", "X_train");
            var Y = FromNumpyArray(@"[[1,0]]", "Y_train");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true, 0)
                .MultiplyLayer(2, 1)
                .Flatten()
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            var w = FromNumpyArray("[[[[-0.54546636]],[[-0.12539268]]],[[[-0.72356474]],[[0.26959312]]]]", "conv2D");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[[[-0.7247112]],[[-0.49400187]]],[[[-0.39867145]],[[0.04389346]]]]", "conv2D_1");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Convolution);
            w = FromNumpyArray("[[-0.209458,-0.8539965],[-0.5895995,0.17340887]]", "dense");
            w.CopyTo(((DenseLayer)network.Layers[5]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.50867134,0.4913287]]");
            TestLossAccuracy(network, X, Y, 0.6759531497955322, 1.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.691145,0.30885503]]");
            TestLossAccuracy(network, X, Y, 0.3694056272506714, 1.0);
        }

        [Test]
        public void TestMultiply_NCHW_1_2_1_1_different_dimension()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(@"[[[[1]],[[0]]]]", "X_train");
            var Y = FromNumpyArray(@"[[1,0]]", "Y_train");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true, 0)
                .MultiplyLayer(2, 1)
                .Flatten()
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            var w = FromNumpyArray("[[[[-0.54546636, -0.12539268],[-0.72356474, 0.26959312]]]]", "conv2D");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[[[-0.8368243],[-0.4603461]]]]", "conv2D_1");
            w.CopyTo(((ConvolutionLayer)network.Layers[2]).Convolution);
            w = FromNumpyArray("[[-0.209458  , -0.8539965 ],[-0.5895995, 0.17340887]]", "dense");
            w.CopyTo(((DenseLayer)network.Layers[5]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.45814985,0.54185015]]");
            TestLossAccuracy(network, X, Y, 0.7805589437484741, 0.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.67311347,0.3268865]]");
            TestLossAccuracy(network, X, Y, 0.3958413600921631, 1.0);
        }

        [Test]
        public void TestMultiply_NCHW_2_3_4_5_different_dimension()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_2_3_4_5, "");
            var Y = FromNumpyArray(Y_2_2, "");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true)
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true)
                .GlobalAvgPooling()
                .MultiplyLayer(3, 1)
                .Flatten()
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            FromConvNumpyArray("[[[[-0.4878799319267273, -0.6471760272979736], [-0.11215460300445557, 0.24113142490386963], [-0.5400518774986267, -0.8205036520957947]]]]", "conv").CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            FromConvNumpyArray("[[[[-0.7247111797332764, -0.3986714482307434], [-0.4940018653869629, 0.04389345645904541]]]]", "conv").CopyTo(((ConvolutionLayer)network.Layers[2]).Convolution);
            FromNumpyArray("[[-0.029460519552230835, 0.1628669798374176], [-0.28001704812049866, -0.23855498433113098], [0.07715305685997009, 0.11627233028411865], [0.32925912737846375, 0.011087954044342041], [0.12424156069755554, -0.05900973081588745], [-0.2703372836112976, 0.12233385443687439], [-0.08240920305252075, 0.006095200777053833], [-0.023135006427764893, 0.08786126971244812], [-0.2075882852077484, -0.3384675085544586], [0.10181871056556702, -0.08105111122131348], [0.04287368059158325, -0.014433145523071289], [-0.050517499446868896, 0.19285127520561218], [0.16756221652030945, -0.06256869435310364], [-0.1878374218940735, -0.17477598786354065], [0.3118181526660919, 0.36103251576423645], [0.16790542006492615, 0.27620890736579895], [0.21295377612113953, -0.15440134704113007], [0.03934970498085022, -0.35186851024627686], [-0.19449061155319214, -0.2855254113674164], [-0.08950188755989075, 0.2891680896282196], [-0.37375181913375854, 0.18617329001426697], [0.07124421000480652, 0.28268447518348694], [0.041756272315979004, 0.13584479689598083], [0.12497344613075256, 0.151188462972641], [0.3146173655986786, -0.22298070788383484], [-0.22048203647136688, -0.30460700392723083], [0.12072917819023132, -0.2646358907222748], [-0.15740737318992615, 0.17554828524589539], [0.13976749777793884, -0.357845664024353], [-0.365357369184494, -0.15716126561164856], [0.14519938826560974, 0.22951403260231018], [0.03488221764564514, 0.1870688498020172], [0.28289076685905457, 0.14199396967887878], [0.31583401560783386, 0.08595579862594604], [0.005727171897888184, 0.2800586521625519], [0.013508498668670654, 0.3192369043827057], [-0.14768590033054352, -0.05077126622200012], [-0.28260645270347595, -0.3034713864326477], [-0.05905658006668091, -0.3151003122329712], [-0.12471392750740051, -0.2689373791217804]]", "dense").CopyTo(((DenseLayer)network.Layers[6]).Weights);
            //predictions before training
            TestPredict(network, X, "[[0.48352786898612976, 0.5164721012115479], [0.510468602180481, 0.48953139781951904]]");
            TestLossAccuracy(network, X, Y, 0.6995362043380737, 0.5);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.6233826875686646, 0.37661728262901306], [0.6570475101470947, 0.3429524898529053]]");
            TestLossAccuracy(network, X, Y, 0.4462968111038208, 1.0);
        }

        [Test]
        public void Test_DepthwiseConvolution()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = FromNumpyArray("[[[[0.0,0.1],[0.2,0.3]],[[0.4,0.5],[0.6,0.7]],[[0.8,0.9],[0.95,1.0]]]]", "x");
            var Y = FromNumpyArray(@"[[1,0]]", "Y_train");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .DepthwiseConvolution(3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 1, lambdaL2Regularization, true)
                .Flatten()
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            var w = FromConvNumpyArray("[[[[-1.8182212e-01],[-2.4118826e-01],[-4.1797549e-02]],[[ 8.9864373e-02],[-2.0126545e-01],[-3.0578366e-01]],[[ 3.1133693e-01],[ 2.8887171e-01],[-1.8024862e-01]]],[[[ 2.5361568e-01],[ 2.5364757e-04],[ 3.7667376e-01]],[[ 2.5444084e-01],[-1.6983339e-01],[ 3.4809512e-01]],[[ 1.6018957e-01],[-1.4399531e-01],[ 4.0002191e-01]]],[[[ 6.4367443e-02],[ 3.8731194e-01],[-3.4315154e-01]],[[-1.1633310e-01],[ 1.9787598e-01],[ 3.6543274e-01]],[[-1.5561658e-01],[ 2.6152426e-01],[-1.7792954e-01]]]]", "depthwiseConv2D_1");
            w.CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            w = FromNumpyArray("[[-0.2895949 , -0.1177094 ],[ 0.23789662,  0.3767377 ],[ 0.41499782,  0.2744642 ], [ 0.00143611,  0.4848951 ], [-0.49549383, -0.20193446], [-0.6493831 ,  0.06595588], [-0.28986678,  0.05255955], [-0.27410832, -0.3834508 ], [ 0.05772114, -0.02405858], [ 0.18792003, -0.33073252], [-0.2710427 ,  0.09345931], [ 0.12703902, -0.14722306]]","dense");
            w.CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.529547, 0.470453]]");
            TestLossAccuracy(network, X, Y, 0.635733425617218, 1.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.8828165, 0.11718353]]");
            TestLossAccuracy(network, X, Y, 0.12463792413473129, 1.0);
        }

        [Test]
        public void Test_Convolution_With_Asymmetric_Padding()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_2_3_4_5, "x");
            var Y = FromNumpyArray(Y_2_2, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 3, 2, ConvolutionLayer.PADDING_TYPE.SAME, 0.00, false)
                .Flatten()
                .Output(Y.Shape[1], 0.00, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            FromConvNumpyArray("[[[[-1.8182212e-01],[-2.4118826e-01],[-4.1797549e-02]],[[ 8.9864373e-02],[-2.0126545e-01],[-3.0578366e-01]],[[ 3.1133693e-01],[ 2.8887171e-01],[-1.8024862e-01]]],[[[ 2.5361568e-01],[ 2.5364757e-04],[ 3.7667376e-01]],[[ 2.5444084e-01],[-1.6983339e-01],[ 3.4809512e-01]],[[ 1.6018957e-01],[-1.4399531e-01],[ 4.0002191e-01]]],[[[ 6.4367443e-02],[ 3.8731194e-01],[-3.4315154e-01]],[[-1.1633310e-01],[ 1.9787598e-01],[ 3.6543274e-01]],[[-1.5561658e-01],[ 2.6152426e-01],[-1.7792954e-01]]]]", "conv").CopyTo(((ConvolutionLayer)network.Layers[1]).Convolution);
            FromNumpyArray("[[0.28597003,  0.4547698],[-0.4759524,   0.7416852],[-0.85341465, -0.40610817],[-0.14299387, -0.7595345],[0.08673978, -0.36584526],[0.855876, -0.65572554]]", "dense").CopyTo(((DenseLayer)network.Layers[3]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.95572853, 0.04427144],[0.6475099, 0.35249016]]");
            TestLossAccuracy(network, X, Y, 0.23995131254196167, 1.0);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.98308647, 0.01691353],[0.961277, 0.03872294]]");
            TestLossAccuracy(network, X, Y, 0.028275400400161743, 1.0);
        }

        [Test]
        public void Test_Convolution_With_Asymmetric_Padding_V2()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double momentum = 0.9;
            var X = FromNumpyArray(X_2_3_4_5, "x");
            var Y = FromNumpyArray(Y_2_2, "y");
            var network = GetNetwork(NetworkConfig.LossFunctionEnum.CategoricalCrossentropy);
            network.Config.WithSGD(momentum, false);
            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Convolution(1, 3, 2, ConvolutionLayer.PADDING_TYPE.SAME, 0.00, false)
                .Flatten()
                .Output(Y.Shape[1], 0.00, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            FromConvNumpyArray("[[[[ 0.34814847],[-0.12127715],[ 0.22354162]],[[-0.06822091],[ 0.14409536],[ 0.1851415 ]],[[ 0.34887362],[-0.20124988],[-0.40070006]]],[[[0.00579837],[ 0.10344726],[-0.2527819 ]],[[ 0.39281756],[-0.04241154],[-0.27652574]],[[ 0.00179639],[-0.14511377],[-0.05352649]]],[[[ 0.2575059 ],[ 0.1235916 ],[-0.26898897]],[[ 0.01372808],[-0.22314253],[ 0.3652693 ]],[[ 0.4061154 ],[0.04825488],[ 0.4062844 ]]]]", "conv").CopyTo(((ConvolutionLayer)network.Layers[2]).Convolution);
            FromNumpyArray("[[ 0.1670317  , 0.6739995 ], [ 0.02281129 , 0.36221045], [ 0.7731834  , 0.41565734], [-0.2101801  , 0.21554786], [-0.1369173  , 0.44515973], [-0.18519688 , 0.5447338 ]]", "dense").CopyTo(((DenseLayer)network.Layers[4]).Weights);

            //predictions before training
            TestPredict(network, X, "[[0.69206095, 0.30793908], [0.45456633, 0.54543364]]");
            TestLossAccuracy(network, X, Y, 0.5782463550567627, 0.5);
            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, X.Shape[0]);
            //predictions after training
            TestPredict(network, X, "[[0.97237974, 0.02762033], [0.9205178, 0.0794822 ]]");
            TestLossAccuracy(network, X, Y, 0.05541396513581276, 1.0);
        }

        private static Network GetNetwork(NetworkConfig.LossFunctionEnum lossFunction)
        {
            var gpuDeviceId = -1;
            return new Network(new NetworkConfig{ Logger = Logger.NullLogger, LossFunction = lossFunction, RandomizeOrder = false, ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM, CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1}, gpuDeviceId);
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
            var dataSet = new InMemoryDataSet(X, Y_expected, new int[batchSize], "", null);
            var observedLossAccuracy = network.ComputeLossAndAccuracyForTestDataSet(batchSize, dataSet);
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

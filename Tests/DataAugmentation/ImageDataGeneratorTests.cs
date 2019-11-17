//TO REMOVE
//using System.Linq;
//using NUnit.Framework;
//using SharpNet.CPU;
//using SharpNet.DataAugmentation;
//using SharpNetTests.Data;

//// ReSharper disable RedundantTypeSpecificationInDefaultExpression

//namespace SharpNetTests.DataAugmentation
//{
//    [TestFixture]
//    public class ImageDataGeneratorTests
//    {
//        [Test]

//        public void Test_InitializeOutputPicture()
//        {
//            var input = Enumerable.Range(0, 16).Select(x=>(float)x).ToArray();
//            var inputShape = new[] {1,1,4, 4};
//            var expected = (float[]) input.Clone();
            
//            //no changes
//            Test_InitializeOutputPicture(input, inputShape, expected,0, 0, ImageDataGenerator.FillModeEnum.Nearest,false, false, 1.0, 1.0,-1, -1, -1, -1, -1,-1,-1,-1,-1, null, -1, 0.0);

//            //flip
//            //horizontal flip
//            expected = new [] {3f,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12};
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, true, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0);
//            //vertical flip
//            expected = new[] { 12f,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3};
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, true, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0);
//            //horizontal & vertical flip
//            expected = new[] { 15f,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, true, true, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0);

//            //cutout
//            //full cutout
//            expected = Enumerable.Repeat(default(float),16).ToArray();
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, 0, 3, 0, 3, -1, -1, -1, -1, -1, null, -1, 0);
//            //single element cutout
//            expected = new[] { 0f,1,2,3, 4,5,default(int),7 ,8,9,10,11 ,12,13,14,15 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, 1, 1, 2, 2, -1, -1, -1, -1, -1, null, -1, 0);
//            //single row cutout
//            expected = new[] { 0f, 1, 2, 3, default(int), default(int), default(int), default(int), 8, 9, 10, 11, 12, 13, 14, 15 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, 1, 1, 0, 3, -1, -1, -1, -1, -1, null, -1, 0);
//            //single col cutout
//            expected = new[] { 0f, 1, default(int),3, 4,5,default(int),7, 8,9,default(int), 11, 12,13,default(int), 15 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, 0, 3, 2, 2, -1, -1, -1, -1, -1, null, -1, 0);
//            //top right cutout
//            expected = new[] { 0f, 1, default(int), default(int), 4, 5, default(int), default(int), 8, 9, 10, 11, 12, 13, 14, 15 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, 0, 1, 2, 3, -1, -1, -1, -1, -1, null, -1, 0);

//            //shift
//            //shift with FillModeEnum.Nearest
//            //1 to the right
//            expected = new[] { 0f,0,1,2, 4,4,5,6, 8,8,9,10, 12,12,13,14 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 1, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0);
//            //2 to the left
//            expected = new[] { 2f,3,3,3, 6,7,7,7, 10,11,11,11, 14,15,15,15 };
//            Test_InitializeOutputPicture(input, inputShape, expected, -2, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0);
//            //3 to the top
//            expected = new[] { 12f,13,14,15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, -3, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0);
//            //1 to the right & 1 to the bottom
//            expected = new[] { 0f,0,1,2, 0,0,1,2, 4,4,5,6, 8,8,9,10 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 1, 1, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0);
//            //sift with FillModeEnum.Reflect
//            //1 to the right
//            expected = new[] { 0f, 0, 1, 2, 4, 4, 5, 6, 8, 8, 9, 10, 12, 12, 13, 14 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 1, 0, ImageDataGenerator.FillModeEnum.Reflect, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0);
//            //2 to the left
//            expected = new[] { 2f, 3, 3, 2, 6, 7, 7, 6, 10, 11, 11, 10, 14, 15, 15, 14 };
//            Test_InitializeOutputPicture(input, inputShape, expected, -2, 0, ImageDataGenerator.FillModeEnum.Reflect, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0);
//            //3 to the top
//            expected = new[] { 12f, 13, 14, 15, 12, 13, 14, 15, 8,9,10,11, 4,5,6,7 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, -3, ImageDataGenerator.FillModeEnum.Reflect, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0);
//            //2 to the right & 2 to the bottom
//            expected = new[] { 5f,4,4,5, 1,0,0,1, 1,0,0,1,  5,4,4,5 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 2, 2, ImageDataGenerator.FillModeEnum.Reflect, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0);

//            //rotations
//            //360° rotation = no rotation
//            expected = new[] { 0f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 360.0);
//            //anti clockwise rotation of 90° (90.0)
//            expected = new[] { 3f,7,11,15, 2,6,10,14, 1,5,9,13, 0,4,8,12 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 90.0);
//            //rotation of 180° 
//            expected = new[] { 15f,14,13,12, 11,10,9,8, 7,6,5,4, 3,2,1,0 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 180.0);
//            //rotation of -180° 
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, -180.0);
//            //clockwise rotation of 90° (-90.0)
//            expected = new[] { 12f,8,4,0, 13,9,5,1, 14,10,6,2, 15,11,7,3 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, -90.0);

//            //zoom
//            //2* width zoom
//            expected = new[] { 0f,0,1,1, 4,4,5,5, 8,8,9,9, 12,12,13,13};
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 2.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0.0);
//            //2* height zoom
//            expected = new[] { 0f,1,2,3, 0,1,2,3, 4,5,6,7, 4,5,6,7 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 2.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0.0);
//            //2*width zoom and 2*height zoom
//            expected = new[] { 0f,0,1,1, 0,0,1,1, 4,4,5,5, 4,4,5,5 };
//            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 2.0, 2.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, null, -1, 0.0);

//            //cutMix
//            var inputShapeCutMix = new[] { 2, 1, 2, 4 };
//            //no cutMix
//            expected = new[] { 0f, 1, 2, 3, 4, 5, 6, 7, 0,0,0,0,0,0,0,0 };
//            Test_InitializeOutputPicture(input, inputShapeCutMix, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1,-1,-1,-1,-1, null, -1, 0);
//            //right side of 2nd picture into right side of 1st picture
//            expected = new[] { 0f, 1, 10,11, 4, 5, 14,15, 0, 0, 0, 0, 0, 0, 0, 0 };
//            Test_InitializeOutputPicture(input, inputShapeCutMix, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, 0, 1, 2, 3, 1, null, -1, 0);
//            //all of 2nd picture into 1st picture
//            expected = new[] { 8f,9,10,11,12,13,14,15, 0, 0, 0, 0, 0, 0, 0, 0 };
//            Test_InitializeOutputPicture(input, inputShapeCutMix, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, 0, 1, 0,3, 1, null, -1, 0);
//            //1 pixel of 2nd picture into 1st picture
//            expected = new[] { 0f, 9, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0 };
//            Test_InitializeOutputPicture(input, inputShapeCutMix, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, 0, 0, 1, 1, 1, null, -1, 0);

//            //cutMix (copy right side of 2nd picture into 1st picture) + cutout (middle of 1st picture)
//            //the CutMix must be performed before the cutout
//            expected = new [] { 0f, 0, 10, 11, 4, 0, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0 };
//            Test_InitializeOutputPicture(input, inputShapeCutMix, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, 0,1,1,2, 0,1,2,3, 1, null, -1, 0.0);


//            //Mixup (mix 2nd picture into 1st picture)
//            //the CutMix must be performed before the cutout
//            expected = new[] { 4f, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0 };
//            Test_InitializeOutputPicture(input, inputShapeCutMix, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.5f, 1, 0.0);
//            expected = new[] { 0.25f*0+0.75f*8, 0.25f*1+0.75f*9, 0.25f*2+0.75f*10, 0.25f*3+0.75f*11,  0.25f*4+0.75f*12, 0.25f*5+0.75f*13, 0.25f*6+0.75f*14, 0.25f*7+0.75f*15, 0, 0, 0, 0, 0, 0, 0, 0 };
//            Test_InitializeOutputPicture(input, inputShapeCutMix, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1.0, 1.0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.25f, 1, 0.0);

//        }

//        private static void Test_InitializeOutputPicture(float[] input, int[] inputShape, float[] expectedOutput,
//            int widthShift, int heightShift, ImageDataGenerator.FillModeEnum _fillMode,
//            bool horizontalFlip, bool verticalFlip,
//            double widthMultiplier, double heightMultiplier,
//            int cutoutRowStart, int cutoutRowEnd, int cutoutColStart, int cutoutColEnd,
//            int cutMixRowStart, int cutMixRowEnd, int cutMixColStart, int cutMixColEnd, int inputPictureIndexForCutMix,
//            float? mixupLambda, int inputPictureIndexForMixup,
//            double rotationAngleInRadians)
//        {
//            var inputTensor = new CpuTensor<float>(inputShape, input, "inputTensor");
//            var expectedOutputTensor = new CpuTensor<float>(inputShape, expectedOutput, "expectedOutput");
//            var bufferOutputTensor = new CpuTensor<float>(inputShape, "bufferOutputTensor");

//            ImageDataGenerator.InitializeOutputPicture(inputTensor, bufferOutputTensor, 0, 
//                widthShift, heightShift, _fillMode,
//                horizontalFlip, verticalFlip,
//                widthMultiplier, heightMultiplier,
//                cutoutRowStart, cutoutRowEnd, cutoutColStart, cutoutColEnd,
//                cutMixRowStart, cutMixRowEnd, cutMixColStart, cutMixColEnd, inputPictureIndexForCutMix,
//                mixupLambda, inputPictureIndexForMixup,
//                rotationAngleInRadians);
//            Assert.IsTrue(TestTensor.SameContent(expectedOutputTensor, bufferOutputTensor, 1e-6));
//        }
//    }
//}

using System.Linq;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Pictures;
using SharpNetTests.Data;


namespace SharpNetTests.Pictures
{
    [TestFixture]
    public class ImageDataGeneratorTests
    {
        [Test]

        public void Test_InitializeOutputPicture()
        {
            var input = Enumerable.Range(0, 16).ToArray();
            var inputShape = new[] {1,1,4, 4};
            var expected = (int[]) input.Clone();
            
            //no changes
            Test_InitializeOutputPicture(input, inputShape, expected,0, 0, ImageDataGenerator.FillModeEnum.Nearest,false, false,-1, -1, -1, -1, 0);

            //flip
            //horizontal flip
            expected = new [] {3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12};
            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, true, false, -1, -1, -1, -1, 0);
            //vertical flip
            expected = new[] { 12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3};
            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, true, -1, -1, -1, -1, 0);
            //horizontal & vertical flip
            expected = new[] { 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, true, true, -1, -1, -1, -1, 0);

            //cutout
            //full cutout
            expected = Enumerable.Repeat(default(int),16).ToArray();
            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 0, 3, 0, 3, 0);
            //single element cutout
            expected = new[] { 0,1,2,3, 4,5,default(int),7 ,8,9,10,11 ,12,13,14,15 };
            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1, 1, 2, 2, 0);
            //single row cutout
            expected = new[] { 0, 1, 2, 3, default(int), default(int), default(int), default(int), 8, 9, 10, 11, 12, 13, 14, 15 };
            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 1, 1, 0, 3, 0);
            //single col cutout
            expected = new[] { 0, 1, default(int),3, 4,5,default(int),7, 8,9,default(int), 11, 12,13,default(int), 15 };
            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 0, 3, 2, 2, 0);
            //top right cutout
            expected = new[] { 0, 1, default(int), default(int), 4, 5, default(int), default(int), 8, 9, 10, 11, 12, 13, 14, 15 };
            Test_InitializeOutputPicture(input, inputShape, expected, 0, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, 0, 1, 2, 3, 0);

            //move with nearest
            //1 to the right
            expected = new[] { 0,0,1,2, 4,4,5,6, 8,8,9,10, 12,12,13,14 };
            Test_InitializeOutputPicture(input, inputShape, expected, 0, 1, ImageDataGenerator.FillModeEnum.Nearest, false, false, -1, -1, -1, -1, 0);
            //2 to the left
            expected = new[] { 2,3,3,3, 6,7,7,7, 10,11,11,11, 14,15,15,15 };
            Test_InitializeOutputPicture(input, inputShape, expected, 0, -2, ImageDataGenerator.FillModeEnum.Nearest, false, false, -1, -1, -1, -1, 0);
            //3 to the top
            expected = new[] { 12,13,14,15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, };
            Test_InitializeOutputPicture(input, inputShape, expected, -3, 0, ImageDataGenerator.FillModeEnum.Nearest, false, false, -1, -1, -1, -1, 0);
            //1 to the right & 1 to the bottom
            expected = new[] { 0,0,1,2, 0,0,1,2, 4,4,5,6, 8,8,9,10 };
            Test_InitializeOutputPicture(input, inputShape, expected, 1, 1, ImageDataGenerator.FillModeEnum.Nearest, false, false, -1, -1, -1, -1, 0);

            //move with reflect
            //1 to the right
            expected = new[] { 0, 0, 1, 2, 4, 4, 5, 6, 8, 8, 9, 10, 12, 12, 13, 14 };
            Test_InitializeOutputPicture(input, inputShape, expected, 0, 1, ImageDataGenerator.FillModeEnum.Reflect, false, false, -1, -1, -1, -1, 0);
            //2 to the left
            expected = new[] { 2, 3, 3, 2, 6, 7, 7, 6, 10, 11, 11, 10, 14, 15, 15, 14 };
            Test_InitializeOutputPicture(input, inputShape, expected, 0, -2, ImageDataGenerator.FillModeEnum.Reflect, false, false, -1, -1, -1, -1, 0);
            //3 to the top
            expected = new[] { 12, 13, 14, 15, 12, 13, 14, 15, 8,9,10,11, 4,5,6,7 };
            Test_InitializeOutputPicture(input, inputShape, expected, -3, 0, ImageDataGenerator.FillModeEnum.Reflect, false, false, -1, -1, -1, -1, 0);
            //2 to the right & 2 to the bottom
            expected = new[] { 5,4,4,5, 1,0,0,1, 1,0,0,1,  5,4,4,5 };
            Test_InitializeOutputPicture(input, inputShape, expected, 2, 2, ImageDataGenerator.FillModeEnum.Reflect, false, false, -1, -1, -1, -1, 0);
        }

        private static void Test_InitializeOutputPicture(int[] input, int[] inputShape, int[] expectedOutput,
            int deltaRowInput, int deltaColInput, ImageDataGenerator.FillModeEnum _fillMode,
            bool horizontalFlip, bool verticalFlip,
            int cutoutRowStart, int cutoutRowEnd, int cutoutColStart, int cutoutColEnd, 
            double rotationAngleInRadians)
        {
            var inputTensor = new CpuTensor<int>(inputShape, input, "inputTensor");
            var expectedOutputTensor = new CpuTensor<int>(inputShape, expectedOutput, "expectedOutput");
            var bufferOutputTensor = new CpuTensor<int>(inputShape, "bufferOutputTensor");

            ImageDataGenerator.InitializeOutputPicture(inputTensor, 0, bufferOutputTensor, 0, 
                deltaRowInput, deltaColInput, _fillMode,
                horizontalFlip, verticalFlip,
                cutoutRowStart, cutoutRowEnd, cutoutColStart, cutoutColEnd, 
                rotationAngleInRadians);
            Assert.IsTrue(TestTensor.SameContent(expectedOutputTensor, bufferOutputTensor));
        }
    }
}

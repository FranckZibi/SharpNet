using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.Networks;

namespace SharpNetTests.DataAugmentation
{
    [TestFixture]
    public class DataAugmentationConfigTests
    {
        [Test]
        public void TestSerialize()
        {
            var da = new NetworkSample();
            da.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10;
            da.WidthShiftRangeInPercentage = 0.15;
            da.HeightShiftRangeInPercentage = 0.3;
            da.HorizontalFlip = true;
            da.VerticalFlip = false;
            da.FillMode = ImageDataGenerator.FillModeEnum.Reflect;
            da.FillModeConstantVal = 0.45;
            da.CutoutPatchPercentage = 0.6;
            da.AlphaCutMix = 0.75;
            da.AlphaMixup = 0.9;
            da.RotationRangeInDegrees = 1.05;
            da.ZoomRange = 1.2;

            //TODO
            //var serialized = da.Serialize();
            //var deserialized = DataAugmentationConfig.ValueOf(Serializer.Deserialize(serialized));
            //var errors = "";
            //var equals = da.Equals(deserialized, 1e-6, "id", ref errors);
            //Assert.IsTrue(equals, errors);
            //deserialized.VerticalFlip = true;
            //equals = da.Equals(deserialized, 1e-6, "id", ref errors);
            //Assert.IsFalse(equals, "VerticalFlip is different");
        }
    }
}
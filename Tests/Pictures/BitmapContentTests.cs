using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.Pictures;

namespace SharpNetTests.Pictures
{
    [TestFixture]
    public class BitmapContentTests
    {
        [Test]
        public void TestUpdateWith_Sum_SumSquare_Count_For_Each_Channel()
        {
            const float epsilon = 1e-6f;
            var bc = new BitmapContent(new[]{1,1,1}, new byte[]{0});
            var _sum_SumSquare_Count_For_Each_Channel = new float[3 * bc.GetChannels()];
            bc.UpdateWith_Sum_SumSquare_Count_For_Each_Channel(_sum_SumSquare_Count_For_Each_Channel);
            AssertAreEqual(new float[]{0,0,1}, _sum_SumSquare_Count_For_Each_Channel, epsilon);

            bc = new BitmapContent(new[] { 3, 2, 2 }, new byte[] { 0,0,0,0 ,1,1,0,0, 2,0,2,0 });
            _sum_SumSquare_Count_For_Each_Channel = new float[3 * bc.GetChannels()];
            bc.UpdateWith_Sum_SumSquare_Count_For_Each_Channel(_sum_SumSquare_Count_For_Each_Channel);
            AssertAreEqual(new float[] { 0,0,4, 2,2,4, 4,8,4 }, _sum_SumSquare_Count_For_Each_Channel, epsilon);
        }

        [Test]
        public void TestMakeSquarePictures()
        {
            var fillingColor = Tuple.Create((byte)253, (byte)254, (byte)255);

            var bc = new BitmapContent(new[] { 3, 1, 1 }, new byte[] { 117,1,2 });
            var bcSquare = bc.MakeSquarePictures(true, false, fillingColor);
            Assert.IsTrue(bc.Shape.SequenceEqual(bcSquare.Shape));
            AssertAreEqual(new byte[] {117,1,2}, bcSquare.SpanContent.ToArray());
            bcSquare = bc.MakeSquarePictures(false, false, fillingColor);
            Assert.IsTrue(bc.Shape.SequenceEqual(bcSquare.Shape));
            AssertAreEqual(new byte[] { 117, 1, 2 }, bcSquare.SpanContent.ToArray());
            bcSquare = bc.MakeSquarePictures(false, true, fillingColor);
            Assert.IsTrue(bc.Shape.SequenceEqual(bcSquare.Shape));
            AssertAreEqual(new byte[] { 117, 1, 2 }, bcSquare.SpanContent.ToArray());

            bc = new BitmapContent(new[] { 3, 3, 1 }, new byte[] { 117,1,2, 3,4,5, 6,7,8 });
            bcSquare = bc.MakeSquarePictures(false, false, fillingColor);
            Assert.IsTrue(new[]{3,3,3}.SequenceEqual(bcSquare.Shape));
            AssertAreEqual(new byte[]
                           {
                               253,117,253,253,1,253,253,2,253,
                               254,3,254,254,4,254,254,5,254,
                               255,6,255,255,7,255,255,8,255,
                           }, bcSquare.SpanContent.ToArray());
            bcSquare = bc.MakeSquarePictures(false, true, fillingColor);
            Assert.IsTrue(new[] { 3, 1, 1 }.SequenceEqual(bcSquare.Shape));
            AssertAreEqual(new byte[]{1,4,7}, bcSquare.SpanContent.ToArray());
            bcSquare = bc.MakeSquarePictures(true, false, fillingColor);
            Assert.IsTrue(new[] { 3, 3, 3 }.SequenceEqual(bcSquare.Shape));
            AssertAreEqual(new byte[]
                           {
                               253,253,253,117,1,2,253,253,253,
                               254,254,254,3,4,5,254,254,254,
                               255,255,255,6,7,8,255,255,255,
                           }, bcSquare.SpanContent.ToArray());
        }

        private static void AssertAreEqual(float[] expected, float[] observed, float epsilon )
        {
            Assert.AreEqual(expected.Length, observed.Length);
            for(int i=0;i<expected.Length;++i)
            {
                Assert.AreEqual(expected[i], observed[i], epsilon);
            }
        }
        private static void AssertAreEqual(byte[] expected, byte[] observed)
        {
            Assert.AreEqual(expected.Length, observed.Length);
            for (int i = 0; i < expected.Length; ++i)
            {
                Assert.AreEqual(expected[i], observed[i]);
            }
        }
    }
}

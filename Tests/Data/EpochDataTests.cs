using System.Linq;
using NUnit.Framework;
using SharpNet.Data;

namespace SharpNetTests.Data
{
    [TestFixture]
    public class EpochDataTests
    {
        [Test]
        public void SerializationTest()
        {
            var epochData1 = new EpochData(1,2.0,3.0,4.0,5.0,6.0,7.0,0);
            var epochData2 = new EpochData(11,12.0,13.0,14.0,15.0,16.0,17.0, 0);
            var epochData3 = new EpochData(21,22.0,23.0,24.0,25.0,26.0,27.0, 0);
            var epochDatas = new [] {epochData1, epochData2, epochData3};
            var serialized = new Serializer().Add("array", epochDatas).ToString();
            var deserialized = Serializer.Deserialize(serialized).Values.First() as EpochData[];
            Assert.IsNotNull(deserialized);
            Assert.AreEqual(3, deserialized.Length);
            for (var i = 0; i < epochDatas.Length; i++)
            {
                Assert.IsTrue(epochDatas[i].Equals(deserialized[i], 1e-8));
            }
        }
    }
}
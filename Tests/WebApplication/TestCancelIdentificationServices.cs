using System.Text;
using NUnit.Framework;
using SharpNetWebApplication.Models;

namespace SharpNetTests.WebApplication
{
    [TestFixture]
    public class TestCancelIdentificationServices
    {
        [Test]
        public void TestComputeIdReturnsSha256HexOfContent()
        {
            //well known SHA-256 test vector for the ASCII string "abc"
            var id = CancelIdentificationServices.ComputeId(Encoding.ASCII.GetBytes("abc"));
            Assert.AreEqual("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", id);
        }

        [Test]
        public void TestComputeIdIsDeterministic()
        {
            var content = new byte[] { 1, 2, 3, 4, 5 };
            Assert.AreEqual(CancelIdentificationServices.ComputeId(content), CancelIdentificationServices.ComputeId((byte[])content.Clone()));
        }

        [Test]
        public void TestComputeIdChangesWhenContentChanges()
        {
            Assert.AreNotEqual(CancelIdentificationServices.ComputeId(new byte[] { 1, 2, 3 }), CancelIdentificationServices.ComputeId(new byte[] { 1, 2, 4 }));
        }
    }
}

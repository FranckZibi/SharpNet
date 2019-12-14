using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.DataAugmentation.Operations;
using SharpNetTests.CPU;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class OperationHelperTests
    {
        [Test]
        public void CheckIntegrityTest()
        {
            var rand = new Random(0);
            var shape = new [] {10, 3, 32, 32};
            var xOriginalMiniBatch = TestCpuTensor.RandomFloatTensor(shape, rand, -1, 1, "");
            var cutout = new Cutout(0, 0, 10, 10);
            var mixup = new Mixup(1, 0, xOriginalMiniBatch);
            var cutMix = new CutMix(0,0,10,10,0,xOriginalMiniBatch);
            var invert = new Invert(null);
            var hFlip = new HorizontalFlip(100);
            var vFlip = new VerticalFlip(100);
            var translateX = new TranslateX(5);
            var translateY = new TranslateY(5);
            var equalize = new Equalize(new List<int[]>(), null);

            //no operations at all is allowed
            var empty = new List<Operation>();
            Assert.DoesNotThrow(()=>OperationHelper.CheckIntegrity(empty));

            //no duplicate operations (apart from Translate / Equalize
            var operations = new List<Operation> { vFlip, invert, hFlip, cutout };
            Assert.DoesNotThrow(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { translateX, vFlip, translateX, invert, hFlip, cutout };
            Assert.DoesNotThrow(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { translateX, vFlip, translateY, invert, translateY, hFlip, cutout };
            Assert.DoesNotThrow(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { equalize, vFlip, translateY, invert, equalize, hFlip, cutout };
            Assert.DoesNotThrow(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { vFlip, invert, hFlip, invert, cutout };
            Assert.Throws<ArgumentException>(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { vFlip, invert, hFlip, hFlip, cutout };
            Assert.Throws<ArgumentException>(() => OperationHelper.CheckIntegrity(operations));

            //cutout is allowed only as last param
            operations = new List<Operation>{invert, hFlip, cutout};
            Assert.DoesNotThrow(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { invert, cutout, hFlip};
            Assert.Throws<ArgumentException>(() => OperationHelper.CheckIntegrity(operations));

            //cutMix and mixup are not allowed at same time
            operations = new List<Operation> { invert, hFlip, mixup };
            Assert.DoesNotThrow(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { invert, hFlip, cutMix };
            Assert.DoesNotThrow(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { invert, hFlip, mixup, cutMix };
            Assert.Throws<ArgumentException>(() => OperationHelper.CheckIntegrity(operations));

            //cutMix must be the last operation (if no Cutout operation is used) or the operation just before Cutout (if Cutout is used)
            operations = new List<Operation> { invert, hFlip, vFlip, cutMix, cutout };
            Assert.DoesNotThrow(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { invert, hFlip, vFlip, cutMix};
            Assert.DoesNotThrow(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { invert, hFlip, cutMix, vFlip, cutout };
            Assert.Throws<ArgumentException>(() => OperationHelper.CheckIntegrity(operations));

            //Mixup must be the last operation (if no Cutout operation is used) or the operation just before Cutout (if Cutout is used)
            operations = new List<Operation> { invert, hFlip, vFlip, mixup, cutout };
            Assert.DoesNotThrow(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { invert, hFlip, vFlip, mixup };
            Assert.DoesNotThrow(() => OperationHelper.CheckIntegrity(operations));
            operations = new List<Operation> { invert, hFlip, mixup, vFlip, cutout };
            Assert.Throws<ArgumentException>(() => OperationHelper.CheckIntegrity(operations));
        }
    }
}
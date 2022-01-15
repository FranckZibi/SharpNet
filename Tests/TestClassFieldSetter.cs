using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet;
// ReSharper disable FieldCanBeMadeReadOnly.Local
// ReSharper disable ConvertToConstant.Local
// ReSharper disable NonReadonlyMemberInGetHashCode

namespace SharpNetTests
{
    [TestFixture]
    public class TestClassFieldSetter
    {
        private enum NotUsedEnum {A,B,C};

        private class TestClass
        {
            public bool Bool = true;
            public int Int = 42;
            public float Float = 42;
            public double Double = 42;
            public string String = "42";
            public NotUsedEnum Enum = NotUsedEnum.C;

            public override int GetHashCode()
            {
                return HashCode.Combine(Bool, Int, Float, Double, String, (int)Enum);
            }
            public override bool Equals(object obj)
            {
                if (ReferenceEquals(null, obj))
                {
                    return false;
                }

                if (ReferenceEquals(this, obj))
                {
                    return true;
                }

                if (obj.GetType() != this.GetType())
                {
                    return false;
                }

                return Equals((TestClass)obj);
            }

            private bool Equals(TestClass other)
            {
                return Bool == other.Bool 
                       && Int == other.Int 
                       && Math.Abs(Float - other.Float) <1e-6
                       && Math.Abs(Double-other.Double) <1e-6
                       && String == other.String 
                       && Enum == other.Enum;
            }
        }

        [Test]
        public void TestSet()
        {
            var p = new TestClass();

            ClassFieldSetter.Set(p, "Bool", false);
            Assert.AreEqual(false, p.Bool);
            ClassFieldSetter.Set(p, "Int", 41);
            Assert.AreEqual(41, p.Int);
            ClassFieldSetter.Set(p, "Float", 40);
            Assert.AreEqual(40, p.Float, 1e-6);
            ClassFieldSetter.Set(p, "Double", 39);
            Assert.AreEqual(39, p.Double, 1e-6);
            ClassFieldSetter.Set(p, "String", "38");
            Assert.AreEqual("38", p.String);
            ClassFieldSetter.Set(p, "Enum", "B");
            Assert.AreEqual(NotUsedEnum.B, p.Enum);
            ClassFieldSetter.Set(p, "Enum", NotUsedEnum.A);
            Assert.AreEqual(NotUsedEnum.A, p.Enum);
        }

        [Test]
        public void TestGet()
        {
            var p = new TestClass();

            p.Bool = false;
            Assert.AreEqual(false, ClassFieldSetter.Get(p, "Bool"));
            p.Int = 41;
            Assert.AreEqual(41, ClassFieldSetter.Get(p, "Int"));
            p.Float = 40;
            Assert.AreEqual(40, ClassFieldSetter.Get(p, "Float"));
            p.Double = 39;
            Assert.AreEqual(39, ClassFieldSetter.Get(p, "Double"));
            p.String= "38";
            Assert.AreEqual("38", ClassFieldSetter.Get(p, "String"));
            p.Enum = NotUsedEnum.A;
            Assert.AreEqual(NotUsedEnum.A, ClassFieldSetter.Get(p, "Enum"));
        }

        [Test]
        public void TestToConfigContent()
        {
            var p = new TestClass();

            // with 'ignoreDefaultValue' to true
            var res1 = ClassFieldSetter.ToConfigContent(p, true).Trim();
            Assert.AreEqual("", res1);
            var mandatoryParametersInConfigFile = new HashSet<string> { "Bool", "Float" };
            res1 = ClassFieldSetter.ToConfigContent(p, true, mandatoryParametersInConfigFile).Trim();
            Assert.AreEqual("Bool = True" + Environment.NewLine + "Float = 42", res1);
            p.Float = 41;
            res1 = ClassFieldSetter.ToConfigContent(p, true).Trim();
            Assert.AreEqual("Float = 41", res1);
            res1 = ClassFieldSetter.ToConfigContent(p, true, mandatoryParametersInConfigFile).Trim();
            Assert.AreEqual("Bool = True" + Environment.NewLine + "Float = 41", res1);

            // with 'ignoreDefaultValue' to false
            var res2 = ClassFieldSetter.ToConfigContent(p, false).Trim();
            var expected = "Bool = True" + Environment.NewLine
                                         + "Double = 42" + Environment.NewLine
                                         + "Enum = C" + Environment.NewLine
                                         + "Float = 41" + Environment.NewLine
                                         + "Int = 42" + Environment.NewLine
                                         + "String = 42";
            Assert.AreEqual(expected, res2);
        }
    }
}

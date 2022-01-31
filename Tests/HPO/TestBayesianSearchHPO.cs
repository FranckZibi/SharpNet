using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using SharpNet.Datasets.Natixis70;
using SharpNet.HPO;
using static SharpNet.HPO.AbstractHyperParameterSearchSpace;

// ReSharper disable FieldCanBeMadeReadOnly.Local
// ReSharper disable MemberCanBePrivate.Local
// ReSharper disable ConvertToConstant.Local

namespace SharpNetTests.HPO;

[TestFixture]

public class TestBayesianSearchHPO
{
    private class TempClass
    {
        public float A = 0;
        public float B = 0;
        public int C = 0;
        public float D = 0;
        public float E = 0;
        public float G = 0;

        public float Cost()
        {
            var cost = A * A
                       + Math.Pow(B - 1, 2)
                       + Math.Abs(C - 2)
                       + Math.Pow(D - 3, 2)
                       + Math.Abs(E - 4)
                       + Math.Abs((E - 4) * (D - 3))
                       + Math.Abs(G);  
            return (float)cost;
        }

        public override string ToString()
        {
            return   "A:"+A
                   + " B:" + B
                   + " C:" + C
                   + " D:" + D
                   + " E:" + E
                   + " G:" + G;
        }
    }



    [Test, Explicit]
    public void TestConvergence()
    {
     
        var testDirectory = Path.Combine(Natixis70Utils.WorkingDirectory, "Natixis70", "Temp");
        var searchSpace = new Dictionary<string, object>
        {
            { "A", Range(-10f, 10f) },
            { "B", Range(-10f, 10f) },
            { "C", Range(2, 2) },
            { "D", Range(-10f, 10f) },
            { "E", Range(-10f, 10f) },
            { "G", Range(0f, 0f) },
        };
        var hpo = new BayesianSearchHPO<TempClass>(searchSpace, 
            () => new TempClass(), 
            _ => true,
            60, 
            RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING, 
            testDirectory, 
            new HashSet<string>());
        hpo.Process(t => t.Cost());
    }
}
using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using SharpNet;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
using static SharpNet.HPO.HyperparameterSearchSpace;

// ReSharper disable FieldCanBeMadeReadOnly.Local
// ReSharper disable MemberCanBePrivate.Local
// ReSharper disable ConvertToConstant.Local

namespace SharpNetTests.HPO;

[TestFixture]

public class TestBayesianSearchHPO
{
    private class TempClass : AbstractSample
    {
        // ReSharper disable once EmptyConstructor
        public TempClass()
        {
        }

        public float A = 0;
        public float B = 0;
        public int C = 0;
        public float D = 0;
        public float E = 0;
        public float G = 0;

        public IScore Cost()
        {
            var cost = A * A
                       + MathF.Pow(B - 1, 2)
                       + MathF.Abs(C - 2)
                       + MathF.Pow(D - 3, 2)
                       + MathF.Abs(E - 4)
                       + MathF.Abs((E - 4) * (D - 3))
                       + MathF.Abs(G);  
            return new Score(cost,EvaluationMetricEnum.Mae);
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
        var testDirectory = Path.Combine(Utils.ChallengesPath, "Natixis70", "Temp");
        var searchSpace = new Dictionary<string, object>
        {
            { "A", Range(-10f, 10f) },
            { "B", Range(-10f, 10f) },
            { "C", Range(2, 2) },
            { "D", Range(-10f, 10f) },
            { "E", Range(-10f, 10f) },
            { "G", Range(0f, 0f) },
        };
        var hpo = new BayesianSearchHPO(searchSpace, () => new TempClass(), testDirectory);
        hpo.Process(t => ((TempClass)t).Cost(), 60 );
    }
}
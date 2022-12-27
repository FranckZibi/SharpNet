using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNetTests.Data;

namespace SharpNetTests.Datasets;

[TestFixture]
public class TesDataFrame
{
    [Test]
    public void TestSumOrAvgForColumns()
    {
        var columns = new[] { "id", "0", "1" };
        var content = new[] {
            1, 0.1f, 0.2f,
            0, 0.2f, 0.3f,
            0, 0.3f, 0.2f,
            2, 5.0f, 6.0f,
            1, 0.1f, 0.1f,
            1, 0.1f, 0.0f,
        };
        var df1 = DataFrame.New(content, columns);

        var df_id_avg = df1.AverageBy("id");
        Assert.AreEqual(df_id_avg.Columns, columns);
        var expectedTensor =  new CpuTensor<float>(new[] { 3, 3 }, new [] { 1, 0.1f, 0.1f, 0, 0.25f, 0.25f, 2, 5f, 6f });
        Assert.IsTrue(TestTensor.SameContent(df_id_avg.FloatCpuTensor(), expectedTensor, 1e-5));

        var df_id_sum = df1.SumBy("id");
        Assert.AreEqual(df_id_sum.Columns, columns);
        expectedTensor = new CpuTensor<float>(new[] { 3, 3 }, new[] { 1, 0.3f, 0.3f, 0, 0.5f, 0.5f, 2, 5f, 6f });
        Assert.IsTrue(TestTensor.SameContent(df_id_sum.FloatCpuTensor(), expectedTensor, 1e-5));

        //var df_0_id_avg = df1.AverageBy("0", "id");
        //Assert.AreEqual(df_0_id_avg.ColumnNames, new[]{"0", "id", "1"});
        //expectedTensor = new CpuTensor<float>(new[] { 4, 3 }, new [] { 0.1f, 1, 0.1f, 0.2f, 0, 0.2f, 0.3f, 0, 0.2f, 5f, 2, 6f });
        //Assert.IsTrue(TestTensor.SameContent(df_0_id_avg.FloatCpuTensor(), expectedTensor, 1e-5));
    }

    [Test]
    public void TestSumOrAvgForMixedColumns()
    {
        var floatTensor = CpuTensor<float>.New(new[] { 0.1f, 0.2f, 0.2f, 0.3f, 0.3f, 0.2f, 5.0f, 6.0f, 0.1f, 0.1f, 0.1f, 0.0f, }, 2);
        var stringTensor = CpuTensor<string>.New(new []{"1", "0", "0", "2", "1", "1"}, 1);
        var df1 = DataFrame.New(floatTensor, stringTensor, null, new[]{"c0", "c1", "id"});

        var df_id_avg = df1.AverageBy("id");
        Assert.AreEqual(df_id_avg.Columns, new[] { "id", "c0", "c1"});
        var expectedFloatTensor = new CpuTensor<float>(new[] { 3, 2 }, new[] { 0.1f, 0.1f, 0.25f, 0.25f, 5f, 6f });
        Assert.IsTrue(TestTensor.SameContent(df_id_avg.FloatTensor, expectedFloatTensor, 1e-5));
        Assert.AreEqual(new[]{"1", "0", "2"}, df_id_avg.StringColumnContent("id"));

        var df_id_sum = df1.SumBy("id");
        Assert.AreEqual(df_id_sum.Columns, new[] { "id", "c0", "c1" });
        expectedFloatTensor = new CpuTensor<float>(new[] { 3, 2 }, new[] { 0.3f, 0.3f, 0.5f, 0.5f, 5f, 6f });
        Assert.IsTrue(TestTensor.SameContent(df_id_sum.FloatTensor, expectedFloatTensor, 1e-5));
        Assert.AreEqual(new[] { "1", "0", "2" }, df_id_sum.StringColumnContent("id"));
    }

    [Test]
    public void Test_sort_values()
    {
        var floatTensor = CpuTensor<float>.New(new[] { 0.1f, 0.4f, 0.2f, 0.5f, 0.3f, 0.6f}, 2);
        var stringTensor = CpuTensor<string>.New(new[] { "C", "A", "B" }, 1);
        var df1 = DataFrame.New(floatTensor, stringTensor, null, new[] { "c0", "c1", "id" });

        var df_id_ascending = df1.sort_values("id", ascending:true);
        Assert.AreEqual(df_id_ascending.Columns, df1.Columns);
        var expectedFloatTensor = new CpuTensor<float>(new[] { 3, 2 }, new[] { 0.2f, 0.5f, 0.3f, 0.6f, 0.1f, 0.4f });
        Assert.IsTrue(TestTensor.SameContent(df_id_ascending.FloatTensor, expectedFloatTensor, 1e-5));
        Assert.AreEqual(new[] { "A", "B", "C" }, df_id_ascending.StringColumnContent("id"));

        var df_id_descending = df1.sort_values("id", ascending: false);
        Assert.AreEqual(df_id_descending.Columns, df1.Columns);
        expectedFloatTensor = new CpuTensor<float>(new[] { 3, 2 }, new[] { 0.1f, 0.4f, 0.3f, 0.6f, 0.2f, 0.5f });
        Assert.IsTrue(TestTensor.SameContent(df_id_descending.FloatTensor, expectedFloatTensor, 1e-5));
        Assert.AreEqual(new[] { "C", "B", "A" }, df_id_descending.StringColumnContent("id"));

        var df_c1_ascending = df1.sort_values("c1", ascending:true);
        Assert.AreEqual(df_c1_ascending.Columns, df1.Columns);
        expectedFloatTensor = new CpuTensor<float>(new[] { 3, 2 }, new[] { 0.1f, 0.4f, 0.2f, 0.5f, 0.3f, 0.6f });
        Assert.IsTrue(TestTensor.SameContent(df_c1_ascending.FloatTensor, expectedFloatTensor, 1e-5));
        Assert.AreEqual(new[] { "C", "A", "B" }, df_c1_ascending.StringColumnContent("id"));

        var df_c2_ascending = df1.sort_values("c1", ascending: false);
        Assert.AreEqual(df_c2_ascending.Columns, df1.Columns);
        expectedFloatTensor = new CpuTensor<float>(new[] { 3, 2 }, new[] { 0.3f, 0.6f, 0.2f, 0.5f, 0.1f, 0.4f});
        Assert.IsTrue(TestTensor.SameContent(df_c2_ascending.FloatTensor, expectedFloatTensor, 1e-5));
        Assert.AreEqual(new[] { "B", "A", "C" }, df_c2_ascending.StringColumnContent("id"));
    }

    [Test]
    public void TestLeftJoinWithoutDuplicatesStringType()
    {
        /*
        left
        2       id      0       1
        3f 		"id2"	5.0f	6.0f 
        0.4f 	"id0"	0.2f	0.3f 
        7f 	    "id1"	0.1f	0.2f

        right:
        e1      id      letter  e2
        1f 		"id2"	"a"		2f	 
        3f 		"id2"	"b" 	4f 
        5f 		"id12"	"c" 	6f 
        0.4f 	"id0"	"d" 	8f
        */

        var leftFloatTensor = CpuTensor<float>.New(new[] { 3f, 5.0f, 6.0f, 0.4f, 0.2f, 0.3f, 7f, 0.1f, 0.2f, }, 3);
        var leftStringTensor = CpuTensor<string>.New(new[] { "id2", "id0", "id1" }, 1);
        var left = new DataFrame(
            new List<Tuple<string, int, int>>
            {
                Tuple.Create("2", DataFrame.FLOAT_TYPE_IDX, 0),
                Tuple.Create("id", DataFrame.STRING_TYPE_IDX, 0),
                Tuple.Create("0", DataFrame.FLOAT_TYPE_IDX, 1),
                Tuple.Create("1", DataFrame.FLOAT_TYPE_IDX, 2)
            },
            leftFloatTensor, leftStringTensor, null);

        var rightFloatTensor = CpuTensor<float>.New(new float[] { 1, 2, 3, 4, 5, 6, 0.4f, 8, }, 2);
        var rightStringTensor = CpuTensor<string>.New(new[] { "id2", "a", "id2", "b", "id12", "c", "id0", "d" }, 2);
        var right = new DataFrame(
            new List<Tuple<string, int, int>>
            {
                Tuple.Create("e1", DataFrame.FLOAT_TYPE_IDX, 0),
                Tuple.Create("e2", DataFrame.FLOAT_TYPE_IDX, 1),
                Tuple.Create("id", DataFrame.STRING_TYPE_IDX, 0),
                Tuple.Create("letter", DataFrame.STRING_TYPE_IDX, 1)
            },
            rightFloatTensor, rightStringTensor, null);

        /*
        join("id")
        3f 		"id2"	5.0f 	6.0f 	1 	"a"	 	2
        0.4f 	"id0"	0.2f 	0.3f 	0.4	"d"		8
        7f 	    "id1"	0.1f 	0.2f    0	null 	0
        */
        var observed = left.LeftJoinWithoutDuplicates(right, new[] { "id" });
        Assert.AreEqual(observed.Columns, new[] { "2", "id", "0", "1", "e1", "e2", "letter" });
        var expectedFloatTensor = new CpuTensor<float>(new[] { 3, 5 }, new[] { 3f, 5.0f, 6.0f, 1, 2, 0.4f, 0.2f, 0.3f, 0.4f, 8, 7f, 0.1f, 0.2f, 0, 0 });
        var expectedStringTensor = new CpuTensor<string>(new[] { 3, 2 }, new[] { "id2", "a", "id0", "d", "id1", null });
        Assert.IsTrue(TestTensor.SameContent(expectedFloatTensor, observed.FloatTensor, 1e-5));
        Assert.IsTrue(TestTensor.SameStringContent(expectedStringTensor, observed.StringTensor));

        /*
       join("2", "id")
        3f      "id2"   5.0f    6.0f    "b"     4f
        0.4f    "id0"   0.2f    0.3f    "d"     8f
        7f      "id1"   0.1f    0.2f    null,   0
       */
        observed = left.LeftJoinWithoutDuplicates(right, new[] { "2", "id" }, new[] { "e1", "id" });
        Assert.AreEqual(observed.Columns, new[] { "2", "id", "0", "1", "e2", "letter" });
        expectedFloatTensor = new CpuTensor<float>(new[] { 3, 4 }, new[]
        {
            3f, 5.0f, 6.0f, 4,
            0.4f, 0.2f, 0.3f, 8, 
            7f, 0.1f, 0.2f, 0
        });
        expectedStringTensor = new CpuTensor<string>(new[] { 3, 2 }, new[] { "id2", "b", "id0", "d", "id1", null });
        Assert.IsTrue(TestTensor.SameContent(expectedFloatTensor, observed.FloatTensor, 1e-5));
        Assert.IsTrue(TestTensor.SameStringContent(expectedStringTensor, observed.StringTensor));
    }

    [Test]
    public void TestLeftJoinWithoutDuplicatesFloat()
    {
        /*
        left:
        col0	col1	digit	year
        7f 		5.0f  	"1"		2005
        0.4f 	0.2f 	"9"		2018
        0.3f 	0.1f	"5"		2020

        right:
        month	letter	year
        1 		"a" 	2020 
        3 		"b" 	2020 
        5 		"c" 	2001 
        7 		"d" 	2005

        join("year"):
        col0	col1	digit	year	month letter
        7f 		5.0f  	"1"		2005	7		"d"
        0.4f 	0.2f 	"9"		2018	na		na
        0.3f 	0.1f	"5"		2020	1		"a"
        */

        var leftFloatTensor = CpuTensor<float>.New(new[]
        {
            7f, 5.0f, 2005, 
            0.4f, 0.2f, 2018, 
            0.3f, 0.1f, 2020,
        }, 3);
        var leftStringTensor = CpuTensor<string>.New(new[]
        {
            "1", 
            "9", 
            "5"
        }, 1);
        var left = new DataFrame(
            new List<Tuple<string, int, int>>
            {
                Tuple.Create("col0", DataFrame.FLOAT_TYPE_IDX, 0),
                Tuple.Create("col1", DataFrame.FLOAT_TYPE_IDX, 1),
                Tuple.Create("digit", DataFrame.STRING_TYPE_IDX, 0),
                Tuple.Create("year", DataFrame.FLOAT_TYPE_IDX, 2)
            },
            leftFloatTensor, leftStringTensor, null);

        var rightFloatTensor = CpuTensor<float>.New(new float[]
        {
            1, 2020, 
            3, 2020, 
            5, 2001, 
            7, 2005,
        }, 2);
        var rightStringTensor = CpuTensor<string>.New(new[]
        {
            "a", 
            "b", 
            "c", 
            "d",
        }, 1);
        var right = new DataFrame(
            new List<Tuple<string, int, int>>
            {
                Tuple.Create("month", DataFrame.FLOAT_TYPE_IDX, 0),
                Tuple.Create("letter", DataFrame.STRING_TYPE_IDX, 0),
                Tuple.Create("year", DataFrame.FLOAT_TYPE_IDX, 1),
            },
            rightFloatTensor, rightStringTensor, null);

        var observed = left.LeftJoinWithoutDuplicates(right, new[] { "year" });

        Assert.AreEqual(observed.Columns, new[] { "col0", "col1", "digit", "year", "month", "letter" });
        var expectedFloatTensor = new CpuTensor<float>(new[] { 3, 4 }, new[] { 7f, 5.0f, 2005, 7, 0.4f, 0.2f, 2018, 0, 0.3f, 0.1f, 2020, 1 });
        var expectedStringTensor = new CpuTensor<string>(new[] { 3, 2 }, new[] { "1", "d", "9", null, "5", "a" });
        Assert.IsTrue(TestTensor.SameContent(expectedFloatTensor, observed.FloatTensor, 1e-5));
        Assert.IsTrue(TestTensor.SameStringContent(expectedStringTensor, observed.StringTensor));
    }

}

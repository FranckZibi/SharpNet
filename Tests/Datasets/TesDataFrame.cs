using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Datasets;

namespace SharpNetTests.Datasets;

[TestFixture]
public class TesDataFrame
{
    [Test]
    public void TestSumOrAvgForColumns()
    {
        var df1 = NewDataFrame(
            new List<Tuple<string, int, int>> { FloatDesc("id", 0), FloatDesc("0", 1), FloatDesc("1", 2) },
            "1,0.1,0.2,0,0.2,0.3,0,0.3,0.2,2,5,6,1,0.1,0.1,1,0.1,0", null, null);

        var df_id_avg = df1.AverageBy("id");
        var expected = NewDataFrame(df1.ColumnsDesc.ToList(), "1,0.1,0.1,0,0.25,0.25,2,5,6");
        AssertSameContent(expected, df_id_avg);    

        var df_id_sum = df1.SumBy("id");
        expected = NewDataFrame(df1.ColumnsDesc.ToList(), "1,0.3,0.3,0,0.5,0.5,2,5,6");
        AssertSameContent(expected, df_id_sum);

        //var df_0_id_avg = df1.AverageBy("0", "id");
        //expected = NewDataFrame(df1.ColumnsDesc.ToList(), "0,0.1,0.1,0,0.2,0.2,0,0.3,0.3,1,0.1,0.1,1,0.1,0.1");
        //AssertSameContent(expected, df_0_id_avg);
    }

    [Test]
    public void TestSumOrAvgForMixedColumns()
    {
        var df1 = NewDataFrame(
            new List<Tuple<string, int, int>> { FloatDesc("c0", 0), FloatDesc("c1", 1), StringDesc("id", 0) },
            "0.1,0.2,0.2,0.3,0.3,0.2,5,6,0.1,0.1,0.1,0", "1,0,0,2,1,1", null);
        var df_id_avg = df1.AverageBy("id");
        var expected = NewDataFrame(
            new List<Tuple<string, int, int>> { StringDesc("id", 0), FloatDesc("c0", 0), FloatDesc("c1", 1) },
            "0.1,0.1,0.25,0.25,5,6", "1,0,2", null);
        AssertSameContent(expected, df_id_avg);

        var df_id_sum = df1.SumBy("id");
        expected = NewDataFrame(
            new List<Tuple<string, int, int>> { StringDesc("id", 0), FloatDesc("c0", 0), FloatDesc("c1", 1) },
            "0.3,0.3,0.5,0.5,5,6", "1,0,2", null);
        AssertSameContent(expected, df_id_sum);
    }

    [Test]
    public void Test_sort_values()
    {
        var df1 = NewDataFrame(
            new List<Tuple<string, int, int>> {FloatDesc("c0", 0), FloatDesc("c1", 1), StringDesc("id", 0) },
            "0.1,0.4,0.2,0.5,0.3,0.6", "C,A,B");

        var df_id_ascending = df1.sort_values("id", ascending:true);
        var expectedDf = NewDataFrame(df1.ColumnsDesc.ToList(), "0.2,0.5,0.3,0.6,0.1,0.4", "A,B,C");
        AssertSameContent(expectedDf, df_id_ascending);

        var df_id_descending = df1.sort_values("id", ascending: false);
        expectedDf = NewDataFrame(df1.ColumnsDesc.ToList(), "0.1,0.4,0.3,0.6,0.2,0.5", "C,B,A");
        AssertSameContent(expectedDf, df_id_descending);

        var df_c1_ascending = df1.sort_values("c1", ascending:true);
        expectedDf = NewDataFrame(df1.ColumnsDesc.ToList(), "0.1,0.4,0.2,0.5,0.3,0.6", "C,A,B");
        AssertSameContent(expectedDf, df_c1_ascending);

        var df_c2_ascending = df1.sort_values("c1", ascending: false);
        expectedDf = NewDataFrame(df1.ColumnsDesc.ToList(), "0.3,0.6,0.2,0.5,0.1,0.4", "B,A,C");
        AssertSameContent(expectedDf, df_c2_ascending);
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

        var left = NewDataFrame(
            new List<Tuple<string, int, int>> { FloatDesc("2", 0), StringDesc("id", 0), FloatDesc("0", 1), FloatDesc("1", 2) },
            "3,5,6,0.4,0.2,0.3,7,0.1,0.2", "id2,id0,id1", null);

        var right = NewDataFrame(
            new List<Tuple<string, int, int>>{ FloatDesc("e1", 0), FloatDesc("e2", 1), StringDesc("id", 0), StringDesc("letter", 1) },
            "1,2,3,4,5,6,0.4,8", "id2,a,id2,b,id12,c,id0,d", null);

        /*
        join("id")
        3f 		"id2"	5.0f 	6.0f 	1 	"a"	 	2
        0.4f 	"id0"	0.2f 	0.3f 	0.4	"d"		8
        7f 	    "id1"	0.1f 	0.2f    0	null 	0
        */
        var observed = left.LeftJoinWithoutDuplicates(right, new[] { "id" });
        var expectedDf = NewDataFrame(
            new List<Tuple<string, int, int>>{FloatDesc("2", 0), StringDesc("id", 0), FloatDesc("0", 1), FloatDesc("1", 2), FloatDesc("e1", 3), FloatDesc("e2", 4), StringDesc("letter", 1) }, 
            "3,5,6,1,2,0.4,0.2,0.3,0.4,8,7,0.1,0.2,0,0", "id2,a,id0,d,id1,"+NULL_STRING_TOKEN, null);
        AssertSameContent(expectedDf, observed);

        /*
       join("2", "id")
        3f      "id2"   5.0f    6.0f    "b"     4f
        0.4f    "id0"   0.2f    0.3f    "d"     8f
        7f      "id1"   0.1f    0.2f    null,   0
       */
        observed = left.LeftJoinWithoutDuplicates(right, new[] { "2", "id" }, new[] { "e1", "id" });
        expectedDf = NewDataFrame(
            new List<Tuple<string, int, int>>{FloatDesc("2", 0), StringDesc("id", 0), FloatDesc("0", 1), FloatDesc("1", 2), FloatDesc("e2", 3), },
            "3,5,6,4,0.4,0.2,0.3,8,7,0.1,0.2,0", "id2,b,id0,d,id1,"+NULL_STRING_TOKEN, null);
        AssertSameContent(expectedDf, observed);
    }

    [Test]
    public void TestHorizontalJoin()
    {
        //0     1       2       3       4       5
        //0f    1f      2f      "0_0"   "0_1"   100
        //3f    4f      5f      "1_0"   "1_3"   101
        var df1 = NewDataFrame(6, "0,1,2,3,4,5", "0_0,0_1,1_0,1_1", "100,101");

        //0     1       2       3       4  
        //-1f   -2f     a       b       -100
        //-4f   -6f     c       d       -101
        var df2 = NewDataFrame(5, "-1,-2,-4,-6", "a,b,c,d", "-100,-101");
        var df0 = DataFrame.MergeHorizontally(df1, df2);
        var df0_expected = NewDataFrame(11, "0,1,2,-1,-2,3,4,5,-4,-6", "0_0,0_1,a,b,1_0,1_1,c,d", "100,-100,101,-101");
        AssertSameContent(df0_expected, df0);


        //3     2
        //"0_0" 2f
        //"1_0" 5f
        var df1_view = df1["3", "2"];
        //4     0       2
        //-100  -1f     a    
        //-101  -4f     c
        var df2_view = df2["4", "0", "2"];
        //3     2       4     0       2
        //"0_0" 2f      -100  -1f     a    
        //"1_0" 5f      -101  -4f     c
        var df0_view = DataFrame.MergeHorizontally(df1_view, df2_view);
        var df0_view_expected = NewDataFrame(
            new[]{ StringDesc("3", 0), FloatDesc("2", 0), IntDesc("4", 0), FloatDesc("0", 1), StringDesc("2", 1)},
            "2,-1,5,-4", "0_0,a,1_0,c", "-100,-101");
        AssertSameContent(df0_view_expected, df0_view);
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
        var left = NewDataFrame(
            new List<Tuple<string, int, int>> { FloatDesc("col0", 0), FloatDesc("col1", 1), StringDesc("digit", 0), FloatDesc("year", 2) },
            "7,5.0,2005,0.4,0.2,2018,0.3,0.1,2020", "1,9,5", null);

        var right = NewDataFrame(
            new List<Tuple<string, int, int>> { FloatDesc("month", 0), StringDesc("letter", 0), FloatDesc("year", 1), },
            "1,2020,3,2020,5,2001,7,2005", "a,b,c,d", null);

        var observed = left.LeftJoinWithoutDuplicates(right, new[] { "year" });
        var expected = NewDataFrame(
            new List<Tuple<string, int, int>> { FloatDesc("col0", 0), FloatDesc("col1", 1), StringDesc("digit", 0), FloatDesc("year", 2),FloatDesc("month", 0), StringDesc("letter", 0), },
            "7,5.0,2005,7,0.4,0.2,2018,0,0.3,0.1,2020,1", "1,d,9,"+ NULL_STRING_TOKEN+",5,a", null);
        AssertSameContent(expected, observed);
    }


    
    private static DataFrame NewDataFrame(int cols, string floatValues, string stringValues = "", string intValues= "")
    {
        var floats = SplitForTests(floatValues, float.Parse);
        var strings = SplitForTests(stringValues, s => Equals(s, NULL_STRING_TOKEN) ? null : s);
        var ints = SplitForTests(intValues, int.Parse);
        int rows = (floats.Length+ strings.Length+ ints.Length)/cols;
        return DataFrame.New(
            CpuTensor<float>.New(floats, floats.Length / rows),
            CpuTensor<string>.New(strings, strings.Length / rows),
            CpuTensor<int>.New(ints, ints.Length / rows),
            Enumerable.Range(0, cols).Select(i => i.ToString()).ToArray()
        );
    }
    private static T[] SplitForTests<T>(string s, Func<string, T> parse)
    {
        if (string.IsNullOrEmpty(s))
        {
            return Array.Empty<T>();
        }
        return s.Split(',').Select(parse).ToArray();
    }

    private static DataFrame NewDataFrame(IList<Tuple<string, int, int>> colDesc, string floatValues, string stringValues = "", string intValues = "")
    {
        var floats = SplitForTests(floatValues, float.Parse);
        var strings = SplitForTests(stringValues, s => Equals(s, NULL_STRING_TOKEN) ? null : s);
        var ints = SplitForTests(intValues, int.Parse);
        int rows = (floats.Length + strings.Length + ints.Length) / colDesc.Count;
        return new DataFrame(
            colDesc.ToList(),
            CpuTensor<float>.New(floats, floats.Length / rows),
            CpuTensor<string>.New(strings, strings.Length / rows),
            CpuTensor<int>.New(ints, ints.Length / rows));
    }

    private static void AssertSameContent(DataFrame a, DataFrame b)
    {
        if (!a.SameContent(b, 1e-5, out var difference))
        {
            Assert.IsTrue(false, difference);
        }
    }
    private static Tuple<string, int, int> FloatDesc(string columnName, int indexInTensor) { return Tuple.Create(columnName, DataFrame.FLOAT_TYPE_IDX, indexInTensor); }
    private static Tuple<string, int, int> StringDesc(string columnName, int indexInTensor) { return Tuple.Create(columnName, DataFrame.STRING_TYPE_IDX, indexInTensor); }
    private static Tuple<string, int, int> IntDesc(string columnName, int indexInTensor) { return Tuple.Create(columnName, DataFrame.INT_TYPE_IDX, indexInTensor); }
    private const string NULL_STRING_TOKEN = "<null>";
}

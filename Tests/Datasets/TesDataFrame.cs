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
        var content = new[]
        {
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
        TestTensor.SameContent(df_id_avg.FloatCpuTensor(), expectedTensor, 1e-5);

        var df_id_sum = df1.SumBy("id");
        Assert.AreEqual(df_id_sum.Columns, columns);
        expectedTensor = new CpuTensor<float>(new[] { 3, 3 }, new[] { 1, 0.3f, 0.3f, 0, 0.5f, 0.5f, 2, 5f, 6f });
        TestTensor.SameContent(df_id_sum.FloatCpuTensor(), expectedTensor, 1e-5);

        //var df_0_id_avg = df1.AverageBy("0", "id");
        //Assert.AreEqual(df_0_id_avg.ColumnNames, new[]{"0", "id", "1"});
        //expectedTensor = new CpuTensor<float>(new[] { 4, 3 }, new [] { 0.1f, 1, 0.1f, 0.2f, 0, 0.2f, 0.3f, 0, 0.2f, 5f, 2, 6f });
        //TestTensor.SameContent(df_0_id_avg.FloatCpuTensor(), expectedTensor, 1e-5);
    }

    [Test]
    public void TestSumOrAvgForMixedColumns()
    {
        var content = new[]
        {
            0.1f, 0.2f,
            0.2f, 0.3f,
            0.3f, 0.2f,
            5.0f, 6.0f,
            0.1f, 0.1f,
            0.1f, 0.0f,
        };
        var floatTensor = CpuTensor<float>.New(content, 2);
        var stringTensor = CpuTensor<string>.New(new []{"1", "0", "0", "2", "1", "1"}, 1);
        var df1 = DataFrame.New(floatTensor, stringTensor, null, new[]{"c0", "c1", "id"});

        var df_id_avg = df1.AverageBy("id");
        Assert.AreEqual(df_id_avg.Columns, new[] { "id", "c0", "c1"});
        var expectedFloatTensor = new CpuTensor<float>(new[] { 3, 3 }, new[] { 0.1f, 0.1f, 0.25f, 0.25f, 5f, 6f });
        TestTensor.SameContent(df_id_avg.FloatTensor, expectedFloatTensor, 1e-5);
        Assert.AreEqual(new[]{"1", "0", "2"}, df_id_avg.StringColumnContent("id"));


        var df_id_sum = df1.SumBy("id");
        Assert.AreEqual(df_id_sum.Columns, new[] { "id", "c0", "c1" });
        expectedFloatTensor = new CpuTensor<float>(new[] { 3, 3 }, new[] { 0.3f, 0.3f, 0.5f, 0.5f, 5f, 6f });
        TestTensor.SameContent(df_id_sum.FloatTensor, expectedFloatTensor, 1e-5);
        Assert.AreEqual(new[] { "1", "0", "2" }, df_id_sum.StringColumnContent("id"));
    }

    [Test]
    public void TestLeftJoinWithoutDuplicates()
    {
        var leftColumns = new[] {"2", "id", "0", "1"};
        var leftContent = new[]
        {
            7f, 2f, 5.0f, 6.0f,
            0.4f, 0f, 0.2f, 0.3f,
            0.3f, 1f, 0.1f, 0.2f,
        };
        var left = DataFrame.New(leftContent, leftColumns);

        var rightColumns = new[] { "e1", "e2", "id" };
        var rightContent = new[]
        {
            1, 2, 2f,
            3, 4, 2f,
            5, 6, 12f,
            7, 8, 0f,
        };
        var right = DataFrame.New(rightContent, rightColumns);
        var observed = left.LeftJoinWithoutDuplicates(right, "id");

        Assert.AreEqual(observed.Columns, new[] { "2", "id", "0", "1", "e1", "e2" });
        var expected = new CpuTensor<float>(new[] { 3, 6 }, new[]
        {
            7f, 2f, 5.0f, 6.0f, 1, 2, 
            0.4f, 0f, 0.2f, 0.3f, 7, 8, 
            0.3f, 1f, 0.1f, 0.2f, 0, 0
        });
        TestTensor.SameContent(observed.FloatCpuTensor(), expected, 1e-5);
    }
}

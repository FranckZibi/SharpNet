// ReSharper disable UnusedMember.Global
// ReSharper disable InconsistentNaming
// ReSharper disable IdentifierTypo
// ReSharper disable CommentTypo

using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.Svm;

public class SVMSample : AbstractModelSample
{
    public SVMSample() : base(_categoricalHyperParameters)
    {
    }
    public override EvaluationMetricEnum GetLoss()
    {
        switch (svm_type)
        {
            case svm_type_enum.C_SVC:
            case svm_type_enum.nu_SVC:
                return EvaluationMetricEnum.CategoricalCrossentropy;
            case svm_type_enum.one_class_SVM:
                return EvaluationMetricEnum.BinaryCrossentropy;
            case svm_type_enum.epsilon_SVR:
            case svm_type_enum.nu_SVR:
                return EvaluationMetricEnum.Mae;
            default:
                throw new NotImplementedException($"can't manage svm_type {svm_type}");
        }
    }
    public override EvaluationMetricEnum GetRankingEvaluationMetric() => GetLoss();
    public override bool FixErrors()
    {
        return true;
    }

    public override void Use_All_Available_Cores()
    {
    }
    public override Model NewModel(AbstractDatasetSample datasetSample, string workingDirectory, string modelName)
    {
        return new SVMModel(this, workingDirectory, modelName);
    }

    private static readonly HashSet<string> _categoricalHyperParameters = new()
    {
        "svm_type",
        "kernel_type",
    };


    #region Hyper-Parameters

    public enum svm_type_enum
    {
        //C-Support Vector Classification. n-class classification (n >= 2),
        //allows imperfect separation of classes with penalty multiplier C for outliers. 
        C_SVC = 0,
        //nu-Support Vector Classification.
        //n-class classification with possible imperfect separation.
        //Parameter Nu (in the range 0..1, the larger the value, the smoother the decision boundary) is used instead of C. 
        nu_SVC = 1,
        //Distribution Estimation (One-class SVM).
        //All the training data are from the same class,
        //SVM builds a boundary that separates the class from the rest of the feature space. 
        one_class_SVM = 2,
        //epsilon-Support Vector Regression.
        //The distance between feature vectors from the training set and the fitting hyper-plane must be less than P.
        //For outliers the penalty multiplier C is used.
        epsilon_SVR = 3,
        //nu-Support Vector Regression.
        //Nu is used instead of p.
        nu_SVR = 4
    }
    //aliases: objective_type, app, application, loss
    // ReSharper disable once MemberCanBePrivate.Global
    public svm_type_enum svm_type = svm_type_enum.C_SVC;

    /// <summary>
    /// Type of a SVM kernel. Possible values are:
    /// </summary>
    public enum kernel_type_enum
    {
        // Linear kernel.
        // No mapping is done, linear discrimination (or regression) is done in the original feature space.
        // It is the fastest option.
        // Linear: u'*v
        linear = 0,

        // Polynomial kernel.
        // Polynomial: (gamma*u'*v + coef0)^degree
        polynomial = 1,

        // Radial basis function (RBF), a good choice in most cases.
        // Radial Basis Function: exp(-gamma*|u-v|^2)
        radial_basis_function = 2,

        // Sigmoid kernel.
        // Sigmoid: tanh(gamma*u'*v + coef0)
        sigmoid = 3,

        // (kernel values in training_set_file)
        precomputed_kernel = 4
    }

    public kernel_type_enum kernel_type = kernel_type_enum.radial_basis_function;


    /// <summary>
    /// degree in kernel function (POLY).
    /// </summary>
    public int degree = DEFAULT_VALUE;

    /// <summary>
    /// gamma of a kernel function (POLY / RBF / SIGMOID).
    /// </summary>
    public double gamma = DEFAULT_VALUE;

    /// <summary>
    /// Parameter coef0 of a kernel function (POLY / SIGMOID).
    /// </summary>
    public double coef0 = DEFAULT_VALUE;

    /// <summary>
    /// Parameter c of a SVM optimization problem (C_SVC / epsilon_SVR / nu_SVR).
    /// </summary>
    public double cost = DEFAULT_VALUE;

    /// <summary>
    /// Parameter nu of a SVM optimization problem (nu_SVC / ONE_CLASS_SVM / nu_SVR).
    /// </summary>
    public double nu = DEFAULT_VALUE;


    /// <summary>
    /// epsilon in loss function of epsilon-SVR (default 0.1)
    /// </summary>
    public double epsilon_SVR = DEFAULT_VALUE;


    /// <summary>
    ///  cache memory size in MB (default 100)
    /// </summary>
    public int cachesize = DEFAULT_VALUE;

    /// <summary>
    /// set tolerance of termination criterion (default 0.001)
    /// </summary>
    public double epsilon = DEFAULT_VALUE;

    /// <summary>
    /// whether to use the shrinking heuristics, 0 or 1 (default 1)
    /// </summary>
    public int shrinking = DEFAULT_VALUE;

    /// <summary>
    /// n: n-fold cross validation mode
    /// </summary>
    public int n_fold_svm = DEFAULT_VALUE;

    #endregion

    public static (DataFrame x, DataFrame y) LoadLibsvm(string path)
    {
        var allLines = System.IO.File.ReadAllLines(path);
        int maxIndex = -1;
        var yArray = new float[allLines.Length];
        foreach (var l in allLines)
        {
            int maxIndexForLine = l.Split().Skip(1).Where(x => x.Length >= 3).Select(c => int.Parse(c.Split(':')[0])).Max();
            maxIndex = Math.Max(maxIndex, maxIndexForLine);
        }
        var xArray = new float[allLines.Length * maxIndex];
        for (int i = 0; i < xArray.Length; ++i)
        {
            xArray[i] = float.NaN;
        }
        for (int i = 0; i < allLines.Length; ++i)
        {
            var splitted = allLines[i].Split(' ');
            yArray[i] = float.Parse(splitted[0]);
            foreach (var e in splitted.Skip(1).Where(x => x.Length >= 3).Select(c => c.Split(':')))
            {
                int index = int.Parse(e[0]);
                var value = float.Parse(e[1]);
                xArray[i * maxIndex + index - 1] = value;
            }
        }
        var x = DataFrame.New(xArray, Enumerable.Range(1, maxIndex).Select(i => i.ToString()).ToArray());
        var y = DataFrame.New(yArray, new[] { "y" });
        return (x, y);
    }


    /// <summary>
    /// The default Search Space for SVM Model
    /// </summary>
    /// <returns></returns>
    // ReSharper disable once UnusedMember.Global
    public static Dictionary<string, object> DefaultSearchSpace()
    {
        var searchSpace = new Dictionary<string, object>
        {
            //uncomment appropriate one
            //{"svm_type", "epsilon_SVR"},      //for Regression Tasks
            //{"svm_type", "nu_SVR"},       //for Regression Tasks

            //{"svm_type", "one_class_SVM"},    //for binary classification

            //{"svm_type", "C_SVC"},        //for multi class classification
            //{"svm_type", "nu_SVC },       //for multi class classification

            { "kernel_type", new[]{ "linear", "polynomial", "radial_basis_function", "sigmoid" } },

            { "cost", new[]{ 0.1, 1, 10, 100 } },
            { "gamma", new[]{ 1, 0.1, 0.01, 0.001 } },


        };

        return searchSpace;
    }

}
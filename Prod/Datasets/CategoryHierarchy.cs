using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

// ReSharper disable EnforceIfStatementBraces

namespace SharpNet.Datasets
{
    [DebuggerDisplay("{" + nameof(ToString) + "()}")]
    public class CategoryHierarchy
    {
        #region private fields
        private readonly string _name;
        private readonly string _displayName;
        private readonly CategoryHierarchy _parentIfAny;
        private readonly bool _onlyOneChildrenValidAtATime;
        private readonly List<CategoryHierarchy> _children = new ();
        private readonly int _startIndex;
        private float[] _lazyRootTemplate = null;
        #endregion

        private int IndexWithProba => HasAssociatedProba ? _startIndex : -1;
        private int IndexWithElementCount
        {
            get
            {
                if (_children.Count == 0)
                {
                    return -1;
                }
                return _startIndex+(HasAssociatedProba?1:0);
            }
        } 

        private int EndIndexExcluded
        {
            get
            {
                if (_children.Count >= 1)
                {
                    return _children.Last().EndIndexExcluded;
                }
                return _startIndex + ((IsRoot||HasAssociatedProba) ?1:0);
            }
        }


        public static CategoryHierarchy NewRoot(string rootName, string displayName)
        {
            return new CategoryHierarchy(rootName, displayName, null);
        }

        #region constructor
        private CategoryHierarchy(string name, string displayName, CategoryHierarchy parentIfAny, bool onlyOneChildrenValidAtATime = true)
        {
            _name = name;
            _displayName = displayName;
            _parentIfAny = parentIfAny;
            _onlyOneChildrenValidAtATime = onlyOneChildrenValidAtATime;

            if (_parentIfAny == null)
            {
                //root node
                _startIndex = 0;
            }
            else
            {
                if (_parentIfAny._children.Count == 0)
                {
                    //first child
                    _startIndex = _parentIfAny.EndIndexExcluded+(_parentIfAny.IsRoot?0:1); // sub categories count
                }
                else
                {
                    _startIndex = _parentIfAny._children.Last().EndIndexExcluded;
                }
                _parentIfAny._children.Add(this);
            }
        }
        #endregion

        public CategoryHierarchy Add(string name, string displayName = null, bool onlyOneChildrenValidAtATime = true)
        {
            return new CategoryHierarchy(name, displayName??name, this, onlyOneChildrenValidAtATime);
        }

        public void AddAllNumbersWithSameNumberOfDigits(string name, string displayName, int maxNumber)
        {
            Debug.Assert(maxNumber < 10000);
            if (maxNumber <= 9)
            {
                var subCategorySingleDigit = Add(name, displayName);
                subCategorySingleDigit.AddRange(1, maxNumber);
                return;
            }
            var subCategory = Add(name, displayName, false);
            var names = new[] { "units", "tens", "hundreds", "thousands" };
            int[] digits = maxNumber.ToString().Select(c => c - '0').ToArray();
            for (int i=0;i<digits.Length;++i)
            {
                int startNumber = i==0? 1 : 0;
                int count = i==0 ? digits[i] : 10;
                var c = subCategory.Add(names[digits.Length-i-1], "");
                c.AddRange(startNumber, count);
            }
        }
        public float[] RootPrediction()
        {
            if (!IsRoot)
            {
                return _parentIfAny.RootPrediction();
            }
            if (_lazyRootTemplate == null)
            {
                var tmp = new List<float>();
                FillTemplate(tmp);
                _lazyRootTemplate = tmp.ToArray();
            }
            return _lazyRootTemplate;
        }

        public Tuple<string,double> ExtractPredictionWithProba(ReadOnlySpan<float> prediction, double parentProba = 1.0)
        {
            string result = (HasAssociatedProba||IsRoot)?_displayName:"";
            if (_children.Count == 0)
            {
                return Tuple.Create(result, parentProba);
            }

            if (IndexWithElementCount != -1 && prediction[IndexWithElementCount] < -0.1)
            {
                if (!string.IsNullOrEmpty(result))
                {
                    return Tuple.Create(result, parentProba);
                }
                if (_onlyOneChildrenValidAtATime)
                {
                    result += "*";
                }
                else
                {
                    result += new string(Enumerable.Repeat('*', _children.Count).ToArray());
                }
                return Tuple.Create(result, parentProba);
            }

            if (_onlyOneChildrenValidAtATime)
            {
                var childrenOrderByProba = new List<Tuple<CategoryHierarchy, float>>();
                foreach(var c in _children)
                {
                    childrenOrderByProba.Add(Tuple.Create(c, prediction[c.IndexWithProba]));
                }
                childrenOrderByProba = childrenOrderByProba.OrderByDescending(t => t.Item2).ToList();
                parentProba *= childrenOrderByProba[0].Item2;
                var mostProbableChildren = childrenOrderByProba[0].Item1.ExtractPredictionWithProba(prediction, parentProba);
                return Tuple.Create(result + mostProbableChildren.Item1, mostProbableChildren.Item2);
            }
            foreach (var c in _children)
            {
                var childrenProba = c.ExtractPredictionWithProba(prediction);
                result += childrenProba.Item1;
                parentProba *= childrenProba.Item2;
            }
            return Tuple.Create(result, parentProba);
        }

        public static string CategoryPathToCategoryName(string[] path)
        {
            if (path == null || path.Length == 0)
            {
                return "";
            }
            return string.Join("/", path.Take(3));
        }

        public float[] ExpectedPrediction(string[] pathExpected)
        {
            var expected = (float[])RootPrediction().Clone();
            if (pathExpected == null)
            {
                return expected;
            }

            GetExpected(pathExpected, 0, expected);
            return expected;
        }
        public override string ToString()
        {
            return _name;
        }


       

        private void GetExpected(string[] pathExpected, int childrenIndexInPathExpected, float[] expected)
        {
            if (HasAssociatedProba)
            {
                expected[IndexWithProba] = 1f;
            }
            if (_children.Count == 0)
            {
                return;
            }

            if (_onlyOneChildrenValidAtATime)
            {
                var childrenDesc = ( childrenIndexInPathExpected >= pathExpected.Length) ? "*" : pathExpected[childrenIndexInPathExpected];
                Debug.Assert(!string.IsNullOrEmpty(childrenDesc));

                if (childrenDesc == "*")
                {
                    //we ignore the sub categories
                    Debug.Assert(_children.Count != 0);
                    expected[IndexWithElementCount] = -10* FloatLengthOfSubCategoriesDescription;
                    return;
                }

                foreach (var c in _children)
                {
                    if (c._name == childrenDesc)
                    {
                        c.GetExpected(pathExpected, childrenIndexInPathExpected + ((IsRoot || _parentIfAny._onlyOneChildrenValidAtATime) ? 1 : 0), expected);
                    }
                    else
                    {
                        Debug.Assert(c.HasAssociatedProba);
                        expected[c.IndexWithProba] = 0f;
                        if (c._children.Count != 0)
                        {
                            expected[c.IndexWithElementCount] = -10 * c.FloatLengthOfSubCategoriesDescription;
                        }
                    }
                }
            }
            else
            {
                if (childrenIndexInPathExpected >= pathExpected.Length)
                {
                    //we ignore the sub categories
                    Debug.Assert(_children.Count != 0);
                    expected[IndexWithElementCount] = -10* FloatLengthOfSubCategoriesDescription;
                    return;
                }
                Debug.Assert(pathExpected.Length- childrenIndexInPathExpected == _children.Count);
                for (var index = 0; index < _children.Count; index++)
                {
                    _children[index].GetExpected(pathExpected, childrenIndexInPathExpected + index, expected);
                }
            }
        }
        private bool HasAssociatedProba => !IsRoot && _parentIfAny._onlyOneChildrenValidAtATime;
        private bool IsRoot => _parentIfAny == null;
        private void FillTemplate(List<float> template)
        {
            if (HasAssociatedProba)
            {
                template.Add(0);
            }
            if (_children.Count >= 1)
            {
                template.Add(10 * _children.Count+(HasAssociatedProba?1:0));
                _children.ForEach(c => c.FillTemplate(template));
            }
        }
        /// <summary>
        /// total number of floats needed to describe current node sub categories (without the associate proba element (if any)
        /// will be 0 if the node has no sub categories
        /// </summary>
        /// <returns></returns>
        private int FloatLengthOfSubCategoriesDescription => EndIndexExcluded - _startIndex-(HasAssociatedProba?1:0);

        private void AddRange(int start, int count)
        {
            Enumerable.Range(start, count).ToList().ForEach((i => Add(i.ToString())));
        }
    }
}

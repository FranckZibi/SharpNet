using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

// ReSharper disable EnforceIfStatementBraces

namespace SharpNet.Datasets
{
    public class CategoryHierarchy
    {
        #region private fields
        private readonly string _name;
        private readonly CategoryHierarchy _parentIfAny;
        private readonly bool _onlyOneChildrenValidAtATime;
        private readonly List<CategoryHierarchy> _children = new List<CategoryHierarchy>();
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


        public static CategoryHierarchy NewRoot(string rootName)
        {
            return new CategoryHierarchy(rootName, null);
        }

        #region constructor
        private CategoryHierarchy(string name, CategoryHierarchy parentIfAny, bool onlyOneChildrenValidAtATime = true)
        {
            _name = name;
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

        public CategoryHierarchy Add(string name, bool onlyOneChildrenValidAtATime = true)
        {
            return new CategoryHierarchy(name, this, onlyOneChildrenValidAtATime);
        }

        public void AddAllNumbersWithSameNumberOfDigits(string name, int maxNumber)
        {
            Debug.Assert(maxNumber < 10000);
            if (maxNumber <= 9)
            {
                var subCategorySingleDigit = Add(name);
                subCategorySingleDigit.AddRange(1, maxNumber);
                return;
            }
            var subCategory = Add(name, false);
            var names = new[] { "units", "tens", "hundreds", "thousands" };
            int[] digits = maxNumber.ToString().Select(c => c - '0').ToArray();
            for (int i=0;i<digits.Length;++i)
            {
                int startNumber = i==0? 1 : 0;
                int count = i==0 ? digits[i] : 10;
                var c = subCategory.Add(names[digits.Length-i-1]);
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

        public string ExtractPrediction(ReadOnlySpan<float> prediction)
        {
            string result = HasAssociatedProba?_name:"";
            if (_children.Count == 0)
            {
                return result;
            }
            if (_onlyOneChildrenValidAtATime)
            {
                var selectedChildren = _children[0];
                foreach(var c in _children.Skip(1))
                {
                    if (prediction[c.IndexWithProba] > prediction[selectedChildren.IndexWithProba])
                    {
                        selectedChildren = c;
                    }
                }
                result += selectedChildren.ExtractPrediction(prediction);
            }
            else
            {
                var childrenPrediction = new List<string>();
                foreach (var c in _children)
                {
                    childrenPrediction.Add(c.ExtractPrediction(prediction));
                }
                result += string.Join("", childrenPrediction);
            }
            return result;
        }

        public static CategoryHierarchy ComputeRootNode()
        {
            var root = NewRoot("");
            root.Add("mint");
            
            var used = root.Add("used");
            var used_star = used.Add("star");
            used_star.AddAllNumbersWithSameNumberOfDigits("2digits", 39);
            used_star.Add("full");
            used_star.Add("empty");
            used_star.AddAllNumbersWithSameNumberOfDigits("1digit", 9);
            var used_gc = used.Add("gc");
            used_gc.AddAllNumbersWithSameNumberOfDigits("4digits", 6999);
            used_gc.AddAllNumbersWithSameNumberOfDigits("3digits", 999);
            used_gc.AddAllNumbersWithSameNumberOfDigits("2digits", 99);
            used_gc.AddAllNumbersWithSameNumberOfDigits("1digit", 9);
            var used_pc = used.Add("pc");
            used_pc.AddAllNumbersWithSameNumberOfDigits("4digits", 4999);
            used_pc.AddAllNumbersWithSameNumberOfDigits("3digits", 999);
            used_pc.AddAllNumbersWithSameNumberOfDigits("2digits", 99);
            used_pc.AddAllNumbersWithSameNumberOfDigits("1digit", 9);
            used.Add("preo1893");
            used.Add("ir");
            used.Add("cad");
            used.Add("cad_perle");
            used.Add("cad_octo");
            used.Add("cad_ondule");
            used.Add("cad_imprime");
            used.Add("cad_passe");
            used.Add("cad_barcelona");
            used.Add("amb");
            used.Add("anchor");
            used.Add("typo");
            used.Add("grille");
            used.Add("grille_ss_fin");
            used.Add("gros_points");
            used.Add("asna");
            used.Add("plume");

            return root;
        }

        public string CategoryPathToCategoryName(string[] path)
        {
            return string.Join("/", path.Take(3));
        }

        public float[] ExpectedPrediction(string[] pathExpected)
        {
            var expected = (float[])RootPrediction().Clone();
            GetExpected(pathExpected, 0, expected);
            return expected;
        }
        public override string ToString()
        {
            return _name;
        }


        public string[] ToPath(string cancel)
        {
            cancel = cancel.Replace("_bleue", "").Replace("_bleu", "").Replace("_rouge", "");
            if (string.IsNullOrEmpty(cancel)) return new string[0];
            if (cancel == "mint") return new[] { "mint" };
            if (cancel.StartsWith("used")) return new[] { "used" };
            if (cancel.StartsWith("cad_perle")) return new[] { "used", "cad_perle" };
            if (cancel.StartsWith("cad_octo")) return new[] { "used", "cad_octo" };
            if (cancel.StartsWith("cad_ondule")) return new[] { "used", "cad_ondule" };
            if (cancel.StartsWith("passe")) return new[] { "used", "cad_passe" };
            if (cancel == "imprime") return new[] { "used", "cad_imprime" };
            if (cancel.StartsWith("barcelona")) return new[] { "used", "cad_barcelona" };
            if (cancel == "cad") return new[] { "used", "cad" };
            if (cancel == "preo1893") return new[] { "used", "preo1893" };
            if (cancel == "typo") return new[] { "used", "typo" };
            if (cancel == "grille_ss_fin") return new[] { "used", "grille_ss_fin" };
            if (cancel == "grille") return new[] { "used", "grille" };
            if (cancel == "gros_points") return new[] { "used", "gros_points" };
            if (cancel == "asna") return new[] { "used", "asna" };
            if (cancel.StartsWith("amb")) return new[] { "used", "amb" };
            if (cancel.StartsWith("ancre")) return new[] { "used", "anchor" };
            if (cancel.StartsWith("plume")) return new[] { "used", "plume" };

            if (cancel.StartsWith("etoile"))
            {
                if (cancel == "etoile") return new[] { "used", "star" };
                if (cancel.StartsWith("etoile_pleine")) return new[] { "used", "star", "full" };
                if (cancel.StartsWith("etoile_evidee")) return new[] { "used", "star", "empty" };
                if (cancel.Length == 7)
                {
                    string units = "*";
                    if (char.IsDigit(cancel[6])) units = cancel[6].ToString();
                    return new[] { "used", "star", "1digit", units };
                }
                if (cancel.Length == 8)
                {
                    string tens = "*";
                    string units = "*";
                    if (char.IsDigit(cancel[6])) tens = cancel[6].ToString();
                    if (char.IsDigit(cancel[7])) units = cancel[7].ToString();
                    return new[] { "used", "star", "2digits", tens, units };
                }
                return null;
            }

            if (cancel.StartsWith("gc") || cancel.StartsWith("pc"))
            {
                var result = new List<string> { "used", cancel.Substring(0, 2) };

                var subCategory = "1digit";
                if (cancel.Length >= 4)
                {
                    subCategory = (Math.Min(cancel.Length, 6) - 2) + "digits";
                }
                result.Add(subCategory);

                for (int i = 2; i < Math.Min(cancel.Length, 6); ++i)
                {
                    if (!char.IsDigit(cancel[i]) && cancel[i] != '*') return null;
                    result.Add(cancel[i].ToString());
                }
                return result.ToArray();
            }
            return null;
        }

        private void GetExpected(string[] pathExpected, int indexInPathExpected, float[] expected)
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
                var desc = (indexInPathExpected >= pathExpected.Length) ? "*" : pathExpected[indexInPathExpected];
                Debug.Assert(!string.IsNullOrEmpty(desc));

                if (desc == "*")
                {
                    //we ignore the sub categories
                    Debug.Assert(_children.Count != 0);
                    expected[IndexWithElementCount] = -10* FloatLengthOfSubCategoriesDescription;
                    return;
                }

                foreach (var c in _children)
                {
                    if (c._name == desc)
                    {
                        c.GetExpected(pathExpected, indexInPathExpected+ ((IsRoot || _parentIfAny._onlyOneChildrenValidAtATime) ? 1 : 0), expected);
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
                if (indexInPathExpected >= pathExpected.Length)
                {
                    //we ignore the sub categories
                    Debug.Assert(_children.Count != 0);
                    expected[IndexWithElementCount] = -10* FloatLengthOfSubCategoriesDescription;
                    return;
                }
                Debug.Assert(pathExpected.Length- indexInPathExpected == _children.Count);
                for (var index = 0; index < _children.Count; index++)
                {
                    _children[index].GetExpected(pathExpected, indexInPathExpected + index, expected);
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

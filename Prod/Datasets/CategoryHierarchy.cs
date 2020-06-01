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
        private readonly List<CategoryHierarchy> _children = new List<CategoryHierarchy>();
        private readonly string _name;
        private readonly bool _onlyOneChildrenValidAtATime;
        private int? _lazyCount;
        private float[] _lazyRootTemplate = null;
        private CategoryHierarchy _parentIfAny;
        #endregion

        #region constructor
        public CategoryHierarchy(string name, bool onlyOneChildrenValidAtATime = true)
        {
            _name = name;
            _onlyOneChildrenValidAtATime = onlyOneChildrenValidAtATime;
        }
        #endregion

        public void Add(CategoryHierarchy child)
        {
            Debug.Assert(child._parentIfAny == null);
            _children.Add(child);
            child._parentIfAny = this;
        }
        public void AddAllNumbersWithSameNumberOfDigits(string name, int maxNumber)
        {
            Debug.Assert(maxNumber < 10000);
            if (maxNumber <= 9)
            {
                var subCategorySingleDigit = new CategoryHierarchy(name);
                subCategorySingleDigit.AddRange(Range(1, maxNumber));
                Add(subCategorySingleDigit);
                return;
            }
            var subCategory = new CategoryHierarchy(name, false);
            var cat = new List<CategoryHierarchy>();
            var names = new[] { "units", "tens", "hundreds", "thousands" };
            int index = 0;
            while (maxNumber > 0)
            {
                int minNumber = maxNumber >= 10 ? 0 : 1;
                int count = maxNumber >= 10 ? 10 : maxNumber;
                var c = new CategoryHierarchy(names[index]);
                c.AddRange(Range(minNumber, count));
                cat.Insert(0, c);
                maxNumber /= 10;
                ++index;
            }
            subCategory.AddRange(cat);
            Add(subCategory);
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

        public static CategoryHierarchy ComputeRootNode()
        {
            var root = new CategoryHierarchy("");
            root.Add(new CategoryHierarchy("mint"));
            
            var used = new CategoryHierarchy("used");
            var used_star = new CategoryHierarchy("star");
            used_star.AddAllNumbersWithSameNumberOfDigits("2digits", 39);
            used_star.Add(new CategoryHierarchy("full"));
            used_star.Add(new CategoryHierarchy("empty"));
            used_star.AddAllNumbersWithSameNumberOfDigits("1digit", 9);
            used.Add(used_star);
            var used_gc = new CategoryHierarchy("gc");
            used_gc.AddAllNumbersWithSameNumberOfDigits("4digits", 6999);
            used_gc.AddAllNumbersWithSameNumberOfDigits("3digits", 999);
            used_gc.AddAllNumbersWithSameNumberOfDigits("2digits", 99);
            used_gc.AddAllNumbersWithSameNumberOfDigits("1digit", 9);
            used.Add(used_gc);
            var used_pc = new CategoryHierarchy("pc");
            used_pc.AddAllNumbersWithSameNumberOfDigits("4digits", 4999);
            used_pc.AddAllNumbersWithSameNumberOfDigits("3digits", 999);
            used_pc.AddAllNumbersWithSameNumberOfDigits("2digits", 99);
            used_pc.AddAllNumbersWithSameNumberOfDigits("1digit", 9);
            used.Add(used_pc);
            used.Add(new CategoryHierarchy("preo1893"));
            used.Add(new CategoryHierarchy("ir"));
            used.Add(new CategoryHierarchy("cad"));
            used.Add(new CategoryHierarchy("cad_perle"));
            used.Add(new CategoryHierarchy("cad_octo"));
            used.Add(new CategoryHierarchy("cad_ondule"));
            used.Add(new CategoryHierarchy("cad_imprime"));
            used.Add(new CategoryHierarchy("cad_passe"));
            used.Add(new CategoryHierarchy("cad_barcelona"));
            used.Add(new CategoryHierarchy("amb"));
            used.Add(new CategoryHierarchy("anchor"));
            used.Add(new CategoryHierarchy("typo"));
            used.Add(new CategoryHierarchy("grille"));
            used.Add(new CategoryHierarchy("grille_ss_fin"));
            used.Add(new CategoryHierarchy("gros_points"));
            used.Add(new CategoryHierarchy("asna"));
            used.Add(new CategoryHierarchy("plume"));
            root.Add(used);

            return root;
        }

        public string CategoryPathToCategoryName(string[] path)
        {
            return string.Join("/", path.Take(3));
        }

        public float[] ExpectedPrediction(string[] pathExpected)
        {
            var expected = (float[])RootPrediction().Clone();
            int indexInExpected = 0;
            GetExpected(pathExpected, 0, expected, ref indexInExpected);
            //System.IO.File.AppendAllText("c:/download/toto2.txt", "new float[]{"+string.Join(", ", expected  )+"}"+System.Environment.NewLine);
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

        private void SetWrongPath(float[] expected, ref int indexInExpected)
        {
            Debug.Assert(HasAssociatedProba);
            expected[indexInExpected++] = 0f;
            if (_children.Count != 0)
            {
                expected[indexInExpected] = -10 * GetLengthOfSubCategoriesDescription();
                indexInExpected += GetLengthOfSubCategoriesDescription();
            }
        }
        private void GetExpected(string[] pathExpected, int indexInPathExpected, float[] expected, ref int indexInExpected)
        {
            if (HasAssociatedProba)
            {
                expected[indexInExpected++] = 1f;
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
                    expected[indexInExpected] = -10*GetLengthOfSubCategoriesDescription();
                    indexInExpected += GetLengthOfSubCategoriesDescription();
                    return;
                }

                ++indexInExpected;
                foreach (var c in _children)
                {
                    if (c._name == desc)
                    {
                        c.GetExpected(pathExpected, indexInPathExpected+ ((IsRoot || _parentIfAny._onlyOneChildrenValidAtATime) ? 1 : 0), expected, ref indexInExpected);
                    }
                    else
                    {
                        c.SetWrongPath(expected, ref indexInExpected);
                    }
                }
            }
            else
            {
                if (indexInPathExpected >= pathExpected.Length)
                {
                    //we ignore the sub categories
                    Debug.Assert(_children.Count != 0);
                    expected[indexInExpected] = -10*GetLengthOfSubCategoriesDescription();
                    indexInExpected += GetLengthOfSubCategoriesDescription();
                    return;
                }
                Debug.Assert(pathExpected.Length- indexInPathExpected == _children.Count);
                ++indexInExpected;
                for (var index = 0; index < _children.Count; index++)
                {
                    _children[index].GetExpected(pathExpected, indexInPathExpected + index, expected, ref indexInExpected);
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
        private int GetLengthOfSubCategoriesDescription()
        {
            if (_lazyCount.HasValue)
            {
                return _lazyCount.Value;
            }
            //int res = HasAssociatedProba ? 1 : 0; //no proba associated with the root element
            if (_children.Count == 0)
            {
                _lazyCount = 0;
                return 0;
            }
            int res = 1; //count of sub categories  (if positive) or number of elements to skip
            res += _children.Select(c => c.GetLengthOfSubCategoriesDescription() + (c.HasAssociatedProba ? 1 : 0)).Sum();
            _lazyCount = res;
            return _lazyCount.Value;

        }
        private static List<CategoryHierarchy> Range(int start, int count)
        {
            return Enumerable.Range(start, count).Select(i => new CategoryHierarchy(i.ToString())).ToList();
        }
        private void AddRange(List<CategoryHierarchy> children)
        {
            children.ForEach(Add);
        }
    }
}

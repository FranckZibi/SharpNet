using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpNet.DataAugmentation.Operations
{
    public static class OperationHelper
    {
        public static void CheckIntegrity(List<Operation> operations)
        {
            if (operations.Count == 0)
            {
                return; //empty operation list is allowed
            }
            var dico = operations.GroupBy(x => x.GetType()).ToDictionary(x => x.Key, x => x.Count());

            ////the following types are allowed to be duplicates in a list
            //var allowedDuplicates = new HashSet<Type> {typeof(TranslateX), typeof(TranslateY), typeof(Equalize), typeof(Sharpness) };
            ////no duplicates operations
            //foreach (var e in dico)
            //{
            //    if (e.Value >= 2 && !allowedDuplicates.Contains(e.Key))
            //    {
            //        throw new ArgumentException(e.Value + " operations of type " + e.Key);
            //    }
            //}

            //the Cutout operation (whe it is used) must be the last one in the least
            if (dico.ContainsKey(typeof(Cutout)) && operations.Last().GetType() != typeof(Cutout))
            {
                throw new ArgumentException(typeof(Cutout) + " operation must be the last one in the list");
            }
            //CutMix and Mixup can't be used at the same time
            if (dico.ContainsKey(typeof(CutMix)) && dico.ContainsKey(typeof(Mixup)))
            {
                throw new ArgumentException(typeof(CutMix) + " and "+typeof(Mixup)+" can't appear at the same time in the list");
            }

            //CutMix and Mixup must be the last operations (only Cutout is allowed to be after them)
            foreach (var t in new[] {typeof(CutMix), typeof(Mixup)})
            {
                if (!dico.ContainsKey(t))
                {
                    continue;
                }
                if (dico.ContainsKey(typeof(Cutout)))
                {
                    if (operations[operations.Count - 2].GetType() != t)
                    {
                        throw new ArgumentException(t + " operation must be just before " + typeof(Cutout));
                    }
                }
                else
                {
                    if (operations.Last().GetType() != t)
                    {
                        throw new ArgumentException(t + " operation must be the last operation");
                    }
                }
            }
        }
    }
}
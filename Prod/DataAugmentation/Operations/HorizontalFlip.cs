namespace SharpNet.DataAugmentation.Operations
{
    public class HorizontalFlip : Operation
    {
        public HorizontalFlip(int[] miniBatchShape) : base(miniBatchShape)
        {
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row, NbCols - col - 1);
        }
    }
}

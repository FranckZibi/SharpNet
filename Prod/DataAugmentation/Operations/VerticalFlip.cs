namespace SharpNet.DataAugmentation.Operations
{
    public class VerticalFlip : Operation
    {
        public VerticalFlip(int[] miniBatchShape) : base(miniBatchShape)
        {
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (NbRows - row - 1, col);
        }
    }
}
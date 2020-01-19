namespace SharpNet.DataAugmentation.Operations
{
    public class HorizontalFlip : Operation
    {
        private readonly int _nbCols;

        public HorizontalFlip(int nbCols)
        {
            _nbCols = nbCols;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row, _nbCols - col - 1);
        }
        public override bool ChangeCoordinates()
        {
            return true;
        }

    }
}

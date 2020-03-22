extern "C" {

    __global__ void Sum(int N, const float* __restrict left, const float* __restrict right, float* __restrict output) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
			output[i] = left[i] + right[i];
	}

    __global__ void UpdateAdamOptimizer(int N, float beta1, float beta2, float epsilon, float multiplicative_factor,
				const float* __restrict dW, float* __restrict W,
				float* __restrict adam_vW, float* __restrict adam_sW) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
			float dw = dW[i];
			adam_vW[i] = beta1*adam_vW[i]+(1-beta1)*dw;
            adam_sW[i] = beta2*adam_sW[i]+(1-beta2)*dw*dw;
			W[i] -= (multiplicative_factor * adam_vW[i]) / (sqrtf(adam_sW[i]) + epsilon);
		}
	}

	__global__ void ComputeAccuracy(int N, int categoryCount, float *countOk, const float* __restrict yExpectedOneHot, const float* __restrict yPredicted) 
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			if (categoryCount == 1)
			{
				float error = fabsf(yExpectedOneHot[i] - yPredicted[i]);
				countOk[i] = (error < 0.5f) ? 1.0f : 0.0f;
				return;
			}

			int startIndex = i * categoryCount;
			int endIndexExcluded = startIndex + categoryCount;
			int maxIndexPredicted = startIndex;
			int maxIndexExpected = startIndex;
			for (int j = startIndex+1; j < endIndexExcluded; ++j)
			{
				if (yPredicted[j] > yPredicted[maxIndexPredicted])
					maxIndexPredicted = j;
				if (yExpectedOneHot[j] > yExpectedOneHot[maxIndexExpected])
					maxIndexExpected = j;
			}
			countOk[i] = (maxIndexPredicted == maxIndexExpected) ? 1.0f : 0.0f;
		}
	}

	__global__ void MultiplyEachRowIntoSingleValue(int nbRows, int nbCols, float *result, const float* __restrict a, const float* __restrict b) 
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < nbRows) {
			a += row*nbCols;
			b += row*nbCols;
			float sumInRow = 0;
			for(int i=0;i<nbCols;++i)
			{
				sumInRow += (*a)*(*b);
				++a;
				++b;
			}
			result[row] = sumInRow;
		}
	}

	__global__ void ComputeCategoricalCrossentropyLoss(int N, int categoryCount, float *losses, const float* __restrict yExpectedOneHot, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			float loss = 0.0f;
			int startIndex = i * categoryCount;
			int endIndexExcluded = startIndex + categoryCount;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float predicted = yPredicted[j];
				float expected = yExpectedOneHot[j];
				if (predicted > 0)
					loss -= expected * logf(predicted);
			}
			losses[i] = loss;
		}
	}

	__global__ void ComputeBinaryCrossentropyLoss(int N, int categoryCount, float *losses, const float* __restrict yExpectedOneHot, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			float loss = 0.0f;
			int startIndex = i * categoryCount;
			int endIndexExcluded = startIndex + categoryCount;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float predicted = yPredicted[j];
				float expected = yExpectedOneHot[j];
				//if ((predicted>0.01)&&(predicted<0.99f))
				if ((predicted>0.0f)&&(predicted<1.0f))
					loss -= (expected*logf(predicted) + (1.0f-expected)*logf(1.0f-predicted))/ categoryCount;
			}
			losses[i] = loss;
		}
	}




	__global__ void ComputeAccuracyFromCategoryIndexes(int N, int categoryCount, float *countOk, const int* __restrict categoryIndexes, const float* __restrict yPredicted) 
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			int categoryIndex = categoryIndexes[i]; /* the expected category index for element at index 'i' */
			int startIndex = i * categoryCount;
			int endIndexExcluded = startIndex + categoryCount;
			int maxIndexPredicted = startIndex;
			for (int j = startIndex+1; j < endIndexExcluded; ++j)
			{
				if (yPredicted[j] > yPredicted[maxIndexPredicted])
					maxIndexPredicted = j;
			}
			countOk[i] = ( (maxIndexPredicted-startIndex) == categoryIndex) ? 1.0f : 0.0f;
		}
	}

	__global__ void ComputeCategoricalCrossentropyLossFromCategoryIndexes(int N, int categoryCount, float *losses, const int* __restrict categoryIndexes, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			int categoryIndex = categoryIndexes[i]; /* the expected category index for element at index 'i' */
			int startIndex = i * categoryCount;
			float predictedForExpectedCategory = yPredicted[startIndex+categoryIndex];
			if (predictedForExpectedCategory > 0)
				losses[i] = -logf(predictedForExpectedCategory);
			else
				losses[i] = 0.0f;
		}
	}

	__global__ void ComputeBinaryCrossentropyLossFromCategoryIndexes(int N, int categoryCount, float *losses, const int* __restrict categoryIndexes, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			int categoryIndex = categoryIndexes[i]; /* the expected category index for element at index 'i' */
			int startIndex = i * categoryCount;
			float loss = 0.0f;
			for (int category = 0; category < categoryCount; ++category)
			{
				float predicted = yPredicted[startIndex+category];
				float error = (category == categoryIndex)? predicted : (1.0f-predicted);
				if (error > 0)
				{
					loss -= logf(error);
				}
			}
			losses[i] = loss/ categoryCount;
		}
	}

	__global__ void Concatenate(int N, int m, float* __restrict concat, int concatMultDim0, const float* __restrict a, int aMultDim0, const float* __restrict b, int bMultDim0)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N)  return;
		int row = i/concatMultDim0;
		int colInConcat = i%concatMultDim0;
		concat[i] = (colInConcat<aMultDim0)?a[row*aMultDim0+colInConcat]:b[row*bMultDim0+colInConcat-aMultDim0];
	}

	__global__ void Split(int N, int m, const float* __restrict concat, int concatMultDim0, float* __restrict a, int aMultDim0, float* __restrict b, int bMultDim0)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N)  return;
		int row = i/concatMultDim0;
		int colInConcat = i%concatMultDim0;
		if (colInConcat<aMultDim0)
			a[row*aMultDim0+colInConcat] = concat[i];
		else
			b[row*bMultDim0+colInConcat-aMultDim0] = concat[i];
	}
}

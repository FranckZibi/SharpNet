extern "C" {

	__device__ inline float sigmoidf(float x) {
		return 1.0f / (1 + expf(-x));
	}

    __global__ void Sum(int N, const float* __restrict left, const float* __restrict right, float* __restrict output) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
			output[i] = left[i] + right[i];
	}

    __global__ void UpdateAdamOptimizer(int N, float beta1, float beta2, float epsilon, float adamW_l2Regularization, float multiplicative_factor,
				const float* __restrict dW, float* __restrict W,
				float* __restrict adam_vW, float* __restrict adam_sW) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
			float dw = dW[i];
			adam_vW[i] = beta1*adam_vW[i]+(1-beta1)*dw;
            adam_sW[i] = beta2*adam_sW[i]+(1-beta2)*dw*dw;
			W[i] -= (multiplicative_factor * adam_vW[i]) / (sqrtf(adam_sW[i]) + epsilon) + adamW_l2Regularization * W[i];
		}
	}
	
	// for each row of tensor 'x' (of shape (N, cols), normalize the value
	//  y = (x-mean)/volatility
	__global__ void StandardizeInPlaceByRow(int N, int cols, int pointsByThread, int threadsByRow, float* __restrict x, float* __restrict row_mean, float* __restrict row_variance, float epsilon) 
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < N) {
			//N == cols * threadsByRow
			int row = idx / threadsByRow;
			int col = pointsByThread*(idx % threadsByRow);
            int startIndex = row*cols+col;
            int endIndex = startIndex + fminf(pointsByThread-1, cols-col-1);

			float slope = 1.0f/sqrtf(row_variance[row] + epsilon);
			float intercept = - row_mean[row]*slope;
            for (int i = startIndex; i <= endIndex; ++i)
            {
                x[i] = slope*x[i]+intercept;
            }
		}
	}

	// for each row of tensor 'x' (of shape (N, cols),
	//  1.normalize the value:
	//		y[row,col] = (x[row,col]-mean[row])/volatility[row]
	// 2. update with gammas and betas
	//      y[row,col] = gamma[col]*y[row,col]+beta[col]
	__global__ void StandardizeRowsInPlaceBroadcastGammasBetas(int N, int cols, int pointsByThread, int threadsByRow, float* __restrict x, float* __restrict row_mean, float* __restrict row_variance, float epsilon, float* col_gammas, float* col_betas) 
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < N) {
			//N == cols * threadsByRow
			int row = idx / threadsByRow;
			int col = pointsByThread*(idx % threadsByRow);
            int startIndex = row*cols+col;
            int endIndex = startIndex + fminf(pointsByThread-1, cols-col-1);

			float slope = 1.0f/sqrtf(row_variance[row] + epsilon);
			float intercept = - row_mean[row]*slope;
            for (int i = startIndex; i <= endIndex; ++i)
            {
                x[i] = col_gammas[col]*(slope*x[i]+intercept)+col_betas[col];
				++col;
            }
		}
	}
	__global__ void numpy_sum_ColByCol(int cols, int rows, const float* __restrict x, float* __restrict sum_buffer) 
	{
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		if (col < cols) {

			float col_sum = 0.0f;
            for (int row = 0; row< rows; ++row)
            {
				col_sum += x[row * cols + col];
            }
			sum_buffer[col] = col_sum;
		}
	}

	__global__ void numpy_sum_RowByRow(int rows, int cols, const float* __restrict x, float* __restrict sum_buffer) 
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < rows) {

			float row_sum = 0.0f;
			int index = row * cols;
            for (int col = 0; col< cols; ++col)
            {
				row_sum += x[index];
				++index;
            }
			sum_buffer[row] = row_sum;
		}
	}


	// compute dmean and  dvariance vectors
	__global__ void LayerNormalizationBackward_dmean_dvariance(int rows, int cols, const float* __restrict x,  const float* __restrict dy,  const float* __restrict gammas,  const float* __restrict mean,  const float* __restrict variance, float epsilon, float* dmean,  float* dvariance) 
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < rows) {
			float mean_row = mean[row];
			float variance_row = variance[row];
			float volatility_row = sqrtf(variance_row + epsilon);
			float dvariance_row = 0;
			float dmean_row = 0;
			float sum_x_row_minus_mean = 0.0f;
			int index =row*cols;
			for (int col = 0; col < cols; ++col)
			{
			    float tmp0 = (dy[index+col] * gammas[col]);
				float current_x_minus_mean = x[index+col]-mean_row;
			    dvariance_row += tmp0 * current_x_minus_mean;
				sum_x_row_minus_mean += current_x_minus_mean;
			    dmean_row -= tmp0;
			}
			dvariance_row *= (-0.5f * powf(variance_row + epsilon, -1.5f));
			dmean_row /= volatility_row;


			dmean_row += dvariance_row*(sum_x_row_minus_mean)*(-2.0f/cols);
			//for (int col = 0; col < cols; ++col)
			//    dmean_row += dvariance_row*(x[index+col] -mean_row) * (-2.0f/cols);

			dmean[row]=dmean_row;
			dvariance[row]=dvariance_row;
		}
	}

	// compute dx tensor
	__global__ void LayerNormalizationBackward_dx(int N, int cols, int pointsByThread, int threadsByRow, const float* __restrict x,  const float* __restrict dy,  float* __restrict dx,  const float* __restrict gammas,  const float* __restrict mean,  const float* __restrict variance, float epsilon, const float* dmean,  const float* dvariance) 
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < N) {
			//N == cols * threadsByRow
			int row = idx / threadsByRow;
			int col = pointsByThread*(idx % threadsByRow);
            int startIndex = row*cols+col;
            int endIndex = startIndex + fminf(pointsByThread-1, cols-col-1);

			float volatility_row = sqrtf(variance[row] + epsilon);
			float slope_x = dvariance[row] * (2.0f / cols);
			float intercept_dx = -dvariance[row] * (2.0f / cols)*mean[row]+ dmean[row] / cols;
			for (int i = startIndex; i <= endIndex; ++i)
			{
			    dx[i] = (dy[i] * gammas[col]) /volatility_row
			                + slope_x * x[i]
			                + intercept_dx;
			                //+ variance[row] * (2.0f / cols) * (x[i] - mean_row)
			                //+ dmean_row / cols;
				++col;
			}
		}
	}



	// for each row of tensor 'x' (of shape (rows, cols), compute the associate mean and variance
	// and stores it in tensor mean and variance (both of shape (rows,1))
	__global__ void Compute_Row_Mean_Variance(int N, int cols, const float* __restrict x, float* __restrict mean, float* __restrict variance, bool unbiasedVariance) 
	{
		int rowId = blockIdx.x * blockDim.x + threadIdx.x;
		if (rowId < N) {
				int startIndex = rowId * cols;
                int endIndex = startIndex + cols - 1;
                double sum = 0.0;
                double sumSquare = 0.0;
                for (int i = startIndex; i <= endIndex; ++i)
                {
					double x_i = x[i];
                    sum += x_i;
                    sumSquare += x_i * x_i;
                }
                float row_mean = (float)(sum / cols);
                mean[rowId] = row_mean;
                int divider = unbiasedVariance ? (cols - 1) : cols;
                float row_variance = (float) abs(sumSquare - cols * row_mean * row_mean) / divider;
                variance[rowId] = row_variance;
		}
	}


	__global__ void ComputeAccuracy(int N, int numClass, float *buffer, const float* __restrict yExpectedOneHot, const float* __restrict yPredicted) 
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < N) {
			if (numClass == 1)
			{
				float error = fabsf(yExpectedOneHot[row] - yPredicted[row]);
				buffer[row] = (error < 0.5f) ? 1.0f : 0.0f;
				return;
			}

			int startIndex = row * numClass;
			int endIndexExcluded = startIndex + numClass;
			int maxIndexPredicted = startIndex;
			int maxIndexExpected = startIndex;
			for (int j = startIndex+1; j < endIndexExcluded; ++j)
			{
				if (yPredicted[j] > yPredicted[maxIndexPredicted])
					maxIndexPredicted = j;
				if (yExpectedOneHot[j] > yExpectedOneHot[maxIndexExpected])
					maxIndexExpected = j;
			}
			buffer[row] = (maxIndexPredicted == maxIndexExpected) ? 1.0f : 0.0f;
		}
	}

	__global__ void ComputeSparseAccuracy(int N, int numClass, float *countOk, const float* __restrict yExpectedSparse, const float* __restrict yPredicted) 
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < N) {
            int expectedClassIndex = (int)(yExpectedSparse[row]+0.1f);
			int startIndex = row * numClass;
			int predictedClassIndex = 0;
			for (int classIndex = 1; classIndex < numClass; ++classIndex)
			{
				if (yPredicted[startIndex+classIndex] > yPredicted[startIndex+predictedClassIndex])
					predictedClassIndex = classIndex;
			}
			countOk[row] = (predictedClassIndex == expectedClassIndex) ? 1.0f : 0.0f;
		}
	}

	__global__ void ArgMax(int N, int cols, float* __restrict buffer, const float* __restrict yPredicted) 
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < N) {
			int startIndex = row * cols;
			int colArgMax = 0;
			for (int col = 1; col < cols; ++col)
			{
				if (yPredicted[startIndex+col] > yPredicted[startIndex+colArgMax])
					colArgMax = col;
			}
			buffer[row] = colArgMax;
		}
	}

	__device__  bool IsAccuratePredictionForCategoricalCrossentropyWithHierarchy(const float* __restrict expected, const float* __restrict predicted, int endIndexExcluded, int *pNexIndexToCheck, int subCategoriesCount)
	{
		int subCategoriesFound = 0;
		int predictedSubCategoryId = -1;
		float bestPredictedSubCategoryProba = -1.0f;
		int expectedSubCategoryId = -1;
		float bestExpectedSubCategoryProba = -1.0f;
		bool isAccurate = true;
		bool previousIndexWasProba = false;

		while (subCategoriesFound < subCategoriesCount && (*pNexIndexToCheck < endIndexExcluded))
		{
			float expectedProba = expected[*pNexIndexToCheck];
			float predictedProba = predicted[*pNexIndexToCheck];
			if (fabsf(expectedProba) < 9.5f)
			{
				previousIndexWasProba = true;
				++subCategoriesFound;
				if (expectedProba > bestExpectedSubCategoryProba)
				{
					bestExpectedSubCategoryProba = expectedProba;
					expectedSubCategoryId = subCategoriesFound - 1;
				}
				if (predictedProba > bestPredictedSubCategoryProba)
				{
					bestPredictedSubCategoryProba = predictedProba;
					predictedSubCategoryId = subCategoriesFound - 1;
				}
				*pNexIndexToCheck += 1;
			}
			else
			{
				int count = (int)(fabsf(expectedProba) + 0.5f) / 10;
				if (expectedProba < 0)
				{
					//we need to skip 'count' indexes
					*pNexIndexToCheck += count;
				}
				else
				{
					*pNexIndexToCheck += 1;
					bool subCategoryIsAccurate = IsAccuratePredictionForCategoricalCrossentropyWithHierarchy(expected, predicted, endIndexExcluded, pNexIndexToCheck, count);
					isAccurate = subCategoryIsAccurate && isAccurate;
				}
				if (!previousIndexWasProba)
				{
					++subCategoriesFound;
				}
				previousIndexWasProba = false;
			}
		}
		return (expectedSubCategoryId == predictedSubCategoryId) && isAccurate;
	}

	__device__ inline float IsCountAssociateWithAboveProba(float f) { return f > 5.0f && ((int)(f + 0.1f)) % 10 == 1; }
	__device__ inline float IsProba(float f) { return fabsf(f) < 5.0f; }
	__device__ inline float ExtractCount(float f) { return (int)(fabsf(f) + 0.5f) / 10; }

	__global__ void ComputeSingleAccuracyForCategoricalCrossentropyWithHierarchy(int N, int nbCols, float* countOk, const float* __restrict expected, const float* __restrict predicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			int nexIndexToCheck = 0;
			countOk[i] = IsAccuratePredictionForCategoricalCrossentropyWithHierarchy(expected + i * nbCols, predicted + i * nbCols, nbCols, &nexIndexToCheck, 10000000)
				? 1.0f 
				: 0.0f;
		}
	}

	__device__  void SoftmaxWithHierarchy(const float* activationParameter, float* y, int endIndexExcluded, int* pNexIndexToCheck)
	{
		float param = activationParameter[*pNexIndexToCheck];
		y[*pNexIndexToCheck] = param;
		int subCategoriesCount = ExtractCount(param);
		*pNexIndexToCheck += 1;
		
		//we only allocate an array if we have more then '10' elements
		int smallIntArray[10];
		int* indexesProba = (subCategoriesCount > 10) ? (int*)malloc(subCategoriesCount * sizeof(int)) : (&smallIntArray[0]);

		float maxProba = -1e9f;
		bool probaFound = false;

		for (int subCategoriesFound = 0; subCategoriesFound < subCategoriesCount; ++subCategoriesFound)
		{
			float expectedProba = activationParameter[*pNexIndexToCheck];
			if (IsProba(expectedProba))
			{
				maxProba = fmaxf(maxProba, y[*pNexIndexToCheck]);
				indexesProba[subCategoriesFound] = *pNexIndexToCheck;
				probaFound = true;
				*pNexIndexToCheck += 1;
				if (*pNexIndexToCheck < endIndexExcluded && IsCountAssociateWithAboveProba(activationParameter[*pNexIndexToCheck]))
				{
					SoftmaxWithHierarchy(activationParameter, y, endIndexExcluded, pNexIndexToCheck);
				}
			}
			else
			{
				SoftmaxWithHierarchy(activationParameter, y, endIndexExcluded, pNexIndexToCheck);
			}
		}

		if (probaFound)
		{
			float sumExp = 0.0f;
			for (int i = 0; i < subCategoriesCount; ++i)
			{
				int idx = indexesProba[i];
				float tmp = expf(y[idx] - maxProba);
				sumExp += tmp;
				y[idx] = tmp;
			}
			for (int i = 0; i < subCategoriesCount; ++i)
			{
				y[indexesProba[i]] /= sumExp;
			}
		}

		if (subCategoriesCount > 10)
		{
			free(indexesProba);
		}
	}

	__global__ void ComputeSoftmaxWithHierarchy(int N, int nbCols, const float* activationParameter, float* y)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			int nexIndexToCheck = 0;
			SoftmaxWithHierarchy(activationParameter, y + i * nbCols, nbCols, &nexIndexToCheck);
		}
	}

	__global__ void ComputeSoftmaxGradientWitHierarchy(int N, int nbCols, const float* activationParameter, const float* y, const float* dy, float* dx)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			float expectedProba = activationParameter[i%nbCols];
			if (IsProba(expectedProba))
			{
				float dyi = dy[i];
				float yi = y[i];
				dx[i] = (fabsf(dyi - 1.0f) < 1e-6) ? (yi * (1 - yi)) : (-yi * dyi);
			}
			else
			{
				dx[i] = expectedProba;
			}
		}
	}

	__global__ void SwishGradient(int N, const float* __restrict Y, const float* __restrict dY, const float* __restrict X, float *dX) 
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < N) {
			float x = X[row];
		    float sigmoid_x = (fabs(x) < 0.0001f) ? 0.5f : Y[row] / x;
            dX[row] = dY[row] * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x));
		}
	}


	__global__ void ComputeLn(int N, const float* __restrict X, float* Y)
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < N) {
			float x = X[row];
			Y[row] = (x<=0) ? (-100.0f) : logf(x);
		}
	}

	__global__ void LnGradient(int N, const float* __restrict dY, const float* __restrict X, float* dX)
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < N) {
			float x = X[row];
			dX[row] = dY[row] / x;
		}
	}

	__global__ void Set1InMainDiagonal(int nbRows, int nbCols, float *result) 
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < nbRows && row < nbCols) {
			result[row*nbCols+row] = 1.0f;
		}
	}

	__global__ void SetAllElementsAboveMainDiagonal(int N, int rows_by_matrix, int cols_by_matrix, float valueForElementsAboveMainDiagonal, float *result) 
	{
		int current = blockIdx.x * blockDim.x + threadIdx.x;
		if (current<N) {
			int matrixId = current/rows_by_matrix;
			int row = current%rows_by_matrix;
			int idx = matrixId*rows_by_matrix*cols_by_matrix+row*cols_by_matrix;
			for(int col = row+1; col<cols_by_matrix; ++col) {
				result[idx+col] = valueForElementsAboveMainDiagonal;
			}
		}
	}

	__global__ void SetToZeroAllElementsBelowMainDiagonal(int nbRows, int nbCols, float *result) 
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < nbRows) {
			result += row*nbCols;
			int last_i = min(row, nbCols);
			for(int i=0;i<last_i;++i)
			{
				result[i] = 0.0f;
			}
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

	__global__ void Clip(int nbRows, int nbCols, float* result, float lower, float upper)
	{
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < nbRows) {
			result += row * nbCols;
			for (int i = 0; i < nbCols; ++i)
			{
				result[i] = fminf(fmaxf(result[i], lower), upper);
			}
		}
	}

	// src tensor (unpadded tensor) has shape (n, c, h_src, w_src)
	// dest tensor (padded tensor) has shape (n, c, h_dest, w_dest) with:
    //		h_dest = top_pad + h_src + bottom_pad;
    //      w_dest = left_pad + w_src + right_pad;
	// N = n*c*h_src = number of distinct rows in 'src' tensor
	__global__ void ApplyZeroPaddingForRowId(int N, int h_src, int w_src, int top_pad, int bottom_pad, int left_pad, int right_pad, float* paddedTensor, float* unpaddedTensor, bool isUnpadding) 
	{
		// 'rowId' is the index of the row in 'src' tensor (0 <= rowId < N with N=n*c*h_src)
		int rowId = blockIdx.x * blockDim.x + threadIdx.x;
		if (rowId < N) {
			//we'll copy the row 'rowId' from 'src' tensor (n, c, h_src, w_src) to 'dest' tensor (n, c, h_dest, w_dest)
            int h_dest = top_pad + h_src + bottom_pad;
            int w_dest = left_pad + w_src + right_pad;
            int row_in = (rowId % h_src);
            int destRowIdx = ((rowId / h_src) * h_dest + row_in + top_pad) * w_dest + left_pad;
            int rowIdx = rowId * w_src;
			if (isUnpadding)
				memcpy(unpaddedTensor+rowIdx, paddedTensor+destRowIdx, sizeof(float)*w_src);
			else
				memcpy(paddedTensor+destRowIdx, unpaddedTensor+rowIdx, sizeof(float)*w_src);
		}
	}


	//'x' shape:                (batchSize, timeSteps, inputSize)
	//'y' shape :               (batchSize, timeSteps, outputSize)
	//'wordEmbedding' shape:    (vocabularySize, embeddingDim)
	__global__ void WordEmbeddingForwardPropagation(int N, int inputSize, int outputSize, int xIndexInLastDimensionToUse, int yIndexInLastDimensionToUse, int copyCountBeforeIndex, int copyCountAfterIndex, int embeddingDim, float* x, float* y, float* wordEmbedding)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;	// in [0, batchSize*timeSteps-1]
		if (i >= N) return;

		int xIndex = xIndexInLastDimensionToUse + i * inputSize;
		int yIndex = yIndexInLastDimensionToUse + i * outputSize;
		//we retrieve the wordIndex 
		int wordIndex = (int)(x[xIndex] + 0.1f);	//in [0, vocabularySize-1]
		int indexInWordEmbedding = wordIndex*embeddingDim;

		//for the current timeStep, we copy the elements from 'x' to 'y' before 'indexInLastDimensionToUse'
		if (copyCountBeforeIndex > 0)
		{
			memcpy(y + yIndex-copyCountBeforeIndex, x + xIndex-copyCountBeforeIndex, sizeof(float) * copyCountBeforeIndex);
		}

		if (embeddingDim>0)
		{
			memcpy(y+ yIndex, wordEmbedding+indexInWordEmbedding, sizeof(float) * embeddingDim);
		}

		//for the current timeStep, we copy the elements from 'x' to 'y' after 'indexInLastDimensionToUse'
		if (copyCountAfterIndex > 0)
		{
			memcpy(y + yIndex+ embeddingDim, x + xIndex+1, sizeof(float) * copyCountAfterIndex);
		}
	}

	// N :						batchSize * timeSteps
	// 'x' & 'dx' shape :       (batchSize, timeSteps, inputSize)
	// 'dy' shape :             (batchSize, timeSteps, outputSize)
	//'dw' shape:				(vocabularySize, embeddingDim)
	__global__ void WordEmbeddingBackwardPropagation(int N, int inputSize, int outputSize, int xIndexInLastDimensionToUse, int yIndexInLastDimensionToUse, int copyCountBeforeIndex, int copyCountAfterIndex, int embeddingDim, float* x, float* dx, float* dy, float* dw)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;	// in [0, batchSize*timeSteps-1]
		if (i >= N) return;

		int dxIndex = xIndexInLastDimensionToUse + i * inputSize;

		//we retrieve the wordIndex 
		int wordIndex = (int)(x[dxIndex] + 0.1f);	//in [0, vocabularySize-1]
		int indexInDw = embeddingDim * wordIndex;

		int dyIndex = yIndexInLastDimensionToUse + i * outputSize;
		for (int embeddingId = 0; embeddingId < embeddingDim; ++embeddingId)
		{
			float valueToAdd = dy[dyIndex+embeddingId];
			atomicAdd(dw+ indexInDw, valueToAdd);
			++indexInDw;
		}

		//we initialize 'dx' for the current batchIndex & timeStep
		//for the current timeStep, we copy the elements from 'dy' to 'dx' before 'indexInLastDimensionToUse'
		if (copyCountBeforeIndex > 0)
		{
			memcpy(dx + dxIndex-copyCountBeforeIndex, dy + dyIndex-copyCountBeforeIndex, sizeof(float) * copyCountBeforeIndex);
		}
		dx[dxIndex] = 0.0f;
		//for the current timeStep, we copy the elements from 'dy' to 'dx' after 'indexInLastDimensionToUse'
		if (copyCountAfterIndex > 0)
		{
			memcpy(dx + dxIndex+ 1, dy + dyIndex + embeddingDim, sizeof(float) * copyCountAfterIndex);
		}
	}

	__global__ void YOLOV3Forward(int N, float* y, float* x, int x_c, int x_h, int x_w, int inputImageHeight, int inputImageWidth, int anchor0Width, int anchor0Height, int anchor1Width, int anchor1Height, int anchor2Width, int anchor2Height) 
	{
		int xpredictionIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (xpredictionIndex < N) {

			int nbAnchors = 3;
			int predictionLength = x_c/nbAnchors;
            int categories = predictionLength - 5;
            int rowStride = inputImageHeight / x_h;
            int colStride = inputImageWidth / x_w;
			int xpredictionIndexBackup = xpredictionIndex;
			int elementId = xpredictionIndex / (nbAnchors*x_h*x_w);
			xpredictionIndex = xpredictionIndex %(nbAnchors*x_h*x_w);
			int boxId = xpredictionIndex / (x_h*x_w);
			xpredictionIndex = xpredictionIndex %(x_h*x_w);
			int x_row = xpredictionIndex / (x_w);
			int x_col = xpredictionIndex %(x_w);

			xpredictionIndex= xpredictionIndexBackup;
			int xIndex = elementId*x_c*x_h*x_w + boxId*predictionLength*x_h*x_w + x_row*x_w  + x_col;
			int yIndex = elementId*x_c*x_h*x_w + x_row*x_c*x_w + x_col*x_c + boxId*predictionLength;

            //box center
            y[yIndex++] = (x_col + sigmoidf(x[xIndex])) * colStride;
            xIndex += x_h*x_w;
            y[yIndex++] = (x_row + sigmoidf(x[xIndex])) * rowStride;
            xIndex += x_h*x_w;

            //box size
            int anchorWidth = (boxId == 0) ? anchor0Width : ((boxId == 1) ? anchor1Width : anchor2Width);
            y[yIndex++] = anchorWidth * expf(x[xIndex]);
            xIndex += x_h*x_w;
            int anchorHeight = (boxId == 0) ? anchor0Height : ((boxId == 1) ? anchor1Height : anchor2Height);
            y[yIndex++] = anchorHeight * expf(x[xIndex]);
            xIndex += x_h*x_w;

            //box confidence
            y[yIndex++] = sigmoidf(x[xIndex]);
            xIndex += x_h*x_w;

            //categories
            for (int i = 0; i < categories; ++i)
            {
                y[yIndex++] = sigmoidf(x[xIndex]);
                xIndex += x_h*x_w;
            }
		}
	}


	// src tensor (tensor before up sampling) has shape (n, c, h_src, w_src)
	// dest tensor (tensor after upsampling) has shape (n, c, rowFactor*h_src, colFactor*w_dest)
	// isUpscaling : true if we are up sampling (from 'src' to 'dest') / false if we are down sampling (from 'dest' to 'src')
	__global__ void UpSampling2D(int N, int channels, int h_src, int w_src, int rowFactor, int colFactor, float* src, float* dest, bool isUpscaling) 
	{
		int srcIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (srcIndex < N) {
			int h_dest = h_src * rowFactor;
			int w_dest = w_src * colFactor;
			float originalElement = src[srcIndex];
			int srcIndexbackup = srcIndex;
	
			int elementId = srcIndex / (channels*h_src*w_src);
			srcIndex = srcIndex %(channels*h_src*w_src);
			int channel = srcIndex / (h_src*w_src);
			srcIndex = srcIndex %(h_src*w_src);
			int row_src = srcIndex / (w_src);
			int col_src = srcIndex %(w_src);
			srcIndex = srcIndexbackup;
			float sum = 0; //only used when down sampling (isUpscaling = false)

			int startOfRow = elementId*(channels*h_dest*w_dest)+channel*(h_dest*w_dest)+ row_src*rowFactor *w_dest + col_src* colFactor;
			for(int rowOffset=0;rowOffset<rowFactor;++rowOffset)
			{
				int idx_dest = startOfRow;
				for(int colOffset=0;colOffset<colFactor;++colOffset)
				{
					if (isUpscaling)
						dest[idx_dest] = originalElement;
					else
						sum += dest[idx_dest];
					++idx_dest;
				}
				startOfRow += w_dest;
			}
			if (!isUpscaling)
				src[srcIndex] = sum;
		}
	}

	__global__ void BinaryCrossentropyLossBuffer(int N, int numClass, float* lossBuffer, const float* __restrict yExpectedOneHot, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			float loss = 0.0f;
			int startIndex = i * numClass;
			int endIndexExcluded = startIndex + numClass;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float predicted = yPredicted[j];
				float expected = yExpectedOneHot[j];
				//if ((predicted>0.01)&&(predicted<0.99f))
				if ((predicted > 0.0f) && (predicted < 1.0f))
					loss -= (expected * logf(predicted) + (1.0f - expected) * logf(1.0f - predicted)) / numClass;
			}
			lossBuffer[i] = loss;
		}
	}

	__global__ void CategoricalCrossentropyLossBuffer(int N, int nummClass, float *lossBuffer, const float* __restrict yExpectedOneHot, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			float loss = 0.0f;
			int startIndex = i * nummClass;
			int endIndexExcluded = startIndex + nummClass;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float predicted = yPredicted[j];
				float expected = yExpectedOneHot[j];
				if (predicted > 0)
					loss -= expected * logf(predicted);
			}
			lossBuffer[i] = loss;
		}
	}

	
	__global__ void CategoricalCrossentropyWithHierarchyLossBuffer(int N, int nbCols, float* lossBuffer, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			float loss = 0.0f;
			int startIndex = i * nbCols;
			int endIndexExcluded = startIndex + nbCols;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float expected = yExpected[j];
				if (fabsf(expected) < 9.5f)
				{
					if (expected > 1e-6f)
					{
						//expected contains a proba between 0 and 1
						float predicted = yPredicted[j];
						loss += expected * logf(fmaxf(1e-6f, predicted));
					}
				}
				else
				{
					if (expected < 0) 
					{
						//expected contains a description : there is no associated loss
						int count = (int)(fabsf(expected) + 0.5f) / 10;
						//we need to skip 'count' indexes
						j += count - 1; //-1 because the for(;;) loop will also increment 'j'
					}
				}
			}
			lossBuffer[i] = -loss;
		}
	}

	__global__ void CategoricalCrossentropyWithHierarchyGradient(int N, int nbCols, float* loss, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			int startIndex = i * nbCols;
			int endIndexExcluded = startIndex + nbCols;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float expected = yExpected[j];
				if (fabsf(expected) < 9.5f)
				{
					//expected contains a proba between 0 and 1
					loss[j] = yPredicted[j] - expected;
				}
				else
				{
					if (expected < 0)
					{
						//expected contains a number of element to skip: there is no associated loss
						int count = (int)(fabsf(expected) + 0.5f) / 10;
						//we need to skip 'count' indexes
						j += count - 1; //-1 because the for(;;) loop will also increment 'j'
					}
				}
			}
		}
	}

	__global__ void HuberLossBuffer(int batchSize, int lineSize, float huberDelta, float* lossBuffer, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < batchSize) {
			int startIndex = i * lineSize;
			int endIndexExcluded = startIndex + lineSize;
			float loss = 0.0f;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float diff = yExpected[j] - yPredicted[j];
				if (fabsf(diff) <= huberDelta)
					loss += 0.5f * diff * diff;
				else
					loss += huberDelta * fabs(diff) - 0.5f * huberDelta * huberDelta;
			}
			lossBuffer[i] = loss;
		}
	}

	__global__ void HuberGradient(int batchSize, int lineSize, float huberDelta, float* huberGradient, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < batchSize) {
			int startIndex = i * lineSize;
			int endIndexExcluded = startIndex + lineSize;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float diff = yPredicted[j] - yExpected[j];
				huberGradient[j] = fmaxf(fminf(diff, huberDelta), -huberDelta) / lineSize;
			}
		}
	}

	__global__ void MseLossBuffer(int batchSize, int lineSize, float* lossBuffer, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < batchSize) {
			int startIndex = i * lineSize;
			int endIndexExcluded = startIndex + lineSize;
			float loss = 0.0f;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float diff = yExpected[j] - yPredicted[j];
				loss += diff * diff;
			}
			lossBuffer[i] = loss / lineSize;
		}
	}

	
	__global__ void MseGradient(int batchSize, int lineSize, float* mseGradient, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < batchSize) {
			int startIndex = i * lineSize;
			int endIndexExcluded = startIndex + lineSize;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float diff = yPredicted[j] - yExpected[j];
				mseGradient[j] = (2 * diff) / lineSize;
			}
		}
	}

	__global__ void SparseCategoricalCrossentropyLossBuffer(int N, int numClass, float *lossBuffer, const float* __restrict yExpectedSparseMatrix, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
            int yClass = (int)(yExpectedSparseMatrix[i]+0.1f);
			float predicted = yPredicted[i * numClass + yClass];
			lossBuffer[i] = -logf(predicted);
		}
	}

	__global__ void SparseCategoricalCrossentropyGradient(int batchSize, int numClass, float* sparseCategoricalCrossentropyGradient, const float* __restrict yExpectedSparseMatrix, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < batchSize) {
			int startIndex = i * numClass;
			int endIndexExcluded = startIndex + numClass;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				sparseCategoricalCrossentropyGradient[j] = yPredicted[j];
			}
            int yClass = (int)(yExpectedSparseMatrix[i]+0.1f);
            sparseCategoricalCrossentropyGradient[i * numClass + yClass] -= 1.0f;
		}
	}

	__global__ void CosineSimilarityLossBuffer(int timeSeriesLength, int yExpectedLength, float* lossBuffer, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int day = blockIdx.x * blockDim.x + threadIdx.x;
		if (day < timeSeriesLength) {
			float top = 0.0f;
            float expectedSquares = 0.0f;
            float predictedSquares = 0.0f;
            for (int t = day; t < yExpectedLength; t+= timeSeriesLength)
            {
                float pred = yPredicted[t];
                float exp = yExpected[t];
                top += pred * exp;
                expectedSquares += exp * exp;
                predictedSquares += pred * pred;
            }
            float l2_norm_expected = sqrtf(expectedSquares);
            float l2_norm_predicted = sqrtf(predictedSquares);
            lossBuffer[day] = top / (l2_norm_expected * l2_norm_predicted);
		}
	}

	__global__ void CosineSimilarityGradient(int timeSeriesLength, int yExpectedLength, float* cosineSimilarityGradient, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int day = blockIdx.x * blockDim.x + threadIdx.x;
		if (day < timeSeriesLength) {
			double top = 0.0;
            double expectedSquares = 0.0;
            double predictedSquares = 0.0;
            for (int t = day; t < yExpectedLength; t+= timeSeriesLength)
            {
                double pred = yPredicted[t];
                double exp = yExpected[t];
                top += pred * exp;
                expectedSquares += exp * exp;
                predictedSquares += pred * pred;
            }
            double l2_norm_expected = sqrt(expectedSquares);
            double l2_norm_predicted = sqrt(predictedSquares);
            double multiplier1 = 1.0/(l2_norm_expected * l2_norm_predicted);
            double mutliplier2 = (-top)/(l2_norm_predicted* l2_norm_predicted * l2_norm_predicted * l2_norm_expected);
            for (int t = day; t < yExpectedLength; t += timeSeriesLength)
            {
                cosineSimilarityGradient[t] = - (float)(multiplier1*yExpected[t] + mutliplier2*yPredicted[t]);
            }
		}
	}

	__global__ void MeanSquaredLogErrorLossBuffer(int batchSize, int lineSize, float* lossBuffer, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < batchSize) {
			int startIndex = i * lineSize;
			int endIndexExcluded = startIndex + lineSize;
			float loss = 0.0f;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float diff = logf(1+yPredicted[j]) - logf(1+yExpected[j]);
				loss += diff * diff;
			}
			lossBuffer[i] = loss / lineSize;
		}
	}

	__global__ void MseOfLogLossBuffer(int batchSize, int lineSize, float* lossBuffer, const float* __restrict yExpected, const float* __restrict yPredicted, float epsilon)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < batchSize) {
			int startIndex = i * lineSize;
			int endIndexExcluded = startIndex + lineSize;
			float loss = 0.0f;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float adjustedPredicted = fmaxf(epsilon, yPredicted[j]);
				float diff = logf(adjustedPredicted) - logf(yExpected[j]);
				loss += diff * diff;
			}
			lossBuffer[i] = loss / lineSize;
		}
	}


	__global__ void MseOfLogGradient(int batchSize, int lineSize, float* mseGradient, const float* __restrict yExpected, const float* __restrict yPredicted, float epsilon)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < batchSize) {
			int startIndex = i * lineSize;
			int endIndexExcluded = startIndex + lineSize;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float adjustedPredicted = fmaxf(epsilon, yPredicted[j]);
				float diff = logf(adjustedPredicted) - logf(yExpected[j]);
				mseGradient[j] = (2 * diff) / (adjustedPredicted*lineSize);
			}
		}
	}


	__global__ void MaeLossBuffer(int batchSize, int lineSize, float* lossBuffer, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < batchSize) {
			int startIndex = i * lineSize;
			int endIndexExcluded = startIndex + lineSize;
			float loss = 0.0f;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float diff = yExpected[j] - yPredicted[j];
				loss += fabsf(diff);
			}
			lossBuffer[i] = loss / lineSize;
		}
	}

	__global__ void MaeGradient(int batchSize, int lineSize, float* mseGradient, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < batchSize) {
			int startIndex = i * lineSize;
			int endIndexExcluded = startIndex + lineSize;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float diff = yPredicted[j] - yExpected[j];
				mseGradient[j] = (diff>=0?1.0f:-1.f) / lineSize;
			}
		}
	}


	// Compute:  y = slope * x + intercept
	__global__ void LinearFunction(int N, float* y, float slope, const float* x, float intercept)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			y[i] = slope * x[i] + intercept;
		}
	}

	__global__ void Concatenate(int N, int m, float* __restrict concat, int concatMultDim0, const float* __restrict a, int aMultDim0, const float* __restrict b, int bMultDim0)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N)  return;
		int row = i/concatMultDim0;
		int colInConcat = i%concatMultDim0;
		if (colInConcat<aMultDim0)
			concat[i] = a[row*aMultDim0+colInConcat];
		else
			concat[i] = b[row*bMultDim0+colInConcat-aMultDim0];

	}

	__global__ void Concatenate3(int N, int m, float* __restrict concat, int concatMultDim0, const float* __restrict a, int aMultDim0, const float* __restrict b, int bMultDim0, const float* __restrict c, int cMultDim0)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N)  return;
		int row = i/concatMultDim0;
		int colInConcat = i%concatMultDim0;
		if (colInConcat<aMultDim0)
			concat[i] = a[row*aMultDim0+colInConcat];
		else
			concat[i] = (colInConcat<(aMultDim0+bMultDim0))?b[row*bMultDim0+colInConcat-aMultDim0]:c[row*cMultDim0+colInConcat-aMultDim0-bMultDim0];
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

	__global__ void Split3(int N, int m, const float* __restrict concat, int concatMultDim0, float* __restrict a, int aMultDim0, float* __restrict b, int bMultDim0, float* __restrict c, int cMultDim0)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N)  return;
		int row = i/concatMultDim0;
		int colInConcat = i%concatMultDim0;
		if (colInConcat<aMultDim0)
			a[row*aMultDim0+colInConcat] = concat[i];
		else if (colInConcat<(aMultDim0+bMultDim0))
			b[row*bMultDim0+colInConcat-aMultDim0] = concat[i];
		else
			c[row*cMultDim0+colInConcat-aMultDim0-bMultDim0] = concat[i];
	}

	// transform a 'src' tensor of shape [a,b,c] to a 'target' tensor of shape [b,a,c] 
	__global__ void Switch_First_2_axis(int N, int aLength, int bLength, int cLength, const float* __restrict src, float *target)
	{
		int idx_src = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx_src < N) {
			int multDim0 = bLength * cLength;
			int a_src = idx_src / multDim0;
			int tmp = idx_src % multDim0;
			int b_src = tmp / cLength;
			int c_src = tmp % cLength;
			int idx_target = b_src * aLength * cLength + a_src * cLength + c_src;
			target[idx_target] = src[idx_src];
		}
	}

	// transform a 'src' tensor of shape [n,c,h] to a 'target' tensor of shape [n,h,c] 
	__global__ void SwitchSecondAndThirdDimension(int N, int nLength, int cLength, int hLength, const float* __restrict src, float* target)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= N)
			return;
		int n = i / cLength;
		int c = i % cLength;
		for(int h=0;h<hLength;++h)
		{
			int src_index = h + c * hLength + n * cLength * hLength;
			int target_index = c + h * cLength + n * cLength * hLength;
			target[target_index] = src[src_index];
		}
	}

	// add positional encoding to 'src' tensor of shape (batchSize, timeSteps, embeddingDim)
	__global__ void UpdateWithPositionalEncoding_AttnIsAllYouNeed(int N, int timeSteps, int embeddingDim, int n, float* __restrict src)
	{
		int src_index = blockIdx.x * blockDim.x + threadIdx.x;
		if (src_index >= N)
			return;
		int timeSteps_embeddingDim = src_index % (timeSteps*embeddingDim);
		int k = timeSteps_embeddingDim / embeddingDim;
		int col = timeSteps_embeddingDim %embeddingDim;
		int i = col / 2;
		float value = col % 2 == 0 
                            ? sinf(k  / powf(n, (2.0f * i) / embeddingDim)) 
                            : cosf(k / powf(n, (2.0f * i) / embeddingDim));
		src[src_index] += value;
	}

	// transform a 'src' tensor of shape >= 3 [A,B,C,*] to a 'target' tensor of shape [A,C,B,*] 
	__global__ void TransposeSecondAndThirdDimension_V1(int N, int B, int C, int D, const float* __restrict src, float* target)
	{

		int src_index = blockIdx.x * blockDim.x + threadIdx.x;
		if (src_index >= N)
			return;
		//int a = src_index / (B*C*D);
		int bcd = src_index % (B*C*D);

		int b = bcd / (C*D);
		int cd = bcd %(C*D);
		int c = cd / D;

		int target_index= src_index+D*(  c*(B-1)-b*(C-1) );
		target[target_index] = src[src_index];
	}

	// transform a 'src' tensor of shape >= 3 [A,B,C,*] to a 'target' tensor of shape [A,C,B,*] 
	__global__ void TransposeSecondAndThirdDimension_V2(int N, int B, int C, int D, const float* __restrict src, float* target)
	{
		int abc = blockIdx.x * blockDim.x + threadIdx.x;
		if (abc >= N)
			return;
		int bc = abc % (B*C);
		int b = bc / C;
		int c = bc %C;
		int src_index = abc*D;
		int target_index= src_index+D*(  c*(B-1)-b*(C-1) );
		for(int d=0;d<D;++d)
		{
			target[target_index+d] = src[src_index+d];
		}
	}

	// transform a 'src' tensor of shape >= 3 [A,B,C,*] to a 'target' tensor of shape [A,C,B,*] 
	__global__ void TransposeSecondAndThirdDimension_V3(int N, int B, int C, int D, const float* __restrict src, float* target)
	{
		int ab = blockIdx.x * blockDim.x + threadIdx.x;
		if (ab >= N)
			return;
		int b = ab % (B);
		for(int cd=0;cd<C*D;++cd)
		{
			int src_index = ab*C*D+cd;
			int c = cd/D;
			int target_index= src_index+D*(  c*(B-1)-b*(C-1) );
			target[target_index] = src[src_index];
		}
	}

	__device__ void warpReduce(volatile float* sdata, int tid) {
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}


	__global__ void Compute_Row_Mean_Variance_V2(int N, float* g_idata, float* mean, float* variance, int cols, int nextColsPowerOf2, bool unbiasedVariance) {

		extern __shared__ float sdata[];
		int tid = threadIdx.x;
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		float* sum_data = sdata;
		float* sumSquare_data = &sdata[cols+1];

		//the current thread 'tid' will take care of all elements in current row in range:
		//	[ start_col_idx, end_col_idx [
		int start_col_idx = (tid * cols) / blockDim.x;
		int end_col_idx = ((tid + 1) * cols) / blockDim.x;

		if (idx < N)
		{
			int g_idata_idx = blockIdx.x * cols + start_col_idx;
			for (int col_idx = start_col_idx; col_idx < end_col_idx; col_idx++)
			{
				float tmp = g_idata[g_idata_idx++];
				sum_data[col_idx] = tmp;
				sumSquare_data[col_idx] = tmp * tmp;
			}
		}
		__syncthreads();

		int min_stride_excluded = cols < 32 ? 0 : 32;
		for (int stride = nextColsPowerOf2 / 2; stride > min_stride_excluded; stride >>= 1) {

			int max_cold_idx = min(min(stride, cols - stride), end_col_idx);
			for (int col_idx = start_col_idx; col_idx < max_cold_idx; col_idx++)
			{
				sum_data[col_idx] += sum_data[col_idx + stride];
				sumSquare_data[col_idx] += sumSquare_data[col_idx + stride];
			}
			__syncthreads();
		}
		if (tid < 32 && min_stride_excluded == 32) {
			warpReduce(sum_data, tid);
			warpReduce(sumSquare_data, tid);
		}

		// write result for this block to global mem
		if (tid == 0) {

			int rowId = blockIdx.x;
			float row_mean = (float)(sum_data[0] / cols);
			mean[rowId] = row_mean;
			int divider = unbiasedVariance ? (cols - 1) : cols;
			float row_variance = (float)abs(sumSquare_data[0] - cols * row_mean * row_mean) / divider;
			variance[rowId] = row_variance;
		}
	}
}



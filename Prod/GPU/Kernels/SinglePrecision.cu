extern "C" {

	__device__ inline float sigmoidf(float x) {
		return 1.0f / (1 + expf(-x));
	}

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

	//'y' shape :               (batchSize, embeddingDim, maxWordCountBySentence)
	//'x' shape:                (batchSize, maxWordCountBySentence)
	//'wordEmbedding' shape:    (vocabularySize, embeddingDim)
	__global__ void WordEmbeddingForwardPropagation(int N, int batchSize, int maxWordCountBySentence, int embeddingDim, int vocabularySize, float* y, float* x, float* wordEmbedding)
	{
		int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (xIndex >= N) return;
		int xRow = xIndex / maxWordCountBySentence; //sentenceId in [0, batchSize-1]
		int xCol = xIndex % maxWordCountBySentence; //word position in sentence, in [0, maxWordCountBySentence-1]
		int wordIndex = (int)(x[xIndex] + 0.1f);	//in [0, vocabularySize-1]
		int indexInWordEmbedding = wordIndex* embeddingDim;
		int indexInY = xRow*(embeddingDim*maxWordCountBySentence)+ xCol;
		for (int embeddingId = 0; embeddingId < embeddingDim; ++embeddingId)
		{
			y[indexInY] = wordEmbedding[indexInWordEmbedding];
			indexInY += maxWordCountBySentence;
			++indexInWordEmbedding;
		}
	}

	//'dw' shape:				(VocabularySize, EmbeddingDim)
	// x shape :                (batchSize,  maxWordCountBySentence)
	// dy shape :               (batchSize, EmbeddingDim,  maxWordCountBySentence)
	__global__ void WordEmbeddingBackwardPropagation(int N, int batchSize, int maxWordCountBySentence, int embeddingDim, int vocabularySize, float* dw, float* x, float* dy)
	{
		int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (xIndex >= N) return;
		int xRow = xIndex / maxWordCountBySentence; //sentenceId, in [0, batchSize-1]
		int xCol = xIndex % maxWordCountBySentence; //word position in sentence in [0, maxWordCountBySentence-1]
		int wordIndex = (int)(x[xIndex] + 0.1f);	//in [0, vocabularySize-1]
		int dwIndex = embeddingDim * wordIndex;
		int dyIndex = xRow * (maxWordCountBySentence * embeddingDim) + xCol;
		for (int embeddingId = 0; embeddingId < embeddingDim; ++embeddingId)
		{
			float valueToAdd = dy[dyIndex];
			atomicAdd(dw+dwIndex, valueToAdd);
			++dwIndex;
			dyIndex += maxWordCountBySentence;
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

	__global__ void ComputeHuberLoss(int N, int categoryCount, float huberDelta, float* losses, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			float loss = 0.0f;
			int startIndex = i * categoryCount;
			int endIndexExcluded = startIndex + categoryCount;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float predicted = yPredicted[j];
				float expected = yExpected[j];
				float diff = expected - predicted;
				if (fabsf(diff) <= huberDelta)
					loss += 0.5f * diff * diff;
				else
					loss += huberDelta*fabs(diff)-0.5f* huberDelta * huberDelta;
			}
			losses[i] = loss;
		}
	}


	__global__ void ComputeLossForCategoricalCrossentropyWithHierarchy(int N, int nbCols, float* losses, const float* __restrict yExpected, const float* __restrict yPredicted)
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
			losses[i] = -loss;
		}
	}

	__global__ void ComputeBackwardPropagationLossCategoricalCrossentropyWithHierarchy(int N, int nbCols, float* loss, const float* __restrict yExpected, const float* __restrict yPredicted)
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
					loss[j] = yPredicted[j]- expected;
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

	__global__ void ComputeBackwardPropagationLossHuber(int N, int nbCols, float huberDelta,  float* loss, const float* __restrict yExpected, const float* __restrict yPredicted)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N) {
			int startIndex = i * nbCols;
			int endIndexExcluded = startIndex + nbCols;
			for (int j = startIndex; j < endIndexExcluded; ++j)
			{
				float diff = yPredicted[j] - yExpected[j];
				loss[j] = fmaxf(fminf(diff, huberDelta), -huberDelta);
			}
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
}


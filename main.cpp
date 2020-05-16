#include <iostream>
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <tuple>
#include "public_interface_channels_dollar.h"

class SlideWinHOGDescriptor
{
private:
	cv::Size winSize_slidewin_;
	cv::Size winStride_slidewin_;
	cv::Size padding_slidewin_;
	cv::Size blockSize_hog_;
	cv::Size blockStride_hog_;
	cv::Size cellSize_hog_;
	int nbins_hog_;

	const int derivAperture_ = 1;
	const double winSigma_ = -1;
	const cv::HOGDescriptor::HistogramNormType histogramNormType_ = cv::HOGDescriptor::L2Hys;
	const double L2HysThreshold_ = 0.2;
	const bool gammaCorrection_ = false;
	const int nlevels_ = cv::HOGDescriptor::DEFAULT_NLEVELS;
	const bool signedGradient_ = false;

	cv::HOGDescriptor hogObj;

	static inline int gcd(int a, int b)
	{
		if (a < b)
			std::swap(a, b);
		while (b > 0)
		{
			int r = a % b;
			a = b;
			b = r;
		}
		return a;
	}

	static inline size_t alignSize(size_t sz, int n)
	{
		CV_DbgAssert((n & (n - 1)) == 0); // n is a power of 2
		return (sz + n - 1) & -n;
	}

public:

	SlideWinHOGDescriptor(cv::Size winSize_slidewin=cv::Size(64,128), cv::Size winStride_slidewin=cv::Size(8,8), cv::Size padding_slidewin=cv::Size(0,0), cv::Size cellSize_hog=cv::Size(8,8), cv::Size blockSize_hog=cv::Size(16,16), cv::Size blockStride_hog=cv::Size(8,8), int nbins_hog=9)
		:hogObj(winSize_slidewin, blockSize_hog, blockSize_hog, cellSize_hog, nbins_hog, derivAperture_, winSigma_, histogramNormType_, L2HysThreshold_, gammaCorrection_, nlevels_, signedGradient_)
	{
		winSize_slidewin_ = winSize_slidewin;
		winStride_slidewin_ = winStride_slidewin;
		padding_slidewin_ = padding_slidewin;
		blockSize_hog_ = blockSize_hog;
		blockStride_hog_ = blockStride_hog;
		cellSize_hog_ = cellSize_hog;
		nbins_hog_ = nbins_hog;		
	}
	
	int numWindowsInImage(const cv::Size& imageSize)
	{
		cv::Size numWindows_XY_directions((imageSize.width - winSize_slidewin_.width) / winStride_slidewin_.width + 1,
			(imageSize.height - winSize_slidewin_.height) / winStride_slidewin_.height + 1);
		return numWindows_XY_directions.area();
	}

	cv::Rect getWindow(const cv::Size& imageSize, int idx)
	{
		int nwindowsX = (imageSize.width - winSize_slidewin_.width) / winStride_slidewin_.width + 1;
		int y = idx / nwindowsX;
		int x = idx - nwindowsX * y;
		return cv::Rect(x*winStride_slidewin_.width, y*winStride_slidewin_.height, winSize_slidewin_.width, winSize_slidewin_.height);
	}

	std::tuple<cv::Mat, cv::Mat> convert_to_Mat_tuple(std::vector<float>& descriptors, const std::vector<cv::Point>& locations)
	{
		int nwindows = locations.size();
		int ndims_feat = hogObj.getDescriptorSize(); // ndims_feat is always equal to descriptors.size() / locations.size()
		cv::Mat feats(nwindows, ndims_feat, CV_32FC1);
		cv::Mat dr(nwindows, 4, CV_32SC1);
		float* ptr_descriptors = descriptors.data();
		float* ptr_feats = feats.ptr<float>(0);
		int* ptr_dr = dr.ptr<int>(0);
		for (size_t i = 0; i < nwindows; i++)
		{
			int idx_start = i * ndims_feat;
			std::copy_n(ptr_descriptors + idx_start, ndims_feat, ptr_feats + idx_start);
			int* ptr_row_dr_cur = ptr_dr + i * 4;
			cv::Point location = locations[i];
			ptr_row_dr_cur[0] = location.x;
			ptr_row_dr_cur[1] = location.y;
			ptr_row_dr_cur[2] = winSize_slidewin_.width;
			ptr_row_dr_cur[3] = winSize_slidewin_.height;
		}

		return std::make_tuple(feats, dr);
	}


	std::tuple<std::vector<float>, std::vector<cv::Point>> feat_extract(const cv::Mat& img)
	{		
		cv::Size cacheStride(gcd(winStride_slidewin_.width, blockStride_hog_.width), gcd(winStride_slidewin_.height, blockStride_hog_.height));
		cv::Size imgSize = img.size();		
		padding_slidewin_.width = (int)alignSize(std::max(padding_slidewin_.width, 0), cacheStride.width);
		padding_slidewin_.height = (int)alignSize(std::max(padding_slidewin_.height, 0), cacheStride.height);
		cv::Size paddedImgSize(imgSize.width + padding_slidewin_.width * 2, imgSize.height + padding_slidewin_.height * 2);
		size_t nwindows = numWindowsInImage(paddedImgSize);
		std::vector<cv::Point> locations(nwindows);
		for (size_t i = 0; i < nwindows; i++)
			locations[i] = getWindow(paddedImgSize, (int)i).tl() - cv::Point(padding_slidewin_);
				
		std::vector<float> descriptors;
		hogObj.compute(img, descriptors, winStride_slidewin_, padding_slidewin_, locations);

		return std::make_tuple(descriptors, locations);
	}
  
	std::tuple<cv::Mat, cv::Mat> feat_extract_multiScale(const cv::Mat& img, double scaleratio=std::pow(2,1.0/8), int max_numScales = 1000)
	{
		int num_scales = fmin(floor(log(static_cast<double>(img.rows) / winSize_slidewin_.height) / log(scaleratio)),
			floor(log(static_cast<double>(img.cols) / winSize_slidewin_.width) / log(scaleratio))) + 1;

		num_scales = std::min(num_scales, max_numScales);

		std::vector<std::vector<float>> descriptors_allScales(num_scales);
		std::vector<std::vector<cv::Point>> locations_allScales(num_scales);
		std::vector<int> nwindows_allScales(num_scales);
		std::vector<double> scales(num_scales);
		size_t total_nwindows_allScales = 0;

		for (size_t s = 0; s < num_scales; s++)
		{
			double scale = std::pow(scaleratio, s);
			cv::Mat img_resized;
			cv::resize(img, img_resized, cv::Size(), 1.0/ scale, 1.0/ scale, cv::INTER_LINEAR);
			std::tie(descriptors_allScales[s], locations_allScales[s]) = feat_extract(img_resized);
			int nwindows = locations_allScales[s].size();
			scales[s] = scale;
			nwindows_allScales[s] = nwindows;
			total_nwindows_allScales += nwindows;
		}

		int ndims_feat = hogObj.getDescriptorSize(); // ndims_feat is always equal to descriptors.size() / locations.size()
		cv::Mat feats(total_nwindows_allScales, ndims_feat, CV_32FC1);
		cv::Mat dr(total_nwindows_allScales, 4, CV_32SC1);
		float* ptr_feats = feats.ptr<float>(0);
		int* ptr_dr = dr.ptr<int>(0);
		
		size_t countUp_total_nwindows_allScales = 0;

		int c = 0;
		for (size_t s = 0; s < num_scales; s++)
		{
			std::copy(descriptors_allScales[s].begin(), descriptors_allScales[s].end(), ptr_feats + countUp_total_nwindows_allScales * ndims_feat);
			
			std::vector<cv::Point>& locations = locations_allScales[s];
			double scale = scales[s];
			int nwindows = nwindows_allScales[s];			
			int idx_start_ptr_row_dr = countUp_total_nwindows_allScales * 4;

			for (size_t i = 0; i < nwindows; i++)
			{
				int* ptr_row_dr_cur = ptr_dr + idx_start_ptr_row_dr + i * 4;
				cv::Point location = locations[i];
				ptr_row_dr_cur[0] = std::round(location.x * scale);
				ptr_row_dr_cur[1] = std::round(location.y * scale);
				ptr_row_dr_cur[2] = std::round(winSize_slidewin_.width * scale);
				ptr_row_dr_cur[3] = std::round(winSize_slidewin_.height * scale);
			}

			countUp_total_nwindows_allScales += nwindows;
		}
		
		return std::make_tuple(feats, dr);			   
	}

	//std::tuple<cv::Mat, cv::Mat> feat_extract_multiScale_ver2(const cv::Mat& img, double scaleratio = std::pow(2, 1.0 / 8), int max_numScales = 1000)
	//{
	//	int nrows_img = img.rows;
	//	int ncols_img = img.cols;

	//	int num_scales = fmin(floor(log(static_cast<double>(nrows_img) / winSize_slidewin_.height) / log(scaleratio)),
	//		floor(log(static_cast<double>(ncols_img) / winSize_slidewin_.width) / log(scaleratio))) + 1;

	//	num_scales = std::min(num_scales, max_numScales);

	//	// find a tight upper bound on total no. of sliding windows needed
	//	double stride_scale_width, stride_scale_height, nsw_rows, nsw_cols;
	//	size_t nslidewins_cur_scale, nslidewins_total = 0;
	//	size_t nslidewins_total_ub = 0;
	//	for (size_t s = 0; s < num_scales; s++)
	//	{
	//		stride_scale_width = winStride_slidewin_.width * std::pow(scaleratio, s);
	//		stride_scale_height = winStride_slidewin_.height * std::pow(scaleratio, s);
	//		nsw_rows = std::floor(nrows_img / stride_scale_height) - std::floor(winSize_slidewin_.height / stride_scale_height) + 1;
	//		nsw_cols = std::floor(ncols_img / stride_scale_width) - std::floor(winSize_slidewin_.width / stride_scale_width) + 1;
	//		// Without the increment below, I get exact computation of number of sliding
	//		// windows, but just in case (to upper bound it)
	//		++nsw_rows; ++nsw_cols;
	//		size_t nslidewins_est_cur_scale = nsw_rows * nsw_cols;
	//		nslidewins_total_ub += nslidewins_est_cur_scale;
	//	}

	//	int ndims_feat = hogObj.getDescriptorSize(); 
	//	cv::Mat feats(nslidewins_total, ndims_feat, CV_32FC1);
	//	cv::Mat dr(nslidewins_total, 4, CV_32SC1);

	//	float* ptr_feats = feats.ptr<float>(0);
	//	int* ptr_dr = dr.ptr<int>(0);
	//	
	//	int countUp_total_nwindows_allScales = 0;

	//	for (size_t s = 0; s < num_scales; s++)
	//	{
	//		// compute how much I need to scale the original image for this current scale s
	//		double scale = std::pow(scaleratio, s);
	//		cv::Mat img_resized;
	//		// get the resized version of the original image with the computed scale
	//		cv::resize(img, img_resized, cv::Size(), 1.0 / scale, 1.0 / scale, cv::INTER_LINEAR);

	//		std::vector<cv::Point> locations;
	//		locations.reserve(nslidewins_allScales[s]);

	//		// run sliding window
	//		for (size_t i = 0; i < nrows_img - winSize_slidewin_.height + 1; i += winStride_slidewin_.height)
	//		{
	//			for (size_t j = 0; j < ncols_img - winSize_slidewin_.width + 1; j += winStride_slidewin_.width)
	//			{
	//				int *ptr_row_dr_cur = ptr_dr + countUp_total_nwindows_allScales * 4;
	//				ptr_row_dr_cur[0] = std::round(j * scale);
	//				ptr_row_dr_cur[1] = std::round(i * scale);
	//				ptr_row_dr_cur[2] = std::round(winSize_slidewin_.width * scale);
	//				ptr_row_dr_cur[3] = std::round(winSize_slidewin_.height * scale);

	//				locations.push_back(cv::Point(j, i));
	//			} // end j
	//		} //end i

	//		std::vector<float> descriptors;
	//		hogObj.compute(img, descriptors, winStride_slidewin_, padding_slidewin_, locations);

	//		std::copy(descriptors.begin(), descriptors.end(), ptr_feats + countUp_total_nwindows_allScales * ndims_feat);

	//		countUp_total_nwindows_allScales += nslidewins_allScales[s];

	//	} //end s

	//	return std::make_tuple(feats, dr);

	//}

};

struct Result_hogSlideWin
{
	float* feats;
	int* dr;
	int nslidewins;
	int ndims_feat;
};

Result_hogSlideWin* extract_hog_multiscale_opencv(unsigned char* img_row_major, int nrows_img, int ncols_img, int nchannels_img, int nrows_winsize=128, int ncols_winsize=64, int nrows_winstride=8, int ncols_winstride=8, double scaleratio=std::pow(2, 1.0 / 8), int max_numScales=1000)
{
	cv::Size winSize_slidewin = cv::Size(ncols_winsize, nrows_winsize);
	cv::Size winStride_slidewin = cv::Size(ncols_winstride, nrows_winstride);

	cv::Size padding_slidewin = cv::Size(0, 0);
	cv::Size cellSize_hog = cv::Size(8, 8);
	cv::Size blockSize_hog = cv::Size(16, 16);
	cv::Size blockStride_hog = cv::Size(8, 8);
	int nbins_hog = 9;
	
	const int derivAperture = 1;
	const double winSigma = -1;
	const cv::HOGDescriptor::HistogramNormType histogramNormType = cv::HOGDescriptor::L2Hys;
	const double L2HysThreshold = 0.2;
	const bool gammaCorrection = false;
	const int nlevels = cv::HOGDescriptor::DEFAULT_NLEVELS;
	const bool signedGradient = false;

	cv::HOGDescriptor hogObj(winSize_slidewin, blockSize_hog, blockSize_hog, cellSize_hog, nbins_hog, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient);

	cv::Mat img(cv::Size(ncols_img, nrows_img), CV_MAKETYPE(CV_8U, nchannels_img), img_row_major);

	int num_scales = fmin(floor(log(static_cast<double>(img.rows) / winSize_slidewin.height) / log(scaleratio)),
		floor(log(static_cast<double>(img.cols) / winSize_slidewin.width) / log(scaleratio))) + 1;
	num_scales = std::min(num_scales, max_numScales);

	// find a tight upper bound on total no. of sliding windows needed
	double stride_scale_width, stride_scale_height, nsw_rows, nsw_cols;
	size_t nslidewins_cur_scale, nslidewins_total = 0;
	size_t nslidewins_total_ub = 0;
	for (size_t s = 0; s < num_scales; s++)
	{
		stride_scale_width = winStride_slidewin.width * std::pow(scaleratio, s);
		stride_scale_height = winStride_slidewin.height * std::pow(scaleratio, s);
		nsw_rows = std::floor(nrows_img / stride_scale_height) - std::floor(winSize_slidewin.height / stride_scale_height) + 1;
		nsw_cols = std::floor(ncols_img / stride_scale_width) - std::floor(winSize_slidewin.width / stride_scale_width) + 1;
		// Without the increment below, I get exact computation of number of sliding
		// windows, but just in case (to upper bound it)
		++nsw_rows; ++nsw_cols;
		size_t nslidewins_est_cur_scale = nsw_rows * nsw_cols;
		nslidewins_total_ub += nslidewins_est_cur_scale;
	}

	int ndims_feat = hogObj.getDescriptorSize();

	Result_hogSlideWin* res = new Result_hogSlideWin();
	res->feats = (float*)malloc(nslidewins_total_ub * ndims_feat * sizeof(float));
	res->dr = (int*)malloc(nslidewins_total_ub * 4 * sizeof(int));
	res->ndims_feat = ndims_feat;

	float* ptr_feats = res->feats;
	int* ptr_dr = res->dr;
	
	size_t countUp_total_nwindows_allScales = 0;

	auto alignSize = [](size_t sz, int n)
	{
		CV_DbgAssert((n & (n - 1)) == 0); // n is a power of 2
		return (sz + n - 1) & -n;
	};

	auto gcd = [](int a, int b)
	{
		if (a < b)
			std::swap(a, b);
		while (b > 0)
		{
			int r = a % b;
			a = b;
			b = r;
		}
		return a;
	};

	auto getWindow = [&winSize_slidewin, &winStride_slidewin](const cv::Size& imageSize, int idx)
	{
		int nwindowsX = (imageSize.width - winSize_slidewin.width) / winStride_slidewin.width + 1;
		int y = idx / nwindowsX;
		int x = idx - nwindowsX * y;
		return cv::Rect(x*winStride_slidewin.width, y*winStride_slidewin.height, winSize_slidewin.width, winSize_slidewin.height);
	};

	auto numWindowsInImage = [&winSize_slidewin, &winStride_slidewin](const cv::Size& imageSize)
	{ 
		cv::Size numWindows_XY_directions((imageSize.width - winSize_slidewin.width) / winStride_slidewin.width + 1,(imageSize.height - winSize_slidewin.height) / winStride_slidewin.height + 1);
		return numWindows_XY_directions.area();
	};
	
	cv::Size cacheStride(gcd(winStride_slidewin.width, blockStride_hog.width), gcd(winStride_slidewin.height, blockStride_hog.height));
		
	for (size_t s = 0; s < num_scales; s++)
	{
		double scale = std::pow(scaleratio, s);
		cv::Mat img_resized;
		cv::resize(img, img_resized, cv::Size(), 1.0 / scale, 1.0 / scale, cv::INTER_LINEAR);
				
		cv::Size imgSize = img_resized.size();
		padding_slidewin.width = (int)alignSize(std::max(padding_slidewin.width, 0), cacheStride.width);
		padding_slidewin.height = (int)alignSize(std::max(padding_slidewin.height, 0), cacheStride.height);
		cv::Size paddedImgSize(imgSize.width + padding_slidewin.width * 2, imgSize.height + padding_slidewin.height * 2);
		size_t nwindows = numWindowsInImage(paddedImgSize);
		std::vector<cv::Point> locations(nwindows);
		for (size_t i = 0; i < nwindows; i++)
			locations[i] = getWindow(paddedImgSize, (int)i).tl() - cv::Point(padding_slidewin);

		std::vector<float> descriptors;
		hogObj.compute(img_resized, descriptors, winStride_slidewin, padding_slidewin, locations);

		std::copy(descriptors.begin(), descriptors.end(), ptr_feats + countUp_total_nwindows_allScales * ndims_feat);

		int idx_start_ptr_row_dr = countUp_total_nwindows_allScales * 4;

		for (size_t i = 0; i < nwindows; i++)
		{
			int* ptr_row_dr_cur = ptr_dr + idx_start_ptr_row_dr + i * 4;
			cv::Point location = locations[i];
			ptr_row_dr_cur[0] = std::round(location.x * scale);
			ptr_row_dr_cur[1] = std::round(location.y * scale);
			ptr_row_dr_cur[2] = std::round(winSize_slidewin.width * scale);
			ptr_row_dr_cur[3] = std::round(winSize_slidewin.height * scale);
		}
			
		countUp_total_nwindows_allScales += nwindows;

	} //end for: s

	res->nslidewins = countUp_total_nwindows_allScales;

	return res;
}

Result_hogSlideWin* extract_hog_multiscale_dollar(unsigned char* img_row_major, int nrows_img, int ncols_img, int nchannels_img, int nrows_winsize = 128, int ncols_winsize = 64, int nrows_winstride = 8, int ncols_winstride = 8, double scaleratio = std::pow(2, 1.0 / 8), int max_numScales = 1000)
{
	cv::Size winSize_slidewin = cv::Size(ncols_winsize, nrows_winsize);
	cv::Size winStride_slidewin = cv::Size(ncols_winstride, nrows_winstride);	
	cv::Mat img(cv::Size(ncols_img, nrows_img), CV_MAKETYPE(CV_8U, nchannels_img), img_row_major);

#define USE_FALZEN

#ifdef USE_FALZEN
	const int binSize = 8;
	const int nOrients = 9;
	const int softBin = -1;
	const int useHog = 2;
	const float clipHog = 0.2;
	const bool full_angle = true;
#else
	const int binSize = 8;
	const int nOrients = 9;
	const int softBin = 1;
	const int useHog = 1;
	const float clipHog = 0.2;
	const bool full_angle = false;
#endif
	
	int num_scales = fmin(floor(log(static_cast<double>(img.rows) / winSize_slidewin.height) / log(scaleratio)),
		floor(log(static_cast<double>(img.cols) / winSize_slidewin.width) / log(scaleratio))) + 1;
	num_scales = std::min(num_scales, max_numScales);

	// find a tight upper bound on total no. of sliding windows needed
	double stride_scale_width, stride_scale_height, nsw_rows, nsw_cols;
	size_t nslidewins_cur_scale, nslidewins_total = 0;
	size_t nslidewins_total_ub = 0;
	for (size_t s = 0; s < num_scales; s++)
	{
		stride_scale_width = winStride_slidewin.width * std::pow(scaleratio, s);
		stride_scale_height = winStride_slidewin.height * std::pow(scaleratio, s);
		nsw_rows = std::floor(nrows_img / stride_scale_height) - std::floor(winSize_slidewin.height / stride_scale_height) + 1;
		nsw_cols = std::floor(ncols_img / stride_scale_width) - std::floor(winSize_slidewin.width / stride_scale_width) + 1;
		// Without the increment below, I get exact computation of number of sliding
		// windows, but just in case (to upper bound it)
		++nsw_rows; ++nsw_cols;
		size_t nslidewins_est_cur_scale = nsw_rows * nsw_cols;
		nslidewins_total_ub += nslidewins_est_cur_scale;
	}

	auto compute_nch_H = [&useHog, &nOrients]()
	{
		if (useHog == 0)
			return nOrients;
		else
		{
			if (useHog == 1)
				return nOrients * 4;
			else
				return nOrients * 3 + 5;
		}
	};

	auto compute_H_from_M_O = [&](float* ptr_M, float* ptr_O, int nrows_img, int ncols_img)
	{
		int nr_H = nrows_img / binSize;
		int nc_H = ncols_img / binSize;
		int nch_H = compute_nch_H();
		float* ptr_H = (float*)malloc(nr_H * nc_H * nch_H * sizeof(float));

		// compute gradient histogram
		if (useHog == 0)
			gradHist(ptr_M, ptr_O, ptr_H, nrows_img, ncols_img, binSize, nOrients, softBin, full_angle);
		if (useHog == 1)
			hog(ptr_M, ptr_O, ptr_H, nrows_img, ncols_img, binSize, nOrients, softBin, full_angle, clipHog);
		else
			fhog(ptr_M, ptr_O, ptr_H, nrows_img, ncols_img, binSize, nOrients, softBin, clipHog);
		return ptr_H;
	};

	auto compute_H_from_image_data=[&](float* ptr_img_colMajor, int nrows_img, int ncols_img, int nchannels_img)
	{
		float* ptr_M = (float*)malloc(nrows_img * ncols_img * sizeof(float));		
		float* ptr_O = (float*)malloc(nrows_img * ncols_img * sizeof(float));
		// compute gradient magnitude and orientation
		gradMag(ptr_img_colMajor, ptr_M, ptr_O, nrows_img, ncols_img, nchannels_img, full_angle);
		float* ptr_H = compute_H_from_M_O(ptr_M, ptr_O, nrows_img, ncols_img);
		free(ptr_M);
		free(ptr_O);
		return ptr_H;
	};

	auto convert_rowMajorImageData_to_colMajor = [](float* ptr_imgDataRowMajor, int nrows_img, int ncols_img, int nchannels_img)
	{
		float* imgData_colMajor = (float*)malloc(nrows_img * ncols_img * nchannels_img * sizeof(float));
		size_t counter = 0;
		int nchannels_x_ncols_img = nchannels_img * ncols_img;
		for (size_t k = 0; k < nchannels_img; k++)
			for (size_t j = 0; j < ncols_img; j++)
				for (size_t i = 0; i < nrows_img; i++)
					imgData_colMajor[counter++] = ptr_imgDataRowMajor[i * nchannels_x_ncols_img + j * nchannels_img + k];
		return imgData_colMajor;
	};
		
	int nch_H = compute_nch_H();
	int ndims_feat = nch_H * (nrows_winsize / binSize) * (ncols_winsize / binSize);

	Result_hogSlideWin* res = new Result_hogSlideWin();
	res->feats = (float*)malloc(nslidewins_total_ub * ndims_feat * sizeof(float));
	res->dr = (int*)malloc(nslidewins_total_ub * 4 * sizeof(int));
	res->ndims_feat = ndims_feat;

	float* ptr_feats = res->feats;
	int* ptr_dr = res->dr;

	size_t countUp_nslidewins = 0;

	img.convertTo(img, CV_32F, 1 / 255.0);

	int ncols_winsize_divBy_binSize = ncols_winsize / binSize;
	int ncols_winstride_divBy_binSize = ncols_winstride / binSize;
	int nrows_winsize_divBy_binSize = nrows_winsize / binSize;
	int nrows_winstride_divBy_binSize = nrows_winstride / binSize;

	for (size_t s = 0; s < num_scales; s++)
	{
		double scale = std::pow(scaleratio, s);
		cv::Mat img_resized;
		cv::resize(img, img_resized, cv::Size(), 1.0 / scale, 1.0 / scale, cv::INTER_LINEAR);
		
		int nrows_img_resized = img_resized.rows;
		int ncols_img_resized = img_resized.cols;
		int nchannels_img_resized = img_resized.channels();
		float* img_resized_colMajor = convert_rowMajorImageData_to_colMajor(img_resized.ptr<float>(0), nrows_img_resized, ncols_img_resized, nchannels_img_resized);
		float* ptr_H = compute_H_from_image_data(img_resized_colMajor, nrows_img_resized, ncols_img_resized, nchannels_img_resized);
		int nrows_H_cur = nrows_img_resized / binSize;
		int ncols_H_cur = ncols_img_resized / binSize;
		int nrows_x_ncols_H_cur = nrows_H_cur * ncols_H_cur;
		
		for (size_t j = 0; j < ncols_H_cur - ncols_winsize_divBy_binSize; j+= ncols_winstride_divBy_binSize)
		{
			for (size_t i = 0; i < nrows_H_cur - nrows_winsize_divBy_binSize; i += nrows_winstride_divBy_binSize)
			{			
				int* ptr_dr_cur = ptr_dr + countUp_nslidewins * 4;
				ptr_dr_cur[0] = std::round(((j + 1)*binSize - binSize) * scale);
				ptr_dr_cur[1] = std::round(((i + 1)*binSize - binSize) * scale);
				ptr_dr_cur[2] = std::round(ncols_winsize * scale);
				ptr_dr_cur[3] = std::round(nrows_winsize * scale);
								
				float* ptr_feats_cur = ptr_feats + countUp_nslidewins * ndims_feat;
				size_t countUp_ndims_feat = 0;
				for (size_t kk = 0; kk < nch_H; kk++)
					for (size_t jj = 0; jj < ncols_H_cur; jj++)
						for (size_t ii = 0; ii < nrows_H_cur; ii++)
							ptr_feats_cur[countUp_ndims_feat++] = ptr_H[kk * nrows_x_ncols_H_cur + jj * nrows_H_cur + ii];						

				countUp_nslidewins++;
			}
		}

		free(ptr_H);

	} //end for: s

	res->nslidewins = countUp_nslidewins;

	return res;
}




#include <chrono>

int main()
{
	
	std::string fpath = "D:/Datasets/CUHK_Square/frames_train/Culture_Square_00151.png";
	//std::string fpath = "C:/Users/Kyaw/Desktop/photo/INRIA_pos_00001.png";

	cv::Mat img = cv::imread(fpath);

	//cv::Size winSize = cv::Size(64, 128);
	//cv::HOGDescriptor hobj(winSize, cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
	//std::vector<float> descriptors;
	//cv::Size winStride(8, 8);
	//cv::Size padding(0, 0);
	////std::vector<cv::Point> locations({cv::Point(32, 64)});
	////std::vector<cv::Point> locations({ cv::Point(0, 0) });
	//std::vector<cv::Point> locations;
	//hobj.compute(img, descriptors, winStride, padding, locations);
	//int dsize = hobj.getDescriptorSize();
	//int totalsize = descriptors.size();
	//int nslidewins = totalsize / dsize;
	//printf("dsize = %d, totalsize = %d\n", dsize, totalsize);
	//int nslidewins_X = ((img.cols - winSize.width) / winStride.width) + 1;
	//int nslidewins_Y = ((img.rows - winSize.height) / winStride.height) + 1;
	//int nslidewins_est = nslidewins_X * nslidewins_Y;
	//printf("nslidewins = %d, nslidewins_est = %d\n", nslidewins, nslidewins_est);
	////hobj.compute(img, descriptors1, winStride, padding, std::vector<cv::Point>({ cv::Point(0, 0) }));
	////hobj.compute(img, descriptors2, winStride, padding, std::vector<cv::Point>({ cv::Point(32, 64) }));
	////printf("descriptorSize = %d\n", hobj.getDescriptorSize());
	////std::cout << descriptors.size() << std::endl;
	////std::cout << locations << std::endl;
	////for (size_t i = 0; i < descriptors.size(); i++)
	////{
	////	printf("(%f, %f), ", descriptors1[i], descriptors2[i]);
	////}

	SlideWinHOGDescriptor swobj(cv::Size(64, 128), cv::Size(8, 8));

	//cv::Mat feats;
	//cv::Mat dr;
	//std::vector<float> descriptors;
	//std::vector<cv::Point> locations;
	//std::tie(descriptors, locations) = swobj.feat_extract(img); 
	//std::tie(feats, dr) = swobj.convert_to_Mat_tuple(descriptors, locations);

	//cv::Mat feats;
	//cv::Mat dr;	
	auto start = std::chrono::high_resolution_clock::now();
	//std::tie(feats, dr) = swobj.feat_extract_multiScale(img);
	//std::tie(feats, dr) = swobj.feat_extract_multiScale_ver2(img);
	//Result_hogSlideWin* res = extract_hog_multiscale_opencv(img.ptr<unsigned char>(0), img.rows, img.cols, img.channels());
	Result_hogSlideWin* res = extract_hog_multiscale_dollar(img.ptr<unsigned char>(0), img.rows, img.cols, img.channels());
	auto finish = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";

	cv::Mat feats(cv::Size(res->ndims_feat, res->nslidewins), CV_32FC1, res->feats);
	cv::Mat dr(cv::Size(4, res->nslidewins), CV_32SC1, res->dr);

	printf("nwindows = %d\n", feats.rows);	
		
	for (size_t i = 0; i < feats.rows; i++)
	{
		int* r = dr.ptr<int>(i);
		cv::Mat img_copy = img.clone();
		cv::rectangle(img_copy, cv::Rect(r[0], r[1], r[2], r[3]), cv::Scalar(255, 0, 0), 2);
		cv::imshow("win", img_copy);
		cv::waitKey(1);
	}
	
	cv::destroyAllWindows();

 
}

#pragma once

/**************************
File Name:CNN_PoolingLayer.hpp
Author: MiaoMiaoYoung

日志：
在构建好BP网络后
再一次，再一次，再一次，继续攻克卷积神经网络

2017年11月16日建立
主要用于卷积网络中的池化层，应用于池化操作
池化时分别为最大池化和平均池化，不同
建立了最大池化、平均池化的过程
建立了最大池化反向传播梯度项的过程，平均池化反向传播梯度项的过程

上述模块没有验证其正确性

更新了类的模块的说明

2017年11月17日
初步验证了最大池化、平均池化正向传播及反向传递的正确性
将类更新为类模板
****************************/

#include"Matrix.hpp"

namespace yDL
{

//#define CNN_TYPE double
//#define Check

	/************************************************

	功能：实现卷积神经网络的卷积层后的激活函数层

	参数：

	-   const int num                  // 输入、输出矩阵的深度（即输入输出图片的深度 - 必须保持一致）
	-   const int In_size              // 输入矩阵的大小（矩阵为方阵）
	-   const int Out_size             // 输出矩阵的大小（矩阵为方阵）
	-   const int Pool_size            // 池化核的大小
	-   const int Stride               // 进行池化时，行走的步长

	说明：

	-   Forward_Max;                   // 池化层前向传播，前向进行最大池化 (const std::vector<Matrix<CNN_TYPE>*>DataIn)
	-   Forward_Ave;                   // 池化层前向传播，前向进行平均池化 (const std::vector<Matrix<CNN_TYPE>*>DataIn)

	-   Backward_Max;                  // 池化层最大池化反向传递，反向传递误差 (const std::vector<Matrix<CNN_TYPE>*>DataIn,std::vector<Matrix<CNN_TYPE>*>GradeOut)
	-   Backward_Ave;                  // 池化层平均池化反向传递，反向传递误差 (std::vector<Matrix<CNN_TYPE>*>GradeOut)

	//API
	-   API_Data_Forward;     // 向前传递输出层的结果 -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // 向后传递当前层的误差 -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class CNN_TYPE>
	class CNN_Pooling_Layer
	{

	private:

		std::vector<Matrix<CNN_TYPE>*> DataOut;                  //输出层结果（当前层的输出）
		std::vector<Matrix<CNN_TYPE>*> GradeIn;                  //输入层梯度（当前层的梯度）

		//矩阵的个数 输入层与输出层都是严格相等的，不然就不对了
		int num;    //输入、输出层矩阵的个数（即深度）

		int Output_size;   //输出层矩阵的大小
		int Input_size;    //输入层矩阵的大小

		int Pooling_size;  //池化矩阵的大小
		int Stride;        //池化行走的步长

		CNN_Pooling_Layer(const CNN_Pooling_Layer&) = delete; //坚决不能有复制构造函数，出事呀

	public:
		explicit CNN_Pooling_Layer(
			const int NUM,
			const int In_size, const int Out_size,
			const int Pool_size, const int stride
		); //构造函数

		~CNN_Pooling_Layer();

		void Forward_Max(const std::vector<Matrix<CNN_TYPE>*>DataIn);        //卷积网络前向传播，最大池化
		void Forward_Ave(const std::vector<Matrix<CNN_TYPE>*>DataIn);        //卷积网络前向传播，平均池化
		void Backward_Max(const std::vector<Matrix<CNN_TYPE>*>DataIn,
			const std::vector<Matrix<CNN_TYPE>*>GradeOut);                   //卷积网络反向传递，最大池化
		void Backward_Ave(const std::vector<Matrix<CNN_TYPE>*>GradeOut);     //卷积网络反向传递，平均池化

		const std::vector<Matrix<CNN_TYPE>*> API_Data_Forward();         //向前传播结果 - API
		const std::vector<Matrix<CNN_TYPE>*> API_Grade_Backward();       //反向传递梯度 - API

#ifdef Check
		friend void test_PoolingLayer();
#endif

	};

	/************************************************

	函数名称：   CNN_Pooling_Layer
	功    能:    CNN_Pooling_Layer的构造函数
	参    数：
				 -  输入层、输出层图片的深度（个数）
				 -  输入层图片的大小（长X宽）-  输出层图片的大小（长X宽）
				 -  池化核的大小（长X宽）    -  池化核行走的步长
	返    回：
	说    明：
				 -  为输出DataOut、输入梯度GradeIn申请相应大小的空间
				 -  DataOut\GradeIn 只是分配空间，并无更多操作，具体需求在其他模块中实现
				 -  可以在程序运行过程中，减少申请内存的步骤，提高效率
	*************************************************/
	template<class CNN_TYPE>
	CNN_Pooling_Layer<CNN_TYPE>::CNN_Pooling_Layer(
		const int NUM,
		const int In_size, const int Out_size,
		const int Pool_size, const int stride
	) :
		num{ NUM },
		Input_size{ In_size }, Output_size{ Out_size },
		Pooling_size{ Pool_size }, Stride{ stride }
	{
		//为输出DataOut申请相应的空间
		for (int i = 0; i < NUM; i++) //外层的卷积核们
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(Out_size, Out_size);
			assert(temp != NULL);
			DataOut.push_back(temp);
		}

		//为输入GradeIn申请相应的空间
		for (int i = 0; i < NUM; i++) //外层的卷积核们
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(In_size, In_size);
			assert(temp != NULL);
			GradeIn.push_back(temp);
		}

	}

	/************************************************
	函数名称：~CNN_Pooling_Layer
	功    能:CNN_Pooling_Layer的析构函数
	参    数：
	返    回：
	说    明：为 DataOut \ GradeIn 释放相应空间
	*************************************************/
	template<class CNN_TYPE>
	CNN_Pooling_Layer<CNN_TYPE>::~CNN_Pooling_Layer()
	{
		//为输出DataOut释放相应的空间
		for (int i = 0; i < num; i++) //外层的卷积核们
			delete DataOut[i];

		//为输入GradeIn释放相应的空间
		for (int i = 0; i < num; i++) //外层的卷积核们
			delete GradeIn[i];
	}

	/************************************************
	函数名称：Forward_Max
	功    能:CNN_Pooling_Layer的最大池化操作，前向传播
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Pooling_Layer<CNN_TYPE>::Forward_Max(const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		for (unsigned int cnt = 0; cnt < DataIn.size(); cnt++)
			(*DataOut[cnt]) = Max_Pooling((*DataIn[cnt]), Pooling_size, Stride);
	}

	/************************************************
	函数名称：Forward_Ave
	功    能:CNN_Pooling_Layer的平均池化操作，前向传播
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Pooling_Layer<CNN_TYPE>::Forward_Ave(const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		for (unsigned int cnt = 0; cnt < DataIn.size(); cnt++)
			(*DataOut[cnt]) = Ave_Pooling((*DataIn[cnt]), Pooling_size, Stride);
	}

	/************************************************
	函数名称：Backward_Max
	功    能:CNN_Pooling_Layer的反向最大池化，后向传递误差
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Pooling_Layer<CNN_TYPE>::Backward_Max(const std::vector<Matrix<CNN_TYPE>*>DataIn, const std::vector<Matrix<CNN_TYPE>*>GradeOut)
	{
		for (unsigned int cnt = 0; cnt < DataIn.size(); cnt++)
		{
			std::vector<std::vector<bool>>sign = Max_Pooling_Sign((*DataIn[cnt]), Pooling_size, Stride);

			(*GradeIn[cnt]).Initialize();

			int counter = 0;
			for (int i = 0; i < Input_size; i++)
				for (int j = 0; j < Input_size; j++)
					if (sign[i][j]) //最大池化进行反向传递误差
						(*GradeIn[cnt])[i][j] = (*GradeOut[cnt]).Get_Value(counter / Output_size, counter%Output_size);
		}
	}

	/************************************************
	函数名称：Backward_Max
	功    能:CNN_Pooling_Layer的反向最大池化，后向传递误差
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Pooling_Layer<CNN_TYPE>::Backward_Ave(const std::vector<Matrix<CNN_TYPE>*>GradeOut)
	{
		Matrix<CNN_TYPE> kron(Pooling_size, Pooling_size);
		for (int i = 0; i < Pooling_size; i++)
			for (int j = 0; j < Pooling_size; j++)
				kron[i][j] = 1;

		double sq = (double)Pooling_size*(double)Pooling_size;
		for (int cnt = 0; cnt < num; cnt++)
		{
			(*GradeIn[cnt]) = Kronecker((*GradeOut[cnt]), kron);
			(*GradeIn[cnt]) *= 1.0 / sq;
		}
	}

	/************************************************
	函数名称:API_Data_Forward
	功    能:CNN_Pooling_Layer 向前（下一层）传递当前层的结果
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_Pooling_Layer<CNN_TYPE>::API_Data_Forward()
	{
		return DataOut;
	}

	/************************************************
	函数名称:API_Grade_Backward
	功    能:CNN_Pooling_Layer 反向（前一层）传递当前层的梯度项
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_Pooling_Layer<CNN_TYPE>::API_Grade_Backward()
	{
		return GradeIn;
	}

#ifdef Check

	/************************************************
	函数名称:
	功    能:CNN_Convolution_Layer 反向（前一层）传递当前层的梯度项
	参    数：
	返    回：
	说    明：
	*************************************************/
	void test_PoolingLayer()
	{
		CNN_Pooling_Layer<double> test(1, 6, 3, 2, 2);

		srand((unsigned int)time(NULL) + rand());

		Matrix<double>test_data(6, 6);
		double test_data_[36];
		for (int cnt = 0; cnt < 36; cnt++)
			test_data_[cnt] = rand() % 36;
		test_data.assigment(test_data_, sizeof(test_data_));

		cout << "test_data" << std::endl << std::endl;
		test_data.show();
		

		std::vector<Matrix<double>*> test_dm;
		test_dm.push_back(&test_data);

		test.Forward_Max(test_dm);

		cout << std::endl << std::endl << "*******************" << std::endl << std::endl;
		cout << "MAX_Forward" << std::endl << std::endl;

		for (unsigned int i = 0; i < test.API_Data_Forward().size(); i++)
			(*(test.API_Data_Forward())[i]).show();

		test.Forward_Ave(test_dm);

		cout << std::endl << std::endl << "*******************" << std::endl << std::endl;
		cout << "AVE_Forward" << std::endl << std::endl;

		for (unsigned int i = 0; i < test.API_Data_Forward().size(); i++)
			(*(test.API_Data_Forward())[i]).show();

		Matrix<double>test_grade(3, 3);
		double test_grade_[9] = { 1,2,3,4,5,6,7,8,9 };
		test_grade.assigment(test_grade_, sizeof(test_grade_));

		std::vector<Matrix<double>*> test_gm;
		test_gm.push_back(&test_grade);

		test.Backward_Max(test_dm, test_gm);

		cout << std::endl << std::endl << "*******************" << std::endl << std::endl;
		cout << "MAX_BACKward" << std::endl << std::endl;

		for (unsigned int i = 0; i < test.API_Data_Forward().size(); i++)
			(*(test.API_Grade_Backward())[i]).show();

		test.Backward_Ave(test_gm);

		cout << std::endl << std::endl << "*******************" << std::endl << std::endl;
		cout << "AVE_BACKward" << std::endl << std::endl;

		for (unsigned int i = 0; i < test.API_Data_Forward().size(); i++)
			(*(test.API_Grade_Backward())[i]).show();
	}

#endif // Check

#undef Check

}

#undef Check

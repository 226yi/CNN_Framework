#pragma once

/**************************
File Name:CNN_ConvolutionLayer.hpp
Author: MiaoMiaoYoung

日志：
在构建好BP网络后
再一次，再一次，再一次，继续攻克卷积神经网络

2017年11月15日建立
主要用于卷积网络卷积层的实现
在编写代码过程中，使用宏定义的方式，目的为了最后取消宏定义而更改为类模板

完成了类的构造函数、析构函数的建立，为类分配空间，
这两个函数没有检查是否正确，没有检查是否存在内存泄漏的问题
初步设计了类的内存分配，考虑如何进行分布式计算

2017年11月16日更新
设计了卷积网络前向传播的过程
设计了卷积网络反向传递的过程
设计了卷积网络卷积核更新参数的过程
并验证以上功能是正确的（初步验证）

//缺少阈值！

2017年11月17日更新
补增了卷积网络过程中的阈值
包括正向过程及反向过程
没有验证阈值在传递过程中的正确性

2017年11月25日更新
进行了Check.cpp检验
阈值理解错误，正在更新

2017年11月28日更新
增加了写入文件功能
用于保存当前训练值，测试通过

2017年12月5日更新
更改了卷积网络卷积层正向传播及更新参数时阈值的作用
之前对阈值的理解有偏差

2017年12月10日更新
void CNN_Convolution_Layer<CNN_TYPE>::Backward(std::vector<Matrix<CNN_TYPE>*>GradeOut)中
对前向梯度进行计算的过程中
没有重新初始化就开始不停的使用累加 (*GradeIn[index]) += Convolution(&data, &kernel_back, Stride);
导致每一次图片训练都会累加了上一次的梯度，导致错误啊！！！
重大疏漏！！！因为类在初始化的时候是进行了所有模块的初始化的，所以就导致这里出现这么大的错误
而且说实话，这种错误太难找了！！！

使用前初始化！！！使用前初始化！！！这些东西框架教不来

检查了一下类似的错误

****************************/

//#define Check

#ifdef Check
extern void test_CNN_ConvolutionLayer(); //Check.cpp 实例验证是否正确
#endif // Check


#include"Matrix.hpp"
#include<string>

namespace yDL
{

	//#define CNN_TYPE double

	/************************************************

	功能：实现卷积神经网络的卷积层

	参数：

	-   const int In_num               // 输入矩阵的深度（即输入图片的深度）
	-   const int Out_num              // 输出矩阵的深度（即输出图片的深度）
	-   const int In_size              // 输入矩阵的大小（矩阵为方阵）
	-   const int Out_size             // 输出矩阵的大小（矩阵为方阵）
	-   const int Kernel_Matrix_size   // 卷积核的大小（为方阵，卷积核的深度与输入矩阵的深度相同，卷积核的个数与输出矩阵的个数相同）
	-   const int Stride               // 卷积核进行卷积时，行走的步长
	-   const T trans_func(const T x)  // 卷积核的初始化函数，默认为NULL，建议外部实现：随机数赋值或以静态变量记忆(用于测试)进行赋值

	说明：

	-   Forward;              // 卷积层前向传播，前向进行卷积 (const std::vector<Matrix<CNN_TYPE>*>DataIn)
	-   Backward;             // 卷积层反向传递，反向传递误差 (std::vector<Matrix<CNN_TYPE>*>GradeOut)
	-   Updata;               // 卷积层卷积核进行参数更新(const std::vector<Matrix<CNN_TYPE>*>DataIn,const std::vector<Matrix<CNN_TYPE>*>GradeOut)

	//API
	-   API_Data_Forward;     // 向前传递输出层的结果 -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // 向后传递当前层的误差 -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class CNN_TYPE>
	class CNN_Convolution_Layer
	{

	protected:
	public:
		std::vector<std::vector<Matrix<CNN_TYPE>*>> Kernel;      //当前层的卷积核们
		std::vector<CNN_TYPE>Kernel_Bias;                        //当前卷积层的阈值

		std::vector<Matrix<CNN_TYPE>*> DataOut;                  //输出层结果（当前层的输出）
		std::vector<Matrix<CNN_TYPE>*> GradeIn;                  //输入层梯度（当前层的梯度）

		int Input_num;    //输入矩阵的个数（即深度）
		int Output_num;   //输出矩阵的个数（即深度）

		int Output_size;   //输出层矩阵的大小
		int Input_size;    //输入层矩阵的大小

		int Kernel_size;   //卷积核矩阵的大小
		int Stride;        //卷积核行走的步长

		CNN_Convolution_Layer(const CNN_Convolution_Layer&) = delete;       //这种类坚决不要有复制构造函数，我拒绝


	public:
		explicit CNN_Convolution_Layer(
			const int In_num, const int Out_num,
			const int In_size, const int Out_size,
			const int Kernel_Matrix_size, const int stride,
			const CNN_TYPE trans_func(const CNN_TYPE x) = NULL
		); //构造函数

		explicit CNN_Convolution_Layer(std::ifstream& fin); //构造函数 - 从文件中读取数据


		~CNN_Convolution_Layer(); //析构函数

		void Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn);        //卷积网络前向传播
		void Backward(const std::vector<Matrix<CNN_TYPE>*>GradeOut);     //卷积网络反向传递
		void Updata(const double LS,
			const std::vector<Matrix<CNN_TYPE>*>DataIn,
			const std::vector<Matrix<CNN_TYPE>*>GradeOut);               //卷积网络卷积核进行权值更新

		const std::vector<Matrix<CNN_TYPE>*> API_Data_Forward();         //向前传播结果 - API
		const std::vector<Matrix<CNN_TYPE>*> API_Grade_Backward();       //反向传递梯度 - API

		void Save_Info(std::ofstream& fout); //将当前数据写入文件中

#ifdef Check
		friend void test_convolution();
		friend void ::test_CNN_ConvolutionLayer();
#endif

	};


	/************************************************

	函数名称：   CNN_Convolution_Layer
	功    能:    CNN_Convolution_Layer的构造函数
	参    数：
	-  输入层图片的深度（个数） -  输出层图片的深度（个数）
	-  输入层图片的大小（长X宽）-  输出层图片的大小（长X宽）
	-  卷积核的大小（长X宽）    -  卷积核行走的步长
	-  卷积核初始化函数（若没有则为全部初始化为0）
	返    回：
	说    明：
	-  为卷积核kernel、输出DataOut、输入梯度GradeIn申请相应大小的空间
	-  trans_func用于初始化Kernel，具体实现建议随机数，或是使用静态变量进行逐个操作
	-  DataOut 只是分配空间，并无更多操作，具体需求在其他模块中实现
	-  GradeIn 不仅分配了空间，为了配合反向传递的操作，在分配空间的同时，全部初始化为0
	-  可以在程序运行过程中，减少申请内存的步骤，提高效率
	*************************************************/
	template<class CNN_TYPE>
	CNN_Convolution_Layer<CNN_TYPE>::CNN_Convolution_Layer(
		const int In_num, const int Out_num,
		const int In_size, const int Out_size,
		const int Kernel_Matrix_size, const int stride,
		const CNN_TYPE trans_func(const CNN_TYPE x)
	) :
		Input_num{ In_num }, Output_num{ Out_num },
		Input_size{ In_size }, Output_size{ Out_size },
		Kernel_size{ Kernel_Matrix_size }, Stride{ stride }
	{

		//为卷积层申请相应的空间
		for (int i = 0; i < Out_num; i++) //外层的卷积核们
		{
			std::vector<Matrix<CNN_TYPE>*>temp_kernels;
			for (int j = 0; j < In_num; j++) //内层的卷积核们
			{
				Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(Kernel_Matrix_size, Kernel_Matrix_size);
				assert(temp != NULL);

				if (trans_func != NULL)
					(*temp) = (*temp).transfer(trans_func);
				else
					(*temp).Initialize();//初始化

				temp_kernels.push_back(temp);
			}

			Kernel.push_back(temp_kernels); //拿上指针
		}

		//为卷积层阈值申请相应空间
		for (int i = 0; i < Out_num; i++)
		{
			CNN_TYPE bias = 0; //阈值初始化为0
			Kernel_Bias.push_back(bias);
		}

		//为输出DataOut申请相应的空间
		for (int i = 0; i < Out_num; i++) //外层的卷积核们
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(Out_size, Out_size);
			assert(temp != NULL);
			DataOut.push_back(temp);
		}

		//为输入GradeIn申请相应的空间
		for (int i = 0; i < In_num; i++) //外层的卷积核们
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(In_size, In_size);
			assert(temp != NULL);
			(*temp).Initialize();
			GradeIn.push_back(temp);
		}

	}

	/************************************************
	函数名称：   CNN_Convolution_Layer
	功    能:    CNN_Convolution_Layer的构造函数 - 从文件中读取当前的数据
	参    数：   文件路径
	-  输入层图片的深度（个数） -  输出层图片的深度（个数）
	-  输入层图片的大小（长X宽）-  输出层图片的大小（长X宽）
	-  卷积核的大小（长X宽）    -  卷积核行走的步长
	-  卷积核初始化函数（若没有则为全部初始化为0）
	返    回：
	说    明：
	-  为卷积核kernel、输出DataOut、输入梯度GradeIn申请相应大小的空间
	-  trans_func用于初始化Kernel，具体实现建议随机数，或是使用静态变量进行逐个操作
	-  DataOut 只是分配空间，并无更多操作，具体需求在其他模块中实现
	-  GradeIn 不仅分配了空间，为了配合反向传递的操作，在分配空间的同时，全部初始化为0
	-  可以在程序运行过程中，减少申请内存的步骤，提高效率
	*************************************************/
	template<class CNN_TYPE>
	CNN_Convolution_Layer<CNN_TYPE>::CNN_Convolution_Layer(std::ifstream& fin)
	{

		//进行层校对
		if (1)
		{
			std::string Proofread_Layer = "";
			fin >> Proofread_Layer;
			assert(Proofread_Layer == "CNN_Convolution_Layer"); //校对 - 是否导入当前层的数据 - 有错则进行报警
		}

		int In_num = 0;
		int Out_num = 0;
		int In_size = 0;
		int Out_size = 0;
		int Kernel_Matrix_size = 0;
		int stride = 0;


		//读入该层基本信息
		if (1)
		{
			std::string Check_inner = ""; //检查位

			fin >> Check_inner;
			assert(Check_inner == "Input_num");
			fin >> In_num;

			fin >> Check_inner;
			assert(Check_inner == "Output_num");
			fin >> Out_num;

			fin >> Check_inner;
			assert(Check_inner == "Input_size");
			fin >> In_size;

			fin >> Check_inner;
			assert(Check_inner == "Output_size");
			fin >> Out_size;

			fin >> Check_inner;
			assert(Check_inner == "Kernel_size");
			fin >> Kernel_Matrix_size;

			fin >> Check_inner;
			assert(Check_inner == "Stride_size");
			fin >> stride;

			//Input_num{ In_num }, Output_num{ Out_num },
			//Input_size{ In_size }, Output_size{ Out_size },
			//Kernel_size{ Kernel_Matrix_size }, Stride{ stride }

			Input_num = In_num;
			Output_num = Out_num;

			Input_size = In_size;
			Output_size = Out_size;

			Kernel_size = Kernel_Matrix_size;
			Stride = stride;

		}


		std::string Check = ""; //检查位
		fin >> Check;
		assert(Check == "CNN_Kernel_Weight");

		//为卷积层申请相应的空间 - 并读取数据
		for (int i = 0; i < Out_num; i++) //外层的卷积核们
		{
			std::vector<Matrix<CNN_TYPE>*>temp_kernels;
			for (int j = 0; j < In_num; j++) //内层的卷积核们
			{
				Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(Kernel_Matrix_size, Kernel_Matrix_size);
				assert(temp != NULL);

				fin >> (*temp);

				if (trans_func != NULL)
					(*temp) = (*temp).transfer(trans_func);
				else
					(*temp).Initialize();//初始化

				temp_kernels.push_back(temp);
			}

			Kernel.push_back(temp_kernels); //拿上指针
		}

		fin >> Check;
		assert(Check == "CNN_Kernel_Bias");

		//为卷积层阈值申请相应空间
		for (int i = 0; i < Out_num; i++)
		{
			CNN_TYPE bias = 0; //阈值初始化为0
			fin >> bias;
			Kernel_Bias.push_back(bias);
		}

		//为输出DataOut申请相应的空间
		for (int i = 0; i < Out_num; i++) //外层的卷积核们
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(Out_size, Out_size);
			assert(temp != NULL);
			DataOut.push_back(temp);
		}

		//为输入GradeIn申请相应的空间
		for (int i = 0; i < In_num; i++) //外层的卷积核们
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(In_size, In_size);
			assert(temp != NULL);
			(*temp).Initialize();
			GradeIn.push_back(temp);
		}

	}


	/************************************************
	函数名称：~CNN_Convolution_Layer
	功    能:CNN_Convolution_Layer的析构函数
	参    数：
	返    回：
	说    明：为卷积核kernel \ DataOut \ GradeIn 释放相应空间
	*************************************************/
	template<class CNN_TYPE>
	CNN_Convolution_Layer<CNN_TYPE>::~CNN_Convolution_Layer()
	{
		//为卷积层释放相应的空间
		for (int i = 0; i < Output_num; i++) //外层的卷积核们
			for (int j = 0; j < Input_num; j++) //内层的卷积核们
				delete Kernel[i][j];

		//为输出DataOut释放相应的空间
		for (int i = 0; i < Output_num; i++) //外层的卷积核们
			delete DataOut[i];

		//为输入GradeIn释放相应的空间
		for (int i = 0; i < Input_num; i++) //外层的卷积核们
			delete GradeIn[i];
	}

	/************************************************
	函数名称：Forward
	功    能:CNN_Convolution_Layer的卷积操作，前向传播
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Convolution_Layer<CNN_TYPE>::Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		for (int cnt = 0; cnt < Output_num; cnt++)
			(*DataOut[cnt]) = Convolution(DataIn, Kernel[cnt], Stride) + (Kernel_Bias[cnt]);  //矩阵的卷积操作
	}

	/************************************************
	函数名称：Backward
	功    能:CNN_Convolution_Layer的反向卷积操作，后向传递
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Convolution_Layer<CNN_TYPE>::Backward(std::vector<Matrix<CNN_TYPE>*>GradeOut)
	{
		//对GradeIn的重新初始化（擦除上一次的梯度的痕迹）
		for (int cnt = 0; cnt < Output_num; cnt++) //对每一张特征图的梯度进行遍历
			for (int index = 0; index < Input_num; index++)
				(*GradeIn[index]).Initialize();

		for (int cnt = 0; cnt < Output_num; cnt++) //对每一张特征图的梯度进行遍历
		{
			//关于指针 - 这里的指针并没有申请空间，其Pading Rot生成的矩阵，会在循环结束后，自动经过析构函数析构，不可以delete

			const int Pad_Size = ((Input_size - 1)*Stride + Kernel_size - Output_size) / 2;
			Matrix<CNN_TYPE>data = (*GradeOut[cnt]).Pading(Pad_Size);

			for (int index = 0; index < Input_num; index++)
			{
				Matrix<CNN_TYPE> kernel_back = (*Kernel[cnt][index])('R');

				(*GradeIn[index]) += Convolution(&data, &kernel_back, Stride);

			}// for 对每一层进行遍历，进行回溯

		}// for 对每一张特征图的梯度进行遍历
	}

	/************************************************
	函数名称:Updata
	功    能:CNN_Convolution_Layer 当前层的卷积核权重更新
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Convolution_Layer<CNN_TYPE>::Updata(const double LS,
		const std::vector<Matrix<CNN_TYPE>*>DataIn,
		const std::vector<Matrix<CNN_TYPE>*>GradeOut)
	{
		for (int cnt = 0; cnt < Output_num; cnt++)//for 对每一个生成的特征图，即每一个卷积核进行遍历
		{
			//卷积核权值更新

			Matrix<CNN_TYPE>* temp_kernel = GradeOut[cnt];  //牵连输出矩阵的当前层

			for (int index = 0; index < Input_num; index++)  // for 对卷积核中的每一层进行遍历
			{
				Matrix<CNN_TYPE> Error(Kernel_size, Kernel_size); //误差矩阵

				Matrix<CNN_TYPE>* temp_data = DataIn[index];   //牵连输入矩阵的当前层

				Error = Convolution(temp_data, temp_kernel, Stride);  //得到当前卷积核的当前层的误差

																	  //cout << endl << endl;
																	  //cout << "error" << endl;
																	  //Error.show();
																	  //cout << endl << endl;
																	  //(*Kernel[cnt][index]).show();

				//cout << (*Kernel[cnt][index]).Count_Ave() << " " << (LS*Error).Count_Ave()<<endl;

				(*Kernel[cnt][index]) -= LS*Error;   //卷积核进行更新

			} // for 对一个卷积核中的每一层进行遍历，对卷积核的权重进行更新


			  //卷积核阈值更新

			Kernel_Bias[cnt] -= LS*(*GradeOut[cnt]).Get_Determinant_Value();

		} //for 对每一个生成的特征图，即每一个卷积核进行遍历

	}

	/************************************************
	函数名称:API_Data_Forward
	功    能:CNN_Convolution_Layer 向前（下一层）传递当前层的结果
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_Convolution_Layer<CNN_TYPE>::API_Data_Forward()
	{
		return DataOut;
	}

	/************************************************
	函数名称:API_Grade_Backward
	功    能:CNN_Convolution_Layer 反向（前一层）传递当前层的梯度项
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_Convolution_Layer<CNN_TYPE>::API_Grade_Backward()
	{
		return GradeIn;
	}

	/************************************************
	函数名称:Save_Info
	功    能:将训练成果储存在文件中，日后可以直接读取
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Convolution_Layer<CNN_TYPE>::Save_Info(std::ofstream&fout)
	{

		fout << "CNN_Convolution_Layer" << std::endl;
		fout << "Input_num " << Input_num << std::endl;
		fout << "Output_num " << Output_num << std::endl;
		fout << "Input_size " << Input_size << std::endl;
		fout << "Output_size " << Output_size << std::endl;
		fout << "Kernel_size " << Kernel_size << std::endl;
		fout << "Stride " << Stride << std::endl;
		fout << std::endl;


		fout << "CNN_Kernel_Weight" << std::endl;
		for (unsigned int cnt = 0; cnt < Kernel.size(); cnt++)
		{
			for (unsigned int index = 0; index < (Kernel[cnt]).size(); index++)
				fout << (*Kernel[cnt][index]) << std::endl;
			fout << std::endl;
		}

		fout << "CNN_Kernel_Bias" << std::endl;
		for (unsigned int cnt = 0; cnt < Kernel_Bias.size(); cnt++)
			fout << (Kernel_Bias[cnt]) << std::endl;

		fout << std::endl << std::endl;

	}

#undef Check
#ifdef Check

	using std::vector;

	const double trans_fun(const double x)
	{
		static double _array[24] = { 1,1,2,2,1,1,1,1,0,1,1,0,1,0,0,1,2,1,2,1,1,2,2,0 };
		static int i = -1;
		i++;
		return _array[i];
	}

	/************************************************
	函数名称:test_Convolution
	功    能:测试卷积网络的卷积层的正向传播，反向传递，卷积核权重更新
	参    数：
	返    回：
	说    明：
	*************************************************/
	void test_convolution()
	{
		CNN_Convolution_Layer<double> test(3, 2, 3, 2, 2, 1, trans_fun);

		cout << "Kernel:" << std::endl;
		vector<vector<Matrix<double>*>>k = test.Kernel;
		for (unsigned int i = 0; i < k.size(); i++)
			for (unsigned int j = 0; j < k[i].size(); j++)
				(*k[i][j]).show();

		cout << std::endl << "**********" << std::endl << std::endl;


		vector<Matrix<double>*>data;
		if (1)
		{
			Matrix<double>* a1 = new(std::nothrow)Matrix<double>(3, 3);
			assert(a1);
			double b1[9] = { 1,2,0,1,1,3,0,2,2 };
			(*a1).assigment(b1, sizeof(b1));

			Matrix<double>* a2 = new(std::nothrow)Matrix<double>(3, 3);
			assert(a2);
			double b2[9] = { 0,2,1,0,3,2,1,1,0 };
			(*a2).assigment(b2, sizeof(b2));

			Matrix<double>* a3 = new(std::nothrow)Matrix<double>(3, 3);
			assert(a3);
			double b3[9] = { 1,2,1,0,1,3,3,3,2 };
			(*a3).assigment(b3, sizeof(b3));

			data.push_back(a1);
			data.push_back(a2);
			data.push_back(a3);
		}
		test.Forward(data);

		vector<Matrix<double>*>grade;
		if (1)
		{
			Matrix<double>* a1 = new(std::nothrow)Matrix<double>(2, 2);
			assert(a1);
			double b1[4] = { -1,-2,-3,-4 };
			(*a1).assigment(b1, sizeof(b1));

			Matrix<double>* a2 = new(std::nothrow)Matrix<double>(2, 2);
			assert(a2);
			double b2[4] = { -5,-6,-7,-8 };
			(*a2).assigment(b2, sizeof(b2));

			grade.push_back(a1);
			grade.push_back(a2);
		}
		test.Backward(grade);

		const vector<Matrix<double>*>result_data = test.API_Data_Forward(); //测试用

		cout << "DataOut:" << std::endl;
		for (unsigned int cnt = 0; cnt < result_data.size(); cnt++)
			(*result_data[cnt]).show();

		cout << std::endl << "**********" << std::endl << std::endl;

		const vector<Matrix<double>*>result_grade = test.API_Grade_Backward(); //测试用
		cout << "GradeIn:" << std::endl;
		for (unsigned int cnt = 0; cnt < result_grade.size(); cnt++)
			(*result_grade[cnt]).show();

		test.Updata(1.0, data, grade);

		cout << "Kernel:" << std::endl;
		vector<vector<Matrix<double>*>>k1 = test.Kernel;
		for (unsigned int i = 0; i < k1.size(); i++)
			for (unsigned int j = 0; j < k1[i].size(); j++)
				(*k1[i][j]).show();

		cout << std::endl << "**********" << std::endl << std::endl;


		test.Forward(data);
		const vector<Matrix<double>*>_result_data = test.API_Data_Forward();  //测试用

		cout << std::endl << "**********" << std::endl << std::endl;

		cout << "DataOut_Again:" << std::endl;
		for (unsigned int cnt = 0; cnt < _result_data.size(); cnt++)
			(*_result_data[cnt]).show();

		for (unsigned int i = 0; i < data.size(); i++)
			delete data[i];

		for (unsigned int i = 0; i < grade.size(); i++)
			delete grade[i];

	}


#endif

#undef Check

}

#undef Check
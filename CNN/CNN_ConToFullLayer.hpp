#pragma once

/**************************
File Name:CNN_ConToFullLayer.hpp
Author: MiaoMiaoYoung

日志：
在构建好BP网络后
再一次，再一次，再一次，继续攻克卷积神经网络

2017年11月16日建立
主要用于卷积网络中特征提取层（卷积）与特征映射层（简单神经网路）之间的连接层
这一层为全连接层，反向传播遵循全连接反向传播规则
尚未开工

2017年11月17日更新
构建了特征提取层向特征映射层的过渡
主要为全连接的构架

2017年11月25日更新
通过Check.cpp进行进一步检查
更新了Updata的类模板使用
更新了初始化缺省的使用
更新了更新权重的操作

2017年11月28日更新
增加了文件写出功能，将网络训练结果得以保存
测试通过
****************************/

#include"Matrix.hpp"

//#define Check

#ifdef Check
extern void test_CNN_ConToFull_Layer(); //Check.cpp 实例验证是否正确
#endif // Check


namespace yDL
{


	/************************************************

	功能：实现卷积神经网络的特征提取层向特征映射层的过渡,全连接层

	参数：

	-   const int Input_num            // 输入矩阵的深度
	-   const int In_size              // 输入矩阵的大小（矩阵为方阵）
	-   const int Output_num           // 输出特征值的个数 （即元素个数，以列向量的形式输出）

	说明：

	-   Forward;            // 过渡层前向传播，线性变化                   (const std::vector<Matrix<CNN_TYPE>*>DataIn)
	-   Backward;           // 过渡层反向传递误差，遵守全连接线性变化规则 (std::vector<Matrix<CNN_TYPE>*>GradeOut)
	-   Updata;             // 过渡层更新当前层的权重及阈值

	//API
	-   API_Data_Forward;     // 向前传递输出层的结果 -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // 向后传递当前层的误差 -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class CNN_TYPE>
	class CNN_ConToFull_Layer
	{
	private:
		Matrix<CNN_TYPE>* Weight;                                //当前层的权重
		Matrix<CNN_TYPE>* Bias;                                  //当前层的阈值

		Matrix<CNN_TYPE>* DataOut;                               //输出层结果（当前层的输出）
		std::vector<Matrix<CNN_TYPE>*> GradeIn;                  //输入层梯度（当前层的梯度）

		int Input_num;    //输入矩阵的个数（即深度）
		int Output_num;   //输出矩阵的个数（即深度）

		int Input_size;    //输入层矩阵的大小

		CNN_ConToFull_Layer(const CNN_ConToFull_Layer&) = delete;

	public:

		explicit CNN_ConToFull_Layer(
			const int In_num, const int Out_num, const int In_size,
			const CNN_TYPE trans_func(const CNN_TYPE x) = NULL
		); //构造函数

		~CNN_ConToFull_Layer(); //析构函数

		void Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn);                     //卷积网络前向传播
		void Backward(const Matrix<CNN_TYPE>* const GradeOut);                        //卷积网络反向传递
		void Updata(const double LS, const Matrix<CNN_TYPE>* const GradeOut,
			const std::vector<Matrix<CNN_TYPE>*>DataIn);                              //卷积网络卷积核进行权值更新

		const Matrix<CNN_TYPE>* const API_Data_Forward();                      //向前传播结果 - API
		const std::vector<Matrix<CNN_TYPE>*> API_Grade_Backward();             //反向传递梯度 - API

		void Save_Info(std::ofstream& fout); //将当前数据写入文件中

#ifdef Check
		friend 	void test_CNN_ConToFull();
		friend  void ::test_CNN_ConToFull_Layer();
#endif // Check


	};

	/************************************************

	函数名称：   CNN_ConToFull_Layer
	功    能:    CNN_ConToFull_Layer的构造函数
	参    数：
	             -  输入层图片的深度（个数） -  输出层图片的深度（个数） - 输入层图片的大小（长X宽）
	             -  卷积核初始化函数（若没有则为全部初始化为0）
	返    回：
	说    明：
         	     -  为权重Weight、阈值Bias、列矩阵输入项DataIn_Column、输出DataOut、输入梯度GradeIn申请相应大小的空间
	             -  trans_func用于初始化Kernel，具体实现建议随机数，或是使用静态变量进行逐个操作
	             -  DataOut 只是分配空间，并无更多操作，具体需求在其他模块中实现
	             -  GradeIn 不仅分配了空间，为了配合反向传递的操作，在分配空间的同时，全部初始化为0
	             -  可以在程序运行过程中，减少申请内存的步骤，提高效率
	*************************************************/
	template<class CNN_TYPE>
	CNN_ConToFull_Layer<CNN_TYPE>::CNN_ConToFull_Layer(
		const int In_num, const int Out_num, const int In_size,
		const CNN_TYPE trans_func(const CNN_TYPE x)
	) :
		Input_num{ In_num }, Output_num{ Out_num },Input_size{ In_size }
	{
		//为权重申请相应的空间
		if (1)
		{
			int size = In_num*In_size*In_size;
			Weight = new(std::nothrow)Matrix<CNN_TYPE>(Out_num, size);
			assert(Weight != NULL);
			if (trans_func != NULL)
				(*Weight) = (*Weight).transfer(trans_func);
			else
				(*Weight).Initialize();
		}

		//为阈值申请相应空间
		if (1)
		{
			Bias = new(std::nothrow)Matrix<CNN_TYPE>(Out_num,1);
			assert(Bias != NULL);
			(*Bias).Initialize();
		}

		//为输出DataOut申请相应的空间
		if (1)
		{
			DataOut = new(std::nothrow)Matrix<CNN_TYPE>(Out_num, 1); //输出为列矩阵
			assert(DataOut != NULL);
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
	函数名称：~CNN_ConToFull_Layer
	功    能:CNN_ConToFull_Layer的析构函数
	参    数：
	返    回：
	说    明：为卷积核kernel \ DataOut \ GradeIn 释放相应空间
	*************************************************/
	template<class CNN_TYPE>
	CNN_ConToFull_Layer<CNN_TYPE>::~CNN_ConToFull_Layer()
	{
		//为权值释放相应的空间
		delete Weight;

		//为阈值释放相应的空间
		delete Bias;

		//为输出DataOut释放相应的空间
		delete DataOut;

		//为输入GradeIn释放相应的空间
		for (int i = 0; i < Input_num; i++) //外层的卷积核们
			delete GradeIn[i];
	}

	/************************************************
	函数名称：Forward
	功    能:CNN_ConToFull_Layer的卷积操作，前向传播
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_ConToFull_Layer<CNN_TYPE>::Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		Matrix<CNN_TYPE> DataIn_Column(Input_num*Input_size*Input_size, 1);  //输入项转化为列矩阵
		DataIn_Column = Column_Joint(DataIn);
		(*DataOut) = (*Weight)*DataIn_Column + (*Bias);
	}

	/************************************************
	函数名称：Backward
	功    能:CNN_ConToFull_Layer的反向卷积操作，后向传递
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_ConToFull_Layer<CNN_TYPE>::Backward(const Matrix<CNN_TYPE>* const GradeOut)
	{
		Matrix<CNN_TYPE> Grade(Input_num*Input_size*Input_size, 1);
		Grade = (*Weight)('T')*(*GradeOut); //反向传播梯度
		Column_Change(Grade, GradeIn);      //将梯度分配至每一个矩阵块中

	}

	/************************************************
	函数名称：Updata
	功    能:CNN_ConToFull_Layer更新当前层上的参数
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_ConToFull_Layer<CNN_TYPE>::Updata(const double LS, 
		const Matrix<CNN_TYPE>* const GradeOut, const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		Matrix<CNN_TYPE>temp(Output_num, Input_num*Input_size*Input_size);
		Matrix<CNN_TYPE> DataIn_Column(Input_num*Input_size*Input_size, 1);  //输入项转化为列矩阵
		DataIn_Column = Column_Joint(DataIn);//这种方法虽然浪费一定资源，但是易于将各个模块独立，更稳健

		temp = (*GradeOut)*DataIn_Column('T'); //得到权重的偏置

		//ofstream fout_test("check\\test_linear3_output.txt", ios::binary); //声明打开保存结果文件
		//fout_test << "error:" << endl;
		//fout_test << temp;
		//fout_test << endl << endl << "Weight";
		//fout_test << (*Weight);
		//fout_test << endl << endl;
		//fout_test.close();

		//cout << (*Weight).Count_Ave() <<' '<< (temp*LS).Count_Ave();

		(*Weight) -= temp*LS;
		(*Bias) -= (*GradeOut)*LS;
	};

	/************************************************
	函数名称:API_Data_Forward
	功    能:CNN_ConToFull_Layer 向前（下一层）传递当前层的结果
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const Matrix<CNN_TYPE>* const CNN_ConToFull_Layer<CNN_TYPE>::API_Data_Forward()
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
	const std::vector<Matrix<CNN_TYPE>*> CNN_ConToFull_Layer<CNN_TYPE>::API_Grade_Backward()
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
	void CNN_ConToFull_Layer<CNN_TYPE>::Save_Info(std::ofstream&fout)
	{

		fout << "CNN_ConToFull_Layer" << std::endl;
		fout << "Input_num " << Input_num << std::endl;
		fout << "Output_num " << Output_num << std::endl;
		fout << "Input_size " << Input_size << std::endl;

		fout << std::endl;

		fout << "CNN_ConToFull_Weight" << std::endl;
		fout << (*Weight) << std::endl;
		fout << std::endl;

		fout << "CNN_ConToFull_Bias" << std::endl;
		fout << (*Bias) << std::endl;

		fout << std::endl << std::endl;

	}


#ifdef Check

	const double trans_1fun(const double x)
	{
		return 1;
	}
	void test_CNN_ConToFull()
	{
		CNN_ConToFull_Layer<double> test(2, 4, 3, trans_1fun);


		std::vector<Matrix<double>*>data;
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

			data.push_back(a1);
			data.push_back(a2);
		}

		test.Forward(data);

		(*test.API_Data_Forward()).show();

		cout << std::endl << std::endl << "*************" << std::endl;

		Matrix<double> Grade(4, 1);
		double g[4] = { 1,2,3,4 };
		Grade.assigment(g, sizeof(g));
		test.Backward(&Grade);

		std::vector<Matrix<double>*>r = test.API_Grade_Backward();

		for (unsigned int i = 0; i < r.size(); i++)
			(*r[i]).show();
	}

#endif // Check 


#undef Check
}

#undef Check

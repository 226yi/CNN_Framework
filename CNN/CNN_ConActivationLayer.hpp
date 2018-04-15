#pragma once

/**************************
File Name:CNN_ConActivationLayer.hpp
Author: MiaoMiaoYoung

日志：
在构建好BP网络后
再一次，再一次，再一次，继续攻克卷积神经网络

2017年11月16日建立
主要用于卷积网络卷积层后激活函数的实现
没有验证该模块其正确性
更新了类的说明

2017年11月17日更新
验证了该层网络的正确性
将该类转化为类模板

2017年11月30日更新
再次检查该层网络是否正确
检验正确

2017年12月5日更新
在卷积网络卷积层反向传播过程中
for (unsigned int cnt = 0; cnt < GradeOut.size(); cnt++)
  (*GradeIn[cnt]) = (*GradeOut[cnt]).transfer(Activate_Func_Derivative);
计算反向梯度时
怎么可能只用梯度就可以算出来的！！！
一定是梯度和数据同时有才能计算出来下一层的梯度
链式法则 Loss/x=Loss/y * y/x
Grade=Loss/y Data_derivate=y/x (/表示偏导)
****************************/

#include"Matrix.hpp"

//#define Check

#ifdef Check
extern void Check_CNN_ConActivation();
#endif // Check


namespace yDL
{
//#define CNN_TYPE double
//#define Check

	/************************************************

	功能：实现卷积神经网络的卷积层后的激活函数层

	参数：

	-   const int num                  // 输入、输出矩阵的深度（即输入、输出图片的深度 - 必须保持相同）
	-   const int size                 // 输入、输出矩阵的大小（矩阵为方阵）

	-   const CNN_TYPE(*Activation_Func)(const CNN_TYPE x)                 //激活函数指针
	-   const CNN_TYPE(*Activation_Func_Derivative)(const CNN_TYPE x)      //激活函数导数的指针

	说明：

	-   Forward;              // 卷积层前向传播，前向进行卷积 (const std::vector<Matrix<CNN_TYPE>*>DataIn)
	-   Backward;             // 卷积层反向传递，反向传递误差 (std::vector<Matrix<CNN_TYPE>*>GradeOut)

	//API
	-   API_Data_Forward;     // 向前传递输出层的结果 -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // 向后传递当前层的误差 -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class CNN_TYPE>
	class CNN_ConActivation_Layer
	{

	private:

		std::vector<Matrix<CNN_TYPE>*> DataOut;                  //输出层结果（当前层的输出）
		std::vector<Matrix<CNN_TYPE>*> GradeIn;                  //输入层梯度（当前层的梯度）

		//矩阵的个数与大小，输入层与输出层都是严格相等的，不然就不对了
		int num;    //输入、输出矩阵的个数（即深度）
		int size;   //输入、输出层矩阵的大小

		const CNN_TYPE (*Activate_Func)(const CNN_TYPE x);               //指向激活函数的指针
		const CNN_TYPE (*Activate_Func_Derivative)(const CNN_TYPE x);    //指向激活函数导数的指针

		CNN_ConActivation_Layer(const CNN_ConActivation_Layer&) = delete; //坚决不能有复制构造函数，出事呀

	public:
		explicit CNN_ConActivation_Layer(
			const int NUM, const int SIZE,                                    //输入、输出矩阵的深度(个数) 输入、输出矩阵的大小
			const CNN_TYPE(*Activation_Func)(const CNN_TYPE x),               //指向激活函数的指针
			const CNN_TYPE(*Activation_Func_Derivative)(const CNN_TYPE x)     //指向激活函数导数的指针
		); //构造函数
		
		~CNN_ConActivation_Layer();

		void Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn);        //卷积网络前向传播
		void Backward(const std::vector<Matrix<CNN_TYPE>*>DataIn, 
			const std::vector<Matrix<CNN_TYPE>*>GradeOut);               //卷积网络反向传递

		const std::vector<Matrix<CNN_TYPE>*> API_Data_Forward();         //向前传播结果 - API
		const std::vector<Matrix<CNN_TYPE>*> API_Grade_Backward();       //反向传递梯度 - API

#ifdef Check
		friend void test_ConActivation();
		friend void ::Check_CNN_ConActivation();
#endif

	};

	/************************************************

	函数名称：   CNN_Convolution_Layer
	功    能:    CNN_Convolution_Layer的构造函数
	参    数：
				 -  输入层\输出层图片的深度（个数）
				 -  输入层\输出层图片的大小（长X宽）
	返    回：
	说    明：
				 -  为输出DataOut、输入梯度GradeIn申请相应大小的空间
				 -  DataOut\GradeIn 只是分配空间，并无更多操作，具体需求在其他模块中实现
				 -  可以在程序运行过程中，减少申请内存的步骤，提高效率
	*************************************************/
	template<class CNN_TYPE>
	CNN_ConActivation_Layer<CNN_TYPE>::CNN_ConActivation_Layer(
		const int NUM, const int SIZE,                                    //输入、输出矩阵的深度(个数) 输入、输出矩阵的大小
		const CNN_TYPE(*Activation_Func)(const CNN_TYPE x),               //指向激活函数的指针
		const CNN_TYPE(*Activation_Func_Derivative)(const CNN_TYPE x)     //指向激活函数导数的指针
	) :
		num{ NUM }, size{ SIZE },
		Activate_Func{ Activation_Func },
		Activate_Func_Derivative{ Activation_Func_Derivative }
	{
		//为输出DataOut申请相应的空间
		for (int i = 0; i < NUM; i++) //外层的卷积核们
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(SIZE, SIZE);
			assert(temp != NULL);
			DataOut.push_back(temp);
		}

		//为输入GradeIn申请相应的空间
		for (int i = 0; i < NUM; i++) //外层的卷积核们
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(SIZE,SIZE);
			assert(temp != NULL);
			GradeIn.push_back(temp);
		}
	}

	/************************************************
	函数名称：~CNN_ConActivation_Layer
	功    能:~CNN_ConActivation_Layer的析构函数
	参    数：
	返    回：
	说    明：为 DataOut \ GradeIn 释放相应空间
	*************************************************/
	template<class CNN_TYPE>
	CNN_ConActivation_Layer<CNN_TYPE>::~CNN_ConActivation_Layer()
	{
		//为输出DataOut释放相应的空间
		for (int i = 0; i < num; i++) //外层的卷积核们
			delete DataOut[i];

		//为输入GradeIn释放相应的空间
		for (int i = 0; i < num; i++) //外层的卷积核们
			delete GradeIn[i];
	}

	/************************************************
	函数名称：Forward
	功    能:CNN_ConActivation_Layer的卷积层后的激活操作，前向传播
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_ConActivation_Layer<CNN_TYPE>::Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		for (unsigned int cnt = 0; cnt < DataIn.size(); cnt++)
			(*DataOut[cnt]) = (*DataIn[cnt]).transfer(Activate_Func);
	}

	/************************************************
	函数名称:Backward
	功    能:CNN_ConActivation_Layer的卷积层后激活函数层的反向传播
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_ConActivation_Layer<CNN_TYPE>::Backward(const std::vector<Matrix<CNN_TYPE>*>DataIn, const std::vector<Matrix<CNN_TYPE>*>GradeOut)
	{
		for (unsigned int cnt = 0; cnt < GradeOut.size(); cnt++)
		{
			(*GradeIn[cnt]) = (*GradeOut[cnt]).Hadamard((*DataIn[cnt]).transfer(Activate_Func_Derivative));

			/*(*GradeOut[cnt]).show();
			cout << endl;
			(*DataIn[cnt]).show();
			cout << endl;
			((*DataIn[cnt]).transfer(Activate_Func_Derivative)).show();
			cout << endl << endl;*/
		}

		//(*GradeIn[cnt]) = (*GradeOut[cnt]).transfer(Activate_Func_Derivative);
	}

	/************************************************
	函数名称:API_Data_Forward
	功    能:CNN_ConActivation_Layer 向前（下一层）传递当前层的结果
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_ConActivation_Layer<CNN_TYPE>::API_Data_Forward()
	{
		return DataOut;
	}

	/************************************************
	函数名称:API_Grade_Backward
	功    能:CNN_ConActivation_Layer 反向（前一层）传递当前层的梯度项
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_ConActivation_Layer<CNN_TYPE>::API_Grade_Backward()
	{
		return GradeIn;
	}

#ifdef Check

	const double trans_fun(const double x)
	{
		return 2*x;
	}

	const double trans_fun1(const double x)
	{
		return x * x;
	}


	void test_ConActivation()
	{
		std::vector<Matrix<double>*>data;

		Matrix<double>a1(3,3);
		double b1[9] = { 1,2,0,1,1,3,0,2,2 };
		a1.assigment(b1, sizeof(b1));

		data.push_back(&a1);

		CNN_ConActivation_Layer<double> test(1,3,trans_fun,trans_fun1);

		test.Forward(data);

		std::vector<Matrix<double>*>grade;

		Matrix<double>a(3, 3);
		double b[9] = { 1,2,3,4,5,6,7,8,9 };
		a.assigment(b, sizeof(b));

		grade.push_back(&a);

		test.Backward(grade);

		for (unsigned int i = 0; i < test.API_Data_Forward().size(); i++)
			(*(test.API_Data_Forward())[i]).show();

		for (unsigned int i = 0; i < test.API_Grade_Backward().size(); i++)
			(*(test.API_Grade_Backward())[i]).show();

	}

#endif // Check

#undef Check

}

#undef Check

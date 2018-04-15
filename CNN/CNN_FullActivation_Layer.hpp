#pragma once

/**************************
File Name:CNN_FullActivation_Layer.hpp
Author: MiaoMiaoYoung

日志：
在构建好BP网络后
再一次，再一次，再一次，继续攻克卷积神经网络

2017年11月17日建立
该层实现全连接层的激活函数
说明未进行更新
未验证其正确性

2017年11月18日更新
验证其正确性

****************************/

#include"Matrix.hpp"

namespace yDL
{

//#define CNN_TYPE double
//#define Check

	/************************************************

	功能：实现卷积神经网络的卷积层后的激活函数层

	参数：

	-   const int Node_num         // 全连接层节点的个数
	-   const CNN_TYPE(*Activation_Func)(const CNN_TYPE x)                 //激活函数指针
	-   const CNN_TYPE(*Activation_Func_Derivative)(const CNN_TYPE x)      //激活函数导数的指针

	说明：

	-   Forward;              // 全连接层前向传播， (const std::vector<Matrix<CNN_TYPE>*>DataIn)
	-   Backward;             // 卷积层反向传递，反向传递误差 (std::vector<Matrix<CNN_TYPE>*>GradeOut)

	//API
	-   API_Data_Forward;     // 向前传递输出层的结果 -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // 向后传递当前层的误差 -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class CNN_TYPE>
	class CNN_FullActivation_Layer
	{
	private:

		Matrix<CNN_TYPE>* DataOut;    //输出层的结果（即当前层的输出）
		Matrix<CNN_TYPE>* GradeIn;    //输入层的梯度（即当前层的梯度）

		int Node_Num; //该层节点的个数

		const CNN_TYPE(*Activate_Func)(const CNN_TYPE x);               //指向激活函数的指针
		const CNN_TYPE(*Activate_Func_Derivative)(const CNN_TYPE x);    //指向激活函数导数的指针

		CNN_FullActivation_Layer(const CNN_FullActivation_Layer&) = delete;

	public:
		explicit CNN_FullActivation_Layer(
			const int node_num,
			const CNN_TYPE Activation_Func(const CNN_TYPE x),
			const CNN_TYPE Activation_Func_Derivative(const CNN_TYPE x)
		); //构造函数

		~CNN_FullActivation_Layer(); //析构函数
		
		void Forward(const Matrix<double>* const DataIn);     //正向传播数据
		void Backward(const Matrix<double>* const GradeOut, 
			const Matrix<double>* const DataIn);              //反向传递误差

		const Matrix<CNN_TYPE>* const API_Data_Forward();         //向前传播结果 - API
		const Matrix<CNN_TYPE>* const API_Grade_Backward();       //反向传递梯度 - API

#ifdef Check
		friend void test_FullActivation();
#endif // Check

	};

	/************************************************
	函数名称：   CNN_FullActivation_Layer
	功    能:    CNN_FullActivation_Layer的构造函数
	参    数：
	             -  节点个数
				 -  激活函数的指针
	             -  激活函数导数的指针
	返    回：
	说    明：为DataOut\GradeIn分配空间
	*************************************************/
	template<class CNN_TYPE>
	CNN_FullActivation_Layer<CNN_TYPE>::CNN_FullActivation_Layer(
		const int node_num,
		const CNN_TYPE Activation_Func(const CNN_TYPE x),
		const CNN_TYPE Activation_Func_Derivative(const CNN_TYPE x)
	):
		Node_Num{ node_num },
		Activate_Func{ Activation_Func },
		Activate_Func_Derivative{ Activation_Func_Derivative }
	{
		DataOut = new(std::nothrow)Matrix<CNN_TYPE>(node_num, 1);
		assert(DataOut != NULL);

		GradeIn = new(std::nothrow)Matrix<CNN_TYPE>(node_num, 1);
		assert(GradeIn != NULL);
	}

	/************************************************
	函数名称：~CNN_FullActivation_Layer
	功    能:  CNN_FullActivation_Layer的析构函数
	参    数：
	返    回：
	说    明：为DataOut\GradeIn释放空间
	*************************************************/
	template<class CNN_TYPE>
	CNN_FullActivation_Layer<CNN_TYPE>::~CNN_FullActivation_Layer()
	{
		delete DataOut;
		delete GradeIn;
	}

	/************************************************
	函数名称：Forward
	功    能:CNN_FullActivation_Layer的全连接层后的激活操作，前向传播
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullActivation_Layer<CNN_TYPE>::Forward(const Matrix<double>* const DataIn)
	{
		(*DataOut) = (*DataIn).transfer(Activate_Func);
	}

	/************************************************
	函数名称：Backward
	功    能:CNN_FullActivation_Layer的全连接层后的激活操作，前向传播
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullActivation_Layer<CNN_TYPE>::Backward(
		const Matrix<double>* const GradeOut,
		const Matrix<double>* const DataIn)
	{
		*GradeIn = (*GradeOut).Hadamard((*DataIn).transfer(Activate_Func_Derivative));
	}

	/************************************************
	函数名称：API_Data_Forward
	功    能:提供前向传播API接口
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const Matrix<CNN_TYPE>* const CNN_FullActivation_Layer<CNN_TYPE>::API_Data_Forward()
	{
		return DataOut;
	}

	/************************************************
	函数名称：API_Grade_Backward
	功    能:提供后向传递API接口
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const Matrix<CNN_TYPE>* const CNN_FullActivation_Layer<CNN_TYPE>::API_Grade_Backward()
	{
		return GradeIn;
	}
	
#ifdef Check

	const double trans_fun(const double x)
	{
		return 2 * x;
	}

	const double trans_fun1(const double x)
	{
		return x * x;
	}

	void test_FullActivation()
	{
		CNN_FullActivation_Layer<double> test(3,trans_fun,trans_fun1);
		
		Matrix<double> a(3, 1);

		double as[3] = { 1,2,3 };

		a.assigment(as, sizeof(as));

		std::vector<Matrix<double>*> b;
		b.push_back(&a);

		test.Forward(&a);
		test.Backward(&a,&a);

		(*test.API_Data_Forward()).show();
		(*test.API_Grade_Backward()).show();
	}
#endif // Check

#undef Check	

}// namespace yDL

#undef Check

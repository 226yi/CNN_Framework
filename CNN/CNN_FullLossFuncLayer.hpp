#pragma once

/**************************
File Name:CNN_FullLossFuncLayer.hpp
Author: MiaoMiaoYoung

日志：
在构建好BP网络后
再一次，再一次，再一次，继续攻克卷积神经网络

2017年11月18日建立
对softmax激活函数进行推导后
建立损失函数层，计算出损失函数，并进行反向传递误差
初步验证通过
转化为类模板
更新了类的说明

2017年11月19日更新
更改了计算分数的方式
****************************/

#include"Matrix.hpp"
#include<cmath>

//#define Check

#ifdef Check
extern void test_CNN_FullLossFuncLayer();
#endif // Check


namespace yDL
{

//#define CNN_TYPE double
//#define Check

	/************************************************
	功能：实现卷积神经网络的损失函数层（SoftMax）
	参数：
	-   const int Node_Num
	说明：
	-   Cal_Score    // 计算当前各个标签的分数 (const Matrix<CNN_TYPE>* const DataIn)
	-   Cal_Loss     // 计算当前卷积神经网络的损失(const int label)
	-   Cal_Grade    // 计算当前卷积神经网络向前回传的梯度项

	//API
	-   API_Grade_Backward(); //向后传递当前层的误差--const Matrix<double>* const
	*************************************************/
	template<class CNN_TYPE>
	class CNN_FullLossFunc_Softmax_Layer
	{

	private:
		std::vector<double> score;    //输出层的结果，输出每一个分类的分数
		Matrix<CNN_TYPE>* GradeIn;    //输入层的梯度，计算每一个分类的梯度项

		int node_num;                 //节点个数
		double Loss;                  //当前损失值

		CNN_FullLossFunc_Softmax_Layer(const CNN_FullLossFunc_Softmax_Layer&) = delete; //复制构造函数删去

	public:
		explicit CNN_FullLossFunc_Softmax_Layer(const int Node_Num); //构造函数
		~CNN_FullLossFunc_Softmax_Layer();                           //析构函数

		void Cal_Score(const Matrix<CNN_TYPE>* const DataIn);        //计算当前的损失分数
		void Cal_Loss(const int Label);                              //计算当前的损失值
		void Cal_Grade(const int Label);                             //计算当前返回的梯度项

		const double return_Loss();         //返回当前样例的损失值
		const int return_Label();           //返回当前样例的标签

		const Matrix<CNN_TYPE>* const API_Grade_Backward();            //向后传递当前层的误差

#ifdef Check
		friend void test_CNN_FullLoss_Softmax();
#endif // Check

	};

	/************************************************
	函数名称：CNN_FullLossFunc_Softmax_Layer
	功    能: CNN_FullLossFunc_Softmax_Layer的构造函数
	参    数：结点个数
	返    回：
	说    明：为DataOut\GradeIn分配空间
	*************************************************/
	template<class CNN_TYPE>
	CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::CNN_FullLossFunc_Softmax_Layer(const int Node_Num)
		:node_num{ Node_Num }, Loss{ 0 }
	{
		//为分数项分配空间
		for (int cnt = 0; cnt < Node_Num; cnt++)
			score.push_back(0);

		//回传梯度项分配空间
		GradeIn = new(std::nothrow)Matrix<CNN_TYPE>(Node_Num, 1);
		assert(GradeIn != NULL);
	}

	/************************************************
	函数名称：CNN_FullLossFunc_Softmax_Layer
	功    能: ~CNN_FullLossFunc_Softmax_Layer的析构函数
	参    数：结点个数
	返    回：
	说    明：为DataOut\GradeIn释放空间
	*************************************************/
	template<class CNN_TYPE>
	CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::~CNN_FullLossFunc_Softmax_Layer()
	{
		delete GradeIn;
	}

	/************************************************
	函数名称：Cal_Score
	功    能: CNN_FullLossFunc_Softmax_Layer的计算当前所得分数
	参    数：输入项
	返    回：
	说    明：Softmax作为损失函数，进行求其分数的操作
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::Cal_Score(const Matrix<CNN_TYPE>* const DataIn)
	{
		double max = 0;
		for (int cnt = 0; cnt < node_num; cnt++)
			if (cnt == 0)
				max = (*DataIn).Get_Value(cnt, 0);
			else
			{
				double value = (*DataIn).Get_Value(cnt, 0);
				if (value > max)
					max = value;
			}//避免数值爆炸，而将最大的取出来

		double sum = 0;
		for (int cnt = 0; cnt < node_num; cnt++)
			sum += std::exp((*DataIn).Get_Value(cnt, 0) - max);

		for (int cnt = 0; cnt < node_num; cnt++)
			score[cnt] = std::exp((*DataIn).Get_Value(cnt, 0) - max) / sum;

		//double max = 0;
		//for (int cnt = 0; cnt < node_num; cnt++)
		//	if (cnt == 0)
		//		max = (*DataIn).Get_Value(cnt, 0);
		//	else
		//	{
		//		double value = (*DataIn).Get_Value(cnt, 0);
		//		if (value > max)
		//			max = value;
		//	}//避免数值爆炸，而将最大的取出来
		//double sum = 0;
		//for (int cnt = 0; cnt < node_num; cnt++)
		//	sum += std::exp((*DataIn).Get_Value(cnt, 0) - max);
		//for (int cnt = 0; cnt < node_num; cnt++)
		//	score[cnt] = std::exp((*DataIn).Get_Value(cnt, 0) - max) / sum;

	}

	/************************************************
	函数名称：Cal_Loss
	功    能: CNN_FullLossFunc_Softmax_Layer的计算当前网络损失值
	参    数：当前正确分类的标签
	返    回：
	说    明：Softmax作为损失函数，进行求其损失值的操作
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::Cal_Loss(const int Label)
	{
		const double eps = 1e-200;
		Loss = -log(score[Label]+eps);
	}

	/************************************************
	函数名称：Cal_Grade
	功    能: CNN_FullLossFunc_Softmax_Layer的计算当前回传梯度项
	参    数：当前正确分类的标签
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::Cal_Grade(const int Label)
	{
		for (int cnt = 0; cnt < node_num; cnt++)
			if (cnt == Label)
				(*GradeIn)[cnt][0] = (CNN_TYPE)(score[cnt] - 1.0);
			else
				(*GradeIn)[cnt][0] = (CNN_TYPE)(score[cnt]);
	}

	/************************************************
	函数名称：API_Grade_Backward
	功    能: CNN_FullLossFunc_Softmax_Layer的API接口，回传梯度项
	参    数：当前正确分类的标签
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const Matrix<CNN_TYPE>* const CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::API_Grade_Backward()
	{
		return GradeIn;
	}

	/************************************************
	函数名称：return_Loss
	功    能: CNN_FullLossFunc_Softmax_Layer返回当前的损失值
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const double CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::return_Loss()
	{
		return Loss;
	}

	/************************************************
	函数名称：return_Label
	功    能: CNN_FullLossFunc_Softmax_Layer返回当前样例的标签
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const int CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::return_Label()
	{
		double max_value = 0;
		int label = 0;

		for (int cnt = 0; cnt < node_num; cnt++)
		{
			if (cnt == 0)
			{
				max_value =score[cnt];
				label = cnt;
				continue;
			}

			double temp_value= score[cnt];
			if (temp_value > max_value)
			{
				max_value = temp_value;
				label = cnt;
			}
			else if (fabs(temp_value - max_value) < 1e-5)
				label = -1;

		}

		return label;
	}


#ifdef Check

	/************************************************
	函数名称：test_CNN_FullLoss_Softmax
	功    能: CNN_FullLossFunc_Softmax_Layer的测试函数
	参    数：
	返    回：
	说    明：
	*************************************************/
	void test_CNN_FullLoss_Softmax()
	{
		Matrix<double> DataIn(3, 1);
		double d[3] = { 1,2,3 };
		DataIn.assigment(d, sizeof(d));

		CNN_FullLossFunc_Softmax_Layer<double> test(3);

		test.Cal_Score(&DataIn);
		test.Cal_Loss(1);
		test.Cal_Grade(1);

		(*test.API_Grade_Backward()).show();
	}

#endif // Check

#undef Check
}

#undef Check

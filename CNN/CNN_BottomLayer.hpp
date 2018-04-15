#pragma once

/**************************
File Name:CNN_BottomLayer.hpp
Author: MiaoMiaoYoung

日志：
在构建好BP网络后
再一次，再一次，再一次，继续攻克卷积神经网络

2017年11月18日建立
用于神经网络底层的建立
传递处理数据并做相应的预处理

2017年11月19日
添加了归一化预处理
****************************/

#include"Matrix.hpp"

namespace yDL
{

//#define CNN_TYPE double

	/************************************************

	功能：实现卷积神经网络的卷积层

	参数：
	-   const int In_num               // 输入矩阵的深度（即输入图片的深度）
	-   const int In_size              // 输入矩阵的大小（矩阵为方阵）

	说明：

	-   Forward;              // 卷积层前向传播，前向进行卷积 (const std::vector<Matrix<CNN_TYPE>*>DataIn)

	//API
	-   API_Data_Forward;     // 向前传递输出层的结果 -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // 向后传递当前层的误差 -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class T, class CNN_TYPE>
	class CNN_Botton_Layer
	{
		std::vector<Matrix<CNN_TYPE>*> DataOut;                  //输出层结果（当前层的输出）
		std::vector<Matrix<CNN_TYPE>*> GradeIn;                  //输入层梯度（当前层的梯度）

		int Input_num;    //输入矩阵的个数（即深度）
		int Input_size;   //输入层矩阵的大小

		CNN_Botton_Layer(const CNN_Botton_Layer&) = delete;

	public:

		CNN_Botton_Layer(const int In_num, const int In_size); //构造函数
		~CNN_Botton_Layer(); //析构函数

		void Load_Data(const T* const data);       //装载数据

		void Zero_Center();                        //数据预处理，中心化
		void Normalization();                      //数据预处理，归一化

		const std::vector<Matrix<CNN_TYPE>*> API_Data_Forward();                //向前传播结果 - API
		const std::vector<Matrix<CNN_TYPE>*> API_Grade_Backward();              //反向传递梯度 - API

	};

	/************************************************
	函数名称：   CNN_Botton_Layer
	功    能:    CNN_Botton_Layer的构造函数
	参    数：	-  输入层图片的深度（个数） -  输入层图片的大小（长X宽）
	返    回：
	说    明：	-  DataOut 只是分配空间，并无更多操作，具体需求在其他模块中实现
	*************************************************/
	template<class T, class CNN_TYPE>
	CNN_Botton_Layer<T,CNN_TYPE>::CNN_Botton_Layer(const int In_num, const int In_size):
		Input_num{ In_num }, Input_size{ In_size }
	{
		for (int cnt = 0; cnt < Input_num; cnt++)
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(In_size, In_size);
			assert(temp != NULL);

			DataOut.push_back(temp);
		}
	}

	/************************************************
	函数名称：~CNN_Botton_Layer
	功    能: CNN_Botton_Layer的析构函数
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class T, class CNN_TYPE>
	CNN_Botton_Layer<T,CNN_TYPE>::~CNN_Botton_Layer()
	{
		for (int i = 0; i < Input_num; i++)
		{
			delete DataOut[i];
			DataOut[i] = NULL;
		}
	}

	/************************************************
	函数名称：Load_Data
	功    能: CNN_Botton_Layer的装载数据
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class T, class CNN_TYPE>
	void CNN_Botton_Layer<T,CNN_TYPE>::Load_Data(const T* const data)
	{
		for (int cnt = 0;cnt < Input_num; cnt++)
		{
			for (int i = 0; i < Input_size; i++)
				for (int j = 0; j < Input_size; j++)
					(*DataOut[cnt])[i][j] = data[cnt*Input_size*Input_size + i*Input_size + j];
		}
	}

	/************************************************
	函数名称：Zero_Center
	功    能: CNN_Botton_Layer的预处理，零中心化
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class T, class CNN_TYPE>
	void CNN_Botton_Layer<T,CNN_TYPE>::Zero_Center()
	{
		for (unsigned int cnt = 0; cnt < DataOut.size(); cnt++)
		{
			(*DataOut[cnt]) = (*DataOut[cnt]).Zero_Center();
		}
	}

	/************************************************
	函数名称：Zero_Center
	功    能: CNN_Botton_Layer的预处理，归一化
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class T, class CNN_TYPE>
	void CNN_Botton_Layer<T, CNN_TYPE>::Normalization()
	{
		for (unsigned int cnt = 0; cnt < DataOut.size(); cnt++)
		{
			(*DataOut[cnt]) = (*DataOut[cnt]).Normalization(0,225);
		}
	}

	/************************************************
	函数名称:API_Data_Forward
	功    能:CNN_Botton_Layer 向前（下一层）传递当前层的结果
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class T, class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_Botton_Layer<T,CNN_TYPE>::API_Data_Forward()
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
	template<class T, class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_Botton_Layer<T,CNN_TYPE>::API_Grade_Backward()
	{
		return GradeIn;
	}

#undef Check

}

#undef Check

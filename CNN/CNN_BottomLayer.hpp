#pragma once

/**************************
File Name:CNN_BottomLayer.hpp
Author: MiaoMiaoYoung

��־��
�ڹ�����BP�����
��һ�Σ���һ�Σ���һ�Σ��������˾��������

2017��11��18�ս���
����������ײ�Ľ���
���ݴ������ݲ�����Ӧ��Ԥ����

2017��11��19��
����˹�һ��Ԥ����
****************************/

#include"Matrix.hpp"

namespace yDL
{

//#define CNN_TYPE double

	/************************************************

	���ܣ�ʵ�־��������ľ����

	������
	-   const int In_num               // ����������ȣ�������ͼƬ����ȣ�
	-   const int In_size              // �������Ĵ�С������Ϊ����

	˵����

	-   Forward;              // �����ǰ�򴫲���ǰ����о�� (const std::vector<Matrix<CNN_TYPE>*>DataIn)

	//API
	-   API_Data_Forward;     // ��ǰ���������Ľ�� -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // ��󴫵ݵ�ǰ������ -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class T, class CNN_TYPE>
	class CNN_Botton_Layer
	{
		std::vector<Matrix<CNN_TYPE>*> DataOut;                  //�����������ǰ��������
		std::vector<Matrix<CNN_TYPE>*> GradeIn;                  //������ݶȣ���ǰ����ݶȣ�

		int Input_num;    //�������ĸ���������ȣ�
		int Input_size;   //��������Ĵ�С

		CNN_Botton_Layer(const CNN_Botton_Layer&) = delete;

	public:

		CNN_Botton_Layer(const int In_num, const int In_size); //���캯��
		~CNN_Botton_Layer(); //��������

		void Load_Data(const T* const data);       //װ������

		void Zero_Center();                        //����Ԥ�������Ļ�
		void Normalization();                      //����Ԥ������һ��

		const std::vector<Matrix<CNN_TYPE>*> API_Data_Forward();                //��ǰ������� - API
		const std::vector<Matrix<CNN_TYPE>*> API_Grade_Backward();              //���򴫵��ݶ� - API

	};

	/************************************************
	�������ƣ�   CNN_Botton_Layer
	��    ��:    CNN_Botton_Layer�Ĺ��캯��
	��    ����	-  �����ͼƬ����ȣ������� -  �����ͼƬ�Ĵ�С����X��
	��    �أ�
	˵    ����	-  DataOut ֻ�Ƿ���ռ䣬���޸����������������������ģ����ʵ��
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
	�������ƣ�~CNN_Botton_Layer
	��    ��: CNN_Botton_Layer����������
	��    ����
	��    �أ�
	˵    ����
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
	�������ƣ�Load_Data
	��    ��: CNN_Botton_Layer��װ������
	��    ����
	��    �أ�
	˵    ����
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
	�������ƣ�Zero_Center
	��    ��: CNN_Botton_Layer��Ԥ���������Ļ�
	��    ����
	��    �أ�
	˵    ����
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
	�������ƣ�Zero_Center
	��    ��: CNN_Botton_Layer��Ԥ������һ��
	��    ����
	��    �أ�
	˵    ����
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
	��������:API_Data_Forward
	��    ��:CNN_Botton_Layer ��ǰ����һ�㣩���ݵ�ǰ��Ľ��
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class T, class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_Botton_Layer<T,CNN_TYPE>::API_Data_Forward()
	{
		return DataOut;
	}

	/************************************************
	��������:API_Grade_Backward
	��    ��:CNN_Convolution_Layer ����ǰһ�㣩���ݵ�ǰ����ݶ���
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class T, class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_Botton_Layer<T,CNN_TYPE>::API_Grade_Backward()
	{
		return GradeIn;
	}

#undef Check

}

#undef Check

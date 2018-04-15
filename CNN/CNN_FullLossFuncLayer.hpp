#pragma once

/**************************
File Name:CNN_FullLossFuncLayer.hpp
Author: MiaoMiaoYoung

��־��
�ڹ�����BP�����
��һ�Σ���һ�Σ���һ�Σ��������˾��������

2017��11��18�ս���
��softmax����������Ƶ���
������ʧ�����㣬�������ʧ�����������з��򴫵����
������֤ͨ��
ת��Ϊ��ģ��
���������˵��

2017��11��19�ո���
�����˼�������ķ�ʽ
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
	���ܣ�ʵ�־�����������ʧ�����㣨SoftMax��
	������
	-   const int Node_Num
	˵����
	-   Cal_Score    // ���㵱ǰ������ǩ�ķ��� (const Matrix<CNN_TYPE>* const DataIn)
	-   Cal_Loss     // ���㵱ǰ������������ʧ(const int label)
	-   Cal_Grade    // ���㵱ǰ�����������ǰ�ش����ݶ���

	//API
	-   API_Grade_Backward(); //��󴫵ݵ�ǰ������--const Matrix<double>* const
	*************************************************/
	template<class CNN_TYPE>
	class CNN_FullLossFunc_Softmax_Layer
	{

	private:
		std::vector<double> score;    //�����Ľ�������ÿһ������ķ���
		Matrix<CNN_TYPE>* GradeIn;    //�������ݶȣ�����ÿһ��������ݶ���

		int node_num;                 //�ڵ����
		double Loss;                  //��ǰ��ʧֵ

		CNN_FullLossFunc_Softmax_Layer(const CNN_FullLossFunc_Softmax_Layer&) = delete; //���ƹ��캯��ɾȥ

	public:
		explicit CNN_FullLossFunc_Softmax_Layer(const int Node_Num); //���캯��
		~CNN_FullLossFunc_Softmax_Layer();                           //��������

		void Cal_Score(const Matrix<CNN_TYPE>* const DataIn);        //���㵱ǰ����ʧ����
		void Cal_Loss(const int Label);                              //���㵱ǰ����ʧֵ
		void Cal_Grade(const int Label);                             //���㵱ǰ���ص��ݶ���

		const double return_Loss();         //���ص�ǰ��������ʧֵ
		const int return_Label();           //���ص�ǰ�����ı�ǩ

		const Matrix<CNN_TYPE>* const API_Grade_Backward();            //��󴫵ݵ�ǰ������

#ifdef Check
		friend void test_CNN_FullLoss_Softmax();
#endif // Check

	};

	/************************************************
	�������ƣ�CNN_FullLossFunc_Softmax_Layer
	��    ��: CNN_FullLossFunc_Softmax_Layer�Ĺ��캯��
	��    ����������
	��    �أ�
	˵    ����ΪDataOut\GradeIn����ռ�
	*************************************************/
	template<class CNN_TYPE>
	CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::CNN_FullLossFunc_Softmax_Layer(const int Node_Num)
		:node_num{ Node_Num }, Loss{ 0 }
	{
		//Ϊ���������ռ�
		for (int cnt = 0; cnt < Node_Num; cnt++)
			score.push_back(0);

		//�ش��ݶ������ռ�
		GradeIn = new(std::nothrow)Matrix<CNN_TYPE>(Node_Num, 1);
		assert(GradeIn != NULL);
	}

	/************************************************
	�������ƣ�CNN_FullLossFunc_Softmax_Layer
	��    ��: ~CNN_FullLossFunc_Softmax_Layer����������
	��    ����������
	��    �أ�
	˵    ����ΪDataOut\GradeIn�ͷſռ�
	*************************************************/
	template<class CNN_TYPE>
	CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::~CNN_FullLossFunc_Softmax_Layer()
	{
		delete GradeIn;
	}

	/************************************************
	�������ƣ�Cal_Score
	��    ��: CNN_FullLossFunc_Softmax_Layer�ļ��㵱ǰ���÷���
	��    ����������
	��    �أ�
	˵    ����Softmax��Ϊ��ʧ������������������Ĳ���
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
			}//������ֵ��ը����������ȡ����

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
		//	}//������ֵ��ը����������ȡ����
		//double sum = 0;
		//for (int cnt = 0; cnt < node_num; cnt++)
		//	sum += std::exp((*DataIn).Get_Value(cnt, 0) - max);
		//for (int cnt = 0; cnt < node_num; cnt++)
		//	score[cnt] = std::exp((*DataIn).Get_Value(cnt, 0) - max) / sum;

	}

	/************************************************
	�������ƣ�Cal_Loss
	��    ��: CNN_FullLossFunc_Softmax_Layer�ļ��㵱ǰ������ʧֵ
	��    ������ǰ��ȷ����ı�ǩ
	��    �أ�
	˵    ����Softmax��Ϊ��ʧ����������������ʧֵ�Ĳ���
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::Cal_Loss(const int Label)
	{
		const double eps = 1e-200;
		Loss = -log(score[Label]+eps);
	}

	/************************************************
	�������ƣ�Cal_Grade
	��    ��: CNN_FullLossFunc_Softmax_Layer�ļ��㵱ǰ�ش��ݶ���
	��    ������ǰ��ȷ����ı�ǩ
	��    �أ�
	˵    ����
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
	�������ƣ�API_Grade_Backward
	��    ��: CNN_FullLossFunc_Softmax_Layer��API�ӿڣ��ش��ݶ���
	��    ������ǰ��ȷ����ı�ǩ
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	const Matrix<CNN_TYPE>* const CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::API_Grade_Backward()
	{
		return GradeIn;
	}

	/************************************************
	�������ƣ�return_Loss
	��    ��: CNN_FullLossFunc_Softmax_Layer���ص�ǰ����ʧֵ
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	const double CNN_FullLossFunc_Softmax_Layer<CNN_TYPE>::return_Loss()
	{
		return Loss;
	}

	/************************************************
	�������ƣ�return_Label
	��    ��: CNN_FullLossFunc_Softmax_Layer���ص�ǰ�����ı�ǩ
	��    ����
	��    �أ�
	˵    ����
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
	�������ƣ�test_CNN_FullLoss_Softmax
	��    ��: CNN_FullLossFunc_Softmax_Layer�Ĳ��Ժ���
	��    ����
	��    �أ�
	˵    ����
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

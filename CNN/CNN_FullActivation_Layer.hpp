#pragma once

/**************************
File Name:CNN_FullActivation_Layer.hpp
Author: MiaoMiaoYoung

��־��
�ڹ�����BP�����
��һ�Σ���һ�Σ���һ�Σ��������˾��������

2017��11��17�ս���
�ò�ʵ��ȫ���Ӳ�ļ����
˵��δ���и���
δ��֤����ȷ��

2017��11��18�ո���
��֤����ȷ��

****************************/

#include"Matrix.hpp"

namespace yDL
{

//#define CNN_TYPE double
//#define Check

	/************************************************

	���ܣ�ʵ�־��������ľ�����ļ������

	������

	-   const int Node_num         // ȫ���Ӳ�ڵ�ĸ���
	-   const CNN_TYPE(*Activation_Func)(const CNN_TYPE x)                 //�����ָ��
	-   const CNN_TYPE(*Activation_Func_Derivative)(const CNN_TYPE x)      //�����������ָ��

	˵����

	-   Forward;              // ȫ���Ӳ�ǰ�򴫲��� (const std::vector<Matrix<CNN_TYPE>*>DataIn)
	-   Backward;             // ����㷴�򴫵ݣ����򴫵���� (std::vector<Matrix<CNN_TYPE>*>GradeOut)

	//API
	-   API_Data_Forward;     // ��ǰ���������Ľ�� -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // ��󴫵ݵ�ǰ������ -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class CNN_TYPE>
	class CNN_FullActivation_Layer
	{
	private:

		Matrix<CNN_TYPE>* DataOut;    //�����Ľ��������ǰ��������
		Matrix<CNN_TYPE>* GradeIn;    //�������ݶȣ�����ǰ����ݶȣ�

		int Node_Num; //�ò�ڵ�ĸ���

		const CNN_TYPE(*Activate_Func)(const CNN_TYPE x);               //ָ�򼤻����ָ��
		const CNN_TYPE(*Activate_Func_Derivative)(const CNN_TYPE x);    //ָ�򼤻��������ָ��

		CNN_FullActivation_Layer(const CNN_FullActivation_Layer&) = delete;

	public:
		explicit CNN_FullActivation_Layer(
			const int node_num,
			const CNN_TYPE Activation_Func(const CNN_TYPE x),
			const CNN_TYPE Activation_Func_Derivative(const CNN_TYPE x)
		); //���캯��

		~CNN_FullActivation_Layer(); //��������
		
		void Forward(const Matrix<double>* const DataIn);     //���򴫲�����
		void Backward(const Matrix<double>* const GradeOut, 
			const Matrix<double>* const DataIn);              //���򴫵����

		const Matrix<CNN_TYPE>* const API_Data_Forward();         //��ǰ������� - API
		const Matrix<CNN_TYPE>* const API_Grade_Backward();       //���򴫵��ݶ� - API

#ifdef Check
		friend void test_FullActivation();
#endif // Check

	};

	/************************************************
	�������ƣ�   CNN_FullActivation_Layer
	��    ��:    CNN_FullActivation_Layer�Ĺ��캯��
	��    ����
	             -  �ڵ����
				 -  �������ָ��
	             -  �����������ָ��
	��    �أ�
	˵    ����ΪDataOut\GradeIn����ռ�
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
	�������ƣ�~CNN_FullActivation_Layer
	��    ��:  CNN_FullActivation_Layer����������
	��    ����
	��    �أ�
	˵    ����ΪDataOut\GradeIn�ͷſռ�
	*************************************************/
	template<class CNN_TYPE>
	CNN_FullActivation_Layer<CNN_TYPE>::~CNN_FullActivation_Layer()
	{
		delete DataOut;
		delete GradeIn;
	}

	/************************************************
	�������ƣ�Forward
	��    ��:CNN_FullActivation_Layer��ȫ���Ӳ��ļ��������ǰ�򴫲�
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullActivation_Layer<CNN_TYPE>::Forward(const Matrix<double>* const DataIn)
	{
		(*DataOut) = (*DataIn).transfer(Activate_Func);
	}

	/************************************************
	�������ƣ�Backward
	��    ��:CNN_FullActivation_Layer��ȫ���Ӳ��ļ��������ǰ�򴫲�
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullActivation_Layer<CNN_TYPE>::Backward(
		const Matrix<double>* const GradeOut,
		const Matrix<double>* const DataIn)
	{
		*GradeIn = (*GradeOut).Hadamard((*DataIn).transfer(Activate_Func_Derivative));
	}

	/************************************************
	�������ƣ�API_Data_Forward
	��    ��:�ṩǰ�򴫲�API�ӿ�
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	const Matrix<CNN_TYPE>* const CNN_FullActivation_Layer<CNN_TYPE>::API_Data_Forward()
	{
		return DataOut;
	}

	/************************************************
	�������ƣ�API_Grade_Backward
	��    ��:�ṩ���򴫵�API�ӿ�
	��    ����
	��    �أ�
	˵    ����
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

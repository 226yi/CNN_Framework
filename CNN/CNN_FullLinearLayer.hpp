#pragma once

/**************************
File Name:CNN_FullLinearLayer.h
Author: MiaoMiaoYoung

��־��
�ڹ�����BP�����
��һ�Σ���һ�Σ���һ�Σ��������˾��������

2017��11��17�ս���
�ò�ʵ��ȫ���Ӳ�����Է���
δ��֤����ȷ��

2017��11��18�ո���
�������Ϊ��ģ��
��֤����ȷ��

2017��11��25�ո���
����Check.cpp���н�һ�����
��updata�и�Ϊ��ģ�����
����ͨ��

2017��11��28�ո���
�������ļ�д�����ܣ�������ѵ��������Ա���
����ͨ��
****************************/

#include"Matrix.hpp"

//#define Check

#ifdef Check
extern void test_CNN_FullLinear_Layer(); //Check.cpp ʵ����֤�Ƿ���ȷ
#endif // Check

namespace yDL
{

//#define CNN_TYPE double

/************************************************
	���ܣ�ʵ������������Է����
	������
  	      -   const int In_num,
	      -   const int Out_num,
	      -   const double LearningSpeed,
	      -   const vector<const Matrix<double>* const> InPointer,
	      -   const vector<const int> LaPointer
	˵����
	      -   f(x)=w(T)*x+b    //���Է���
	      -   Forward(const Matrix<double>&DataIn);                                    //���򴫲�����
	      -   Backward(const Matrix<double>&GradeOut);                                 //���򴫵����
	      -   Updata(const Matrix<double>& GradeOut, const Matrix<double>& DataIn);    //���в����ĸ���

	//API
	      -   API_Data_Forward();   //��ǰ���������Ľ��--const Matrix<double>&
	      -   API_Grade_Backward(); //��󴫵ݵ�ǰ������--const Matrix<double>&
	*************************************************/
	template<class CNN_TYPE>
	class CNN_FullLinear_Layer
	{

	private:

		Matrix<CNN_TYPE>* Weight;  //��ǰ���Ȩ��
		Matrix<CNN_TYPE>* Bias;    //��ǰ�����ֵ

		Matrix<CNN_TYPE>* DataOut;    //�����Ľ��������ǰ��������
		Matrix<CNN_TYPE>* GradeIn;    //�������ݶȣ�����ǰ����ݶȣ�

		int Input_num;       //����ڵ�ĸ���
		int Output_num;      //����ڵ�ĸ���

		CNN_FullLinear_Layer(const CNN_FullLinear_Layer&) = delete;

	public:
		explicit CNN_FullLinear_Layer(
			const int In_num,
			const int Out_num,
			const CNN_TYPE trans_func(const CNN_TYPE x) = NULL
		);//���캯��

		~CNN_FullLinear_Layer();  //��������

		void Forward(const Matrix<CNN_TYPE>* const DataIn);                                      //���򴫲�����
		void Backward(const Matrix<CNN_TYPE>* const GradeOut);                                   //���򴫵����
		void Updata(
			const double LS,
			const Matrix<CNN_TYPE>* const GradeOut,
			const Matrix<CNN_TYPE>* const DataIn
		); //���в����ĸ���

	  //�ṩAPI�ӿ�
		const Matrix<CNN_TYPE>* const API_Data_Forward();   //��ǰ���������Ľ��
		const Matrix<CNN_TYPE>* const API_Grade_Backward(); //��󴫵ݵ�ǰ������

		void Save_Info(std::ofstream& fout); //����ǰ����д���ļ���

#ifdef Check
		friend void test_FullLinear();
		friend void ::test_CNN_FullLinear_Layer(); //Check.cpp ʵ����֤�Ƿ���ȷ
#endif // Check

	};

	/************************************************
	�������ƣ�CNN_FullLinear_Layer
	��    ��: CNN_FullLinear_Layer�Ĺ��캯��
	��    ����
 	           -  ����ڵ����
	           -  ����ڵ����
			   -  ��ʼ��Weight������Ĭ��ΪNULL��
	��    �أ�
	˵    ����ΪDataOut\GradeIn\Weight\Bias����ռ�
	*************************************************/
	template<class CNN_TYPE>
	CNN_FullLinear_Layer<CNN_TYPE>::CNN_FullLinear_Layer(
		const int In_num,const int Out_num,
		const CNN_TYPE trans_func(const CNN_TYPE x)
	):
		Input_num{ In_num },
		Output_num{ Out_num}
	{
		Weight = new(std::nothrow)Matrix<CNN_TYPE>(Out_num, In_num);
		assert(Weight != NULL);

		if (trans_func == NULL)
			(*Weight).Initialize();
		else
			(*Weight) = (*Weight).transfer(trans_func);

		Bias = new(std::nothrow)Matrix<CNN_TYPE>(Out_num, 1);
		assert(Bias != NULL);
		(*Bias).Initialize();

		DataOut = new(std::nothrow)Matrix<CNN_TYPE>(Out_num, 1);
		assert(DataOut != NULL);

		GradeIn = new(std::nothrow)Matrix<CNN_TYPE>(In_num, 1);
		assert(GradeIn != NULL);
	}

	/************************************************
	�������ƣ�~CNN_FullLinear_Layer
	��    ��:  CNN_FullLinear_Layer����������
	��    ����
	��    �أ�
	˵    ����ΪDataOut\GradeIn\Weight\Bias�ͷſռ�
	*************************************************/
	template<class CNN_TYPE>
	CNN_FullLinear_Layer<CNN_TYPE>::~CNN_FullLinear_Layer()
	{
		delete Weight;
		delete Bias;
		delete DataOut;
		delete GradeIn;
	}

	/************************************************
	�������ƣ�Forward
	��    ��:���򴫲�
	��    ����API������Ҫѵ��������
	��    �أ�
	˵    �����������򴫲����㣬����˷�
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullLinear_Layer<CNN_TYPE>::Forward(const Matrix<CNN_TYPE>* const DataIn)
	{
		*DataOut = (*Weight)*(*DataIn) + (*Bias);
	}

	/************************************************
	��������:Backward
	��    ��:���򴫵����
	��    ����API������һ����ݶ���
	��    �أ�
	˵    ������ʽ������з��򴫲�
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullLinear_Layer<CNN_TYPE>::Backward(const Matrix<CNN_TYPE>* const GradeOut)
	{
		*GradeIn = (*Weight)('T')*(*GradeOut); //���򴫲��ݶ�
	}

	/************************************************
	�������ƣ�Updata
	��    ��:���µ�ǰ���ϵĲ���
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullLinear_Layer<CNN_TYPE>::Updata(
		const double LS,
		const Matrix<CNN_TYPE>* const GradeOut,
		const Matrix<CNN_TYPE>* const DataIn
	)
	{
		Matrix<CNN_TYPE>temp(Output_num, Input_num);
		temp = (*GradeOut)*(*DataIn)('T'); //�õ�Ȩ�ص�ƫ��

		//if (1)
		//{
		//	ofstream fout_test("check\\test_linear5_output.txt", ios::binary); //�����򿪱������ļ�
		//	fout_test << "error:" << endl;
		//	fout_test << temp;
		//	fout_test << endl << endl << "Weight";
		//	fout_test << (*Weight);
		//	fout_test << endl << endl;
		//	fout_test.close();
		//}

		//cout << (*Weight).Count_Ave() << ' ' << (temp*LS).Count_Ave() << endl;

		*Weight -= temp*LS;
		*Bias -= (*GradeOut)*LS;
	}

	/************************************************
	�������ƣ�API_Data_Forward
	��    ��:�ṩǰ�򴫲�API�ӿ�
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	const Matrix<CNN_TYPE>* const CNN_FullLinear_Layer<CNN_TYPE>::API_Data_Forward()
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
	const Matrix<CNN_TYPE>* const CNN_FullLinear_Layer<CNN_TYPE>::API_Grade_Backward()
	{
		return GradeIn;
	}

	/************************************************
	��������:Save_Info
	��    ��:��ѵ���ɹ��������ļ��У��պ����ֱ�Ӷ�ȡ
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullLinear_Layer<CNN_TYPE>::Save_Info(std::ofstream&fout)
	{
		fout << "CNN_FullLinear_Layer" << std::endl;
		fout << "Input_num " << Input_num << std::endl;
		fout << "Output_num " << Output_num << std::endl;
		fout << std::endl;
		
		fout << "CNN_Linear_Weight" << std::endl;
		fout << (*Weight)<<std::endl;
		fout << std::endl;

		fout << "CNN_Linear_Bias" << std::endl;
		fout << (*Bias) << std::endl;

		fout << std::endl << std::endl;

	}

#ifdef Check

	const double a(const double x)
	{
		return 1;
	}

	void test_FullLinear()
	{
		CNN_FullLinear_Layer<double> test(4, 2,a);

		Matrix<double> as(4, 1);
		double bs[4] = { 1,2,3,4 };
		as.assigment(bs, sizeof(bs));
		
		test.Forward(&as);

		(*test.API_Data_Forward()).show();

		Matrix<double> ad(2, 1);
		double bd[2] = { 1,2 };
		ad.assigment(bd, sizeof(bd));

		test.Backward(&ad);

		(*test.API_Grade_Backward()).show();
	}
#endif // Check
#undef Check

}

#undef Check

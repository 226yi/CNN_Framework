#pragma once

/**************************
File Name:CNN_ConActivationLayer.hpp
Author: MiaoMiaoYoung

��־��
�ڹ�����BP�����
��һ�Σ���һ�Σ���һ�Σ��������˾��������

2017��11��16�ս���
��Ҫ���ھ����������󼤻����ʵ��
û����֤��ģ������ȷ��
���������˵��

2017��11��17�ո���
��֤�˸ò��������ȷ��
������ת��Ϊ��ģ��

2017��11��30�ո���
�ٴμ��ò������Ƿ���ȷ
������ȷ

2017��12��5�ո���
�ھ���������㷴�򴫲�������
for (unsigned int cnt = 0; cnt < GradeOut.size(); cnt++)
  (*GradeIn[cnt]) = (*GradeOut[cnt]).transfer(Activate_Func_Derivative);
���㷴���ݶ�ʱ
��ô����ֻ���ݶȾͿ���������ģ�����
һ�����ݶȺ�����ͬʱ�в��ܼ��������һ����ݶ�
��ʽ���� Loss/x=Loss/y * y/x
Grade=Loss/y Data_derivate=y/x (/��ʾƫ��)
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

	���ܣ�ʵ�־��������ľ�����ļ������

	������

	-   const int num                  // ���롢����������ȣ������롢���ͼƬ����� - ���뱣����ͬ��
	-   const int size                 // ���롢�������Ĵ�С������Ϊ����

	-   const CNN_TYPE(*Activation_Func)(const CNN_TYPE x)                 //�����ָ��
	-   const CNN_TYPE(*Activation_Func_Derivative)(const CNN_TYPE x)      //�����������ָ��

	˵����

	-   Forward;              // �����ǰ�򴫲���ǰ����о�� (const std::vector<Matrix<CNN_TYPE>*>DataIn)
	-   Backward;             // ����㷴�򴫵ݣ����򴫵���� (std::vector<Matrix<CNN_TYPE>*>GradeOut)

	//API
	-   API_Data_Forward;     // ��ǰ���������Ľ�� -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // ��󴫵ݵ�ǰ������ -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class CNN_TYPE>
	class CNN_ConActivation_Layer
	{

	private:

		std::vector<Matrix<CNN_TYPE>*> DataOut;                  //�����������ǰ��������
		std::vector<Matrix<CNN_TYPE>*> GradeIn;                  //������ݶȣ���ǰ����ݶȣ�

		//����ĸ������С�������������㶼���ϸ���ȵģ���Ȼ�Ͳ�����
		int num;    //���롢�������ĸ���������ȣ�
		int size;   //���롢��������Ĵ�С

		const CNN_TYPE (*Activate_Func)(const CNN_TYPE x);               //ָ�򼤻����ָ��
		const CNN_TYPE (*Activate_Func_Derivative)(const CNN_TYPE x);    //ָ�򼤻��������ָ��

		CNN_ConActivation_Layer(const CNN_ConActivation_Layer&) = delete; //��������и��ƹ��캯��������ѽ

	public:
		explicit CNN_ConActivation_Layer(
			const int NUM, const int SIZE,                                    //���롢�����������(����) ���롢�������Ĵ�С
			const CNN_TYPE(*Activation_Func)(const CNN_TYPE x),               //ָ�򼤻����ָ��
			const CNN_TYPE(*Activation_Func_Derivative)(const CNN_TYPE x)     //ָ�򼤻��������ָ��
		); //���캯��
		
		~CNN_ConActivation_Layer();

		void Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn);        //�������ǰ�򴫲�
		void Backward(const std::vector<Matrix<CNN_TYPE>*>DataIn, 
			const std::vector<Matrix<CNN_TYPE>*>GradeOut);               //������練�򴫵�

		const std::vector<Matrix<CNN_TYPE>*> API_Data_Forward();         //��ǰ������� - API
		const std::vector<Matrix<CNN_TYPE>*> API_Grade_Backward();       //���򴫵��ݶ� - API

#ifdef Check
		friend void test_ConActivation();
		friend void ::Check_CNN_ConActivation();
#endif

	};

	/************************************************

	�������ƣ�   CNN_Convolution_Layer
	��    ��:    CNN_Convolution_Layer�Ĺ��캯��
	��    ����
				 -  �����\�����ͼƬ����ȣ�������
				 -  �����\�����ͼƬ�Ĵ�С����X��
	��    �أ�
	˵    ����
				 -  Ϊ���DataOut�������ݶ�GradeIn������Ӧ��С�Ŀռ�
				 -  DataOut\GradeIn ֻ�Ƿ���ռ䣬���޸����������������������ģ����ʵ��
				 -  �����ڳ������й����У����������ڴ�Ĳ��裬���Ч��
	*************************************************/
	template<class CNN_TYPE>
	CNN_ConActivation_Layer<CNN_TYPE>::CNN_ConActivation_Layer(
		const int NUM, const int SIZE,                                    //���롢�����������(����) ���롢�������Ĵ�С
		const CNN_TYPE(*Activation_Func)(const CNN_TYPE x),               //ָ�򼤻����ָ��
		const CNN_TYPE(*Activation_Func_Derivative)(const CNN_TYPE x)     //ָ�򼤻��������ָ��
	) :
		num{ NUM }, size{ SIZE },
		Activate_Func{ Activation_Func },
		Activate_Func_Derivative{ Activation_Func_Derivative }
	{
		//Ϊ���DataOut������Ӧ�Ŀռ�
		for (int i = 0; i < NUM; i++) //���ľ������
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(SIZE, SIZE);
			assert(temp != NULL);
			DataOut.push_back(temp);
		}

		//Ϊ����GradeIn������Ӧ�Ŀռ�
		for (int i = 0; i < NUM; i++) //���ľ������
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(SIZE,SIZE);
			assert(temp != NULL);
			GradeIn.push_back(temp);
		}
	}

	/************************************************
	�������ƣ�~CNN_ConActivation_Layer
	��    ��:~CNN_ConActivation_Layer����������
	��    ����
	��    �أ�
	˵    ����Ϊ DataOut \ GradeIn �ͷ���Ӧ�ռ�
	*************************************************/
	template<class CNN_TYPE>
	CNN_ConActivation_Layer<CNN_TYPE>::~CNN_ConActivation_Layer()
	{
		//Ϊ���DataOut�ͷ���Ӧ�Ŀռ�
		for (int i = 0; i < num; i++) //���ľ������
			delete DataOut[i];

		//Ϊ����GradeIn�ͷ���Ӧ�Ŀռ�
		for (int i = 0; i < num; i++) //���ľ������
			delete GradeIn[i];
	}

	/************************************************
	�������ƣ�Forward
	��    ��:CNN_ConActivation_Layer�ľ�����ļ��������ǰ�򴫲�
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_ConActivation_Layer<CNN_TYPE>::Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		for (unsigned int cnt = 0; cnt < DataIn.size(); cnt++)
			(*DataOut[cnt]) = (*DataIn[cnt]).transfer(Activate_Func);
	}

	/************************************************
	��������:Backward
	��    ��:CNN_ConActivation_Layer�ľ����󼤻����ķ��򴫲�
	��    ����
	��    �أ�
	˵    ����
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
	��������:API_Data_Forward
	��    ��:CNN_ConActivation_Layer ��ǰ����һ�㣩���ݵ�ǰ��Ľ��
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_ConActivation_Layer<CNN_TYPE>::API_Data_Forward()
	{
		return DataOut;
	}

	/************************************************
	��������:API_Grade_Backward
	��    ��:CNN_ConActivation_Layer ����ǰһ�㣩���ݵ�ǰ����ݶ���
	��    ����
	��    �أ�
	˵    ����
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

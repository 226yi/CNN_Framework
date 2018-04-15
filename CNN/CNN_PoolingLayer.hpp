#pragma once

/**************************
File Name:CNN_PoolingLayer.hpp
Author: MiaoMiaoYoung

��־��
�ڹ�����BP�����
��һ�Σ���һ�Σ���һ�Σ��������˾��������

2017��11��16�ս���
��Ҫ���ھ�������еĳػ��㣬Ӧ���ڳػ�����
�ػ�ʱ�ֱ�Ϊ���ػ���ƽ���ػ�����ͬ
���������ػ���ƽ���ػ��Ĺ���
���������ػ����򴫲��ݶ���Ĺ��̣�ƽ���ػ����򴫲��ݶ���Ĺ���

����ģ��û����֤����ȷ��

���������ģ���˵��

2017��11��17��
������֤�����ػ���ƽ���ػ����򴫲������򴫵ݵ���ȷ��
�������Ϊ��ģ��
****************************/

#include"Matrix.hpp"

namespace yDL
{

//#define CNN_TYPE double
//#define Check

	/************************************************

	���ܣ�ʵ�־��������ľ�����ļ������

	������

	-   const int num                  // ���롢����������ȣ����������ͼƬ����� - ���뱣��һ�£�
	-   const int In_size              // �������Ĵ�С������Ϊ����
	-   const int Out_size             // �������Ĵ�С������Ϊ����
	-   const int Pool_size            // �ػ��˵Ĵ�С
	-   const int Stride               // ���гػ�ʱ�����ߵĲ���

	˵����

	-   Forward_Max;                   // �ػ���ǰ�򴫲���ǰ��������ػ� (const std::vector<Matrix<CNN_TYPE>*>DataIn)
	-   Forward_Ave;                   // �ػ���ǰ�򴫲���ǰ�����ƽ���ػ� (const std::vector<Matrix<CNN_TYPE>*>DataIn)

	-   Backward_Max;                  // �ػ������ػ����򴫵ݣ����򴫵���� (const std::vector<Matrix<CNN_TYPE>*>DataIn,std::vector<Matrix<CNN_TYPE>*>GradeOut)
	-   Backward_Ave;                  // �ػ���ƽ���ػ����򴫵ݣ����򴫵���� (std::vector<Matrix<CNN_TYPE>*>GradeOut)

	//API
	-   API_Data_Forward;     // ��ǰ���������Ľ�� -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // ��󴫵ݵ�ǰ������ -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class CNN_TYPE>
	class CNN_Pooling_Layer
	{

	private:

		std::vector<Matrix<CNN_TYPE>*> DataOut;                  //�����������ǰ��������
		std::vector<Matrix<CNN_TYPE>*> GradeIn;                  //������ݶȣ���ǰ����ݶȣ�

		//����ĸ��� �����������㶼���ϸ���ȵģ���Ȼ�Ͳ�����
		int num;    //���롢��������ĸ���������ȣ�

		int Output_size;   //��������Ĵ�С
		int Input_size;    //��������Ĵ�С

		int Pooling_size;  //�ػ�����Ĵ�С
		int Stride;        //�ػ����ߵĲ���

		CNN_Pooling_Layer(const CNN_Pooling_Layer&) = delete; //��������и��ƹ��캯��������ѽ

	public:
		explicit CNN_Pooling_Layer(
			const int NUM,
			const int In_size, const int Out_size,
			const int Pool_size, const int stride
		); //���캯��

		~CNN_Pooling_Layer();

		void Forward_Max(const std::vector<Matrix<CNN_TYPE>*>DataIn);        //�������ǰ�򴫲������ػ�
		void Forward_Ave(const std::vector<Matrix<CNN_TYPE>*>DataIn);        //�������ǰ�򴫲���ƽ���ػ�
		void Backward_Max(const std::vector<Matrix<CNN_TYPE>*>DataIn,
			const std::vector<Matrix<CNN_TYPE>*>GradeOut);                   //������練�򴫵ݣ����ػ�
		void Backward_Ave(const std::vector<Matrix<CNN_TYPE>*>GradeOut);     //������練�򴫵ݣ�ƽ���ػ�

		const std::vector<Matrix<CNN_TYPE>*> API_Data_Forward();         //��ǰ������� - API
		const std::vector<Matrix<CNN_TYPE>*> API_Grade_Backward();       //���򴫵��ݶ� - API

#ifdef Check
		friend void test_PoolingLayer();
#endif

	};

	/************************************************

	�������ƣ�   CNN_Pooling_Layer
	��    ��:    CNN_Pooling_Layer�Ĺ��캯��
	��    ����
				 -  ����㡢�����ͼƬ����ȣ�������
				 -  �����ͼƬ�Ĵ�С����X��-  �����ͼƬ�Ĵ�С����X��
				 -  �ػ��˵Ĵ�С����X��    -  �ػ������ߵĲ���
	��    �أ�
	˵    ����
				 -  Ϊ���DataOut�������ݶ�GradeIn������Ӧ��С�Ŀռ�
				 -  DataOut\GradeIn ֻ�Ƿ���ռ䣬���޸����������������������ģ����ʵ��
				 -  �����ڳ������й����У����������ڴ�Ĳ��裬���Ч��
	*************************************************/
	template<class CNN_TYPE>
	CNN_Pooling_Layer<CNN_TYPE>::CNN_Pooling_Layer(
		const int NUM,
		const int In_size, const int Out_size,
		const int Pool_size, const int stride
	) :
		num{ NUM },
		Input_size{ In_size }, Output_size{ Out_size },
		Pooling_size{ Pool_size }, Stride{ stride }
	{
		//Ϊ���DataOut������Ӧ�Ŀռ�
		for (int i = 0; i < NUM; i++) //���ľ������
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(Out_size, Out_size);
			assert(temp != NULL);
			DataOut.push_back(temp);
		}

		//Ϊ����GradeIn������Ӧ�Ŀռ�
		for (int i = 0; i < NUM; i++) //���ľ������
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(In_size, In_size);
			assert(temp != NULL);
			GradeIn.push_back(temp);
		}

	}

	/************************************************
	�������ƣ�~CNN_Pooling_Layer
	��    ��:CNN_Pooling_Layer����������
	��    ����
	��    �أ�
	˵    ����Ϊ DataOut \ GradeIn �ͷ���Ӧ�ռ�
	*************************************************/
	template<class CNN_TYPE>
	CNN_Pooling_Layer<CNN_TYPE>::~CNN_Pooling_Layer()
	{
		//Ϊ���DataOut�ͷ���Ӧ�Ŀռ�
		for (int i = 0; i < num; i++) //���ľ������
			delete DataOut[i];

		//Ϊ����GradeIn�ͷ���Ӧ�Ŀռ�
		for (int i = 0; i < num; i++) //���ľ������
			delete GradeIn[i];
	}

	/************************************************
	�������ƣ�Forward_Max
	��    ��:CNN_Pooling_Layer�����ػ�������ǰ�򴫲�
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Pooling_Layer<CNN_TYPE>::Forward_Max(const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		for (unsigned int cnt = 0; cnt < DataIn.size(); cnt++)
			(*DataOut[cnt]) = Max_Pooling((*DataIn[cnt]), Pooling_size, Stride);
	}

	/************************************************
	�������ƣ�Forward_Ave
	��    ��:CNN_Pooling_Layer��ƽ���ػ�������ǰ�򴫲�
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Pooling_Layer<CNN_TYPE>::Forward_Ave(const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		for (unsigned int cnt = 0; cnt < DataIn.size(); cnt++)
			(*DataOut[cnt]) = Ave_Pooling((*DataIn[cnt]), Pooling_size, Stride);
	}

	/************************************************
	�������ƣ�Backward_Max
	��    ��:CNN_Pooling_Layer�ķ������ػ������򴫵����
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Pooling_Layer<CNN_TYPE>::Backward_Max(const std::vector<Matrix<CNN_TYPE>*>DataIn, const std::vector<Matrix<CNN_TYPE>*>GradeOut)
	{
		for (unsigned int cnt = 0; cnt < DataIn.size(); cnt++)
		{
			std::vector<std::vector<bool>>sign = Max_Pooling_Sign((*DataIn[cnt]), Pooling_size, Stride);

			(*GradeIn[cnt]).Initialize();

			int counter = 0;
			for (int i = 0; i < Input_size; i++)
				for (int j = 0; j < Input_size; j++)
					if (sign[i][j]) //���ػ����з��򴫵����
						(*GradeIn[cnt])[i][j] = (*GradeOut[cnt]).Get_Value(counter / Output_size, counter%Output_size);
		}
	}

	/************************************************
	�������ƣ�Backward_Max
	��    ��:CNN_Pooling_Layer�ķ������ػ������򴫵����
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Pooling_Layer<CNN_TYPE>::Backward_Ave(const std::vector<Matrix<CNN_TYPE>*>GradeOut)
	{
		Matrix<CNN_TYPE> kron(Pooling_size, Pooling_size);
		for (int i = 0; i < Pooling_size; i++)
			for (int j = 0; j < Pooling_size; j++)
				kron[i][j] = 1;

		double sq = (double)Pooling_size*(double)Pooling_size;
		for (int cnt = 0; cnt < num; cnt++)
		{
			(*GradeIn[cnt]) = Kronecker((*GradeOut[cnt]), kron);
			(*GradeIn[cnt]) *= 1.0 / sq;
		}
	}

	/************************************************
	��������:API_Data_Forward
	��    ��:CNN_Pooling_Layer ��ǰ����һ�㣩���ݵ�ǰ��Ľ��
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_Pooling_Layer<CNN_TYPE>::API_Data_Forward()
	{
		return DataOut;
	}

	/************************************************
	��������:API_Grade_Backward
	��    ��:CNN_Pooling_Layer ����ǰһ�㣩���ݵ�ǰ����ݶ���
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_Pooling_Layer<CNN_TYPE>::API_Grade_Backward()
	{
		return GradeIn;
	}

#ifdef Check

	/************************************************
	��������:
	��    ��:CNN_Convolution_Layer ����ǰһ�㣩���ݵ�ǰ����ݶ���
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	void test_PoolingLayer()
	{
		CNN_Pooling_Layer<double> test(1, 6, 3, 2, 2);

		srand((unsigned int)time(NULL) + rand());

		Matrix<double>test_data(6, 6);
		double test_data_[36];
		for (int cnt = 0; cnt < 36; cnt++)
			test_data_[cnt] = rand() % 36;
		test_data.assigment(test_data_, sizeof(test_data_));

		cout << "test_data" << std::endl << std::endl;
		test_data.show();
		

		std::vector<Matrix<double>*> test_dm;
		test_dm.push_back(&test_data);

		test.Forward_Max(test_dm);

		cout << std::endl << std::endl << "*******************" << std::endl << std::endl;
		cout << "MAX_Forward" << std::endl << std::endl;

		for (unsigned int i = 0; i < test.API_Data_Forward().size(); i++)
			(*(test.API_Data_Forward())[i]).show();

		test.Forward_Ave(test_dm);

		cout << std::endl << std::endl << "*******************" << std::endl << std::endl;
		cout << "AVE_Forward" << std::endl << std::endl;

		for (unsigned int i = 0; i < test.API_Data_Forward().size(); i++)
			(*(test.API_Data_Forward())[i]).show();

		Matrix<double>test_grade(3, 3);
		double test_grade_[9] = { 1,2,3,4,5,6,7,8,9 };
		test_grade.assigment(test_grade_, sizeof(test_grade_));

		std::vector<Matrix<double>*> test_gm;
		test_gm.push_back(&test_grade);

		test.Backward_Max(test_dm, test_gm);

		cout << std::endl << std::endl << "*******************" << std::endl << std::endl;
		cout << "MAX_BACKward" << std::endl << std::endl;

		for (unsigned int i = 0; i < test.API_Data_Forward().size(); i++)
			(*(test.API_Grade_Backward())[i]).show();

		test.Backward_Ave(test_gm);

		cout << std::endl << std::endl << "*******************" << std::endl << std::endl;
		cout << "AVE_BACKward" << std::endl << std::endl;

		for (unsigned int i = 0; i < test.API_Data_Forward().size(); i++)
			(*(test.API_Grade_Backward())[i]).show();
	}

#endif // Check

#undef Check

}

#undef Check

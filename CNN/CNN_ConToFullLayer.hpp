#pragma once

/**************************
File Name:CNN_ConToFullLayer.hpp
Author: MiaoMiaoYoung

��־��
�ڹ�����BP�����
��һ�Σ���һ�Σ���һ�Σ��������˾��������

2017��11��16�ս���
��Ҫ���ھ��������������ȡ�㣨�����������ӳ��㣨������·��֮������Ӳ�
��һ��Ϊȫ���Ӳ㣬���򴫲���ѭȫ���ӷ��򴫲�����
��δ����

2017��11��17�ո���
������������ȡ��������ӳ���Ĺ���
��ҪΪȫ���ӵĹ���

2017��11��25�ո���
ͨ��Check.cpp���н�һ�����
������Updata����ģ��ʹ��
�����˳�ʼ��ȱʡ��ʹ��
�����˸���Ȩ�صĲ���

2017��11��28�ո���
�������ļ�д�����ܣ�������ѵ��������Ա���
����ͨ��
****************************/

#include"Matrix.hpp"

//#define Check

#ifdef Check
extern void test_CNN_ConToFull_Layer(); //Check.cpp ʵ����֤�Ƿ���ȷ
#endif // Check


namespace yDL
{


	/************************************************

	���ܣ�ʵ�־���������������ȡ��������ӳ���Ĺ���,ȫ���Ӳ�

	������

	-   const int Input_num            // �����������
	-   const int In_size              // �������Ĵ�С������Ϊ����
	-   const int Output_num           // �������ֵ�ĸ��� ����Ԫ�ظ�����������������ʽ�����

	˵����

	-   Forward;            // ���ɲ�ǰ�򴫲������Ա仯                   (const std::vector<Matrix<CNN_TYPE>*>DataIn)
	-   Backward;           // ���ɲ㷴�򴫵�������ȫ�������Ա仯���� (std::vector<Matrix<CNN_TYPE>*>GradeOut)
	-   Updata;             // ���ɲ���µ�ǰ���Ȩ�ؼ���ֵ

	//API
	-   API_Data_Forward;     // ��ǰ���������Ľ�� -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // ��󴫵ݵ�ǰ������ -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class CNN_TYPE>
	class CNN_ConToFull_Layer
	{
	private:
		Matrix<CNN_TYPE>* Weight;                                //��ǰ���Ȩ��
		Matrix<CNN_TYPE>* Bias;                                  //��ǰ�����ֵ

		Matrix<CNN_TYPE>* DataOut;                               //�����������ǰ��������
		std::vector<Matrix<CNN_TYPE>*> GradeIn;                  //������ݶȣ���ǰ����ݶȣ�

		int Input_num;    //�������ĸ���������ȣ�
		int Output_num;   //�������ĸ���������ȣ�

		int Input_size;    //��������Ĵ�С

		CNN_ConToFull_Layer(const CNN_ConToFull_Layer&) = delete;

	public:

		explicit CNN_ConToFull_Layer(
			const int In_num, const int Out_num, const int In_size,
			const CNN_TYPE trans_func(const CNN_TYPE x) = NULL
		); //���캯��

		~CNN_ConToFull_Layer(); //��������

		void Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn);                     //�������ǰ�򴫲�
		void Backward(const Matrix<CNN_TYPE>* const GradeOut);                        //������練�򴫵�
		void Updata(const double LS, const Matrix<CNN_TYPE>* const GradeOut,
			const std::vector<Matrix<CNN_TYPE>*>DataIn);                              //����������˽���Ȩֵ����

		const Matrix<CNN_TYPE>* const API_Data_Forward();                      //��ǰ������� - API
		const std::vector<Matrix<CNN_TYPE>*> API_Grade_Backward();             //���򴫵��ݶ� - API

		void Save_Info(std::ofstream& fout); //����ǰ����д���ļ���

#ifdef Check
		friend 	void test_CNN_ConToFull();
		friend  void ::test_CNN_ConToFull_Layer();
#endif // Check


	};

	/************************************************

	�������ƣ�   CNN_ConToFull_Layer
	��    ��:    CNN_ConToFull_Layer�Ĺ��캯��
	��    ����
	             -  �����ͼƬ����ȣ������� -  �����ͼƬ����ȣ������� - �����ͼƬ�Ĵ�С����X��
	             -  ����˳�ʼ����������û����Ϊȫ����ʼ��Ϊ0��
	��    �أ�
	˵    ����
         	     -  ΪȨ��Weight����ֵBias���о���������DataIn_Column�����DataOut�������ݶ�GradeIn������Ӧ��С�Ŀռ�
	             -  trans_func���ڳ�ʼ��Kernel������ʵ�ֽ��������������ʹ�þ�̬���������������
	             -  DataOut ֻ�Ƿ���ռ䣬���޸����������������������ģ����ʵ��
	             -  GradeIn ���������˿ռ䣬Ϊ����Ϸ��򴫵ݵĲ������ڷ���ռ��ͬʱ��ȫ����ʼ��Ϊ0
	             -  �����ڳ������й����У����������ڴ�Ĳ��裬���Ч��
	*************************************************/
	template<class CNN_TYPE>
	CNN_ConToFull_Layer<CNN_TYPE>::CNN_ConToFull_Layer(
		const int In_num, const int Out_num, const int In_size,
		const CNN_TYPE trans_func(const CNN_TYPE x)
	) :
		Input_num{ In_num }, Output_num{ Out_num },Input_size{ In_size }
	{
		//ΪȨ��������Ӧ�Ŀռ�
		if (1)
		{
			int size = In_num*In_size*In_size;
			Weight = new(std::nothrow)Matrix<CNN_TYPE>(Out_num, size);
			assert(Weight != NULL);
			if (trans_func != NULL)
				(*Weight) = (*Weight).transfer(trans_func);
			else
				(*Weight).Initialize();
		}

		//Ϊ��ֵ������Ӧ�ռ�
		if (1)
		{
			Bias = new(std::nothrow)Matrix<CNN_TYPE>(Out_num,1);
			assert(Bias != NULL);
			(*Bias).Initialize();
		}

		//Ϊ���DataOut������Ӧ�Ŀռ�
		if (1)
		{
			DataOut = new(std::nothrow)Matrix<CNN_TYPE>(Out_num, 1); //���Ϊ�о���
			assert(DataOut != NULL);
		}

		//Ϊ����GradeIn������Ӧ�Ŀռ�
		for (int i = 0; i < In_num; i++) //���ľ������
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(In_size, In_size);
			assert(temp != NULL);
			(*temp).Initialize();
			GradeIn.push_back(temp);
		}
	}

	/************************************************
	�������ƣ�~CNN_ConToFull_Layer
	��    ��:CNN_ConToFull_Layer����������
	��    ����
	��    �أ�
	˵    ����Ϊ�����kernel \ DataOut \ GradeIn �ͷ���Ӧ�ռ�
	*************************************************/
	template<class CNN_TYPE>
	CNN_ConToFull_Layer<CNN_TYPE>::~CNN_ConToFull_Layer()
	{
		//ΪȨֵ�ͷ���Ӧ�Ŀռ�
		delete Weight;

		//Ϊ��ֵ�ͷ���Ӧ�Ŀռ�
		delete Bias;

		//Ϊ���DataOut�ͷ���Ӧ�Ŀռ�
		delete DataOut;

		//Ϊ����GradeIn�ͷ���Ӧ�Ŀռ�
		for (int i = 0; i < Input_num; i++) //���ľ������
			delete GradeIn[i];
	}

	/************************************************
	�������ƣ�Forward
	��    ��:CNN_ConToFull_Layer�ľ��������ǰ�򴫲�
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_ConToFull_Layer<CNN_TYPE>::Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		Matrix<CNN_TYPE> DataIn_Column(Input_num*Input_size*Input_size, 1);  //������ת��Ϊ�о���
		DataIn_Column = Column_Joint(DataIn);
		(*DataOut) = (*Weight)*DataIn_Column + (*Bias);
	}

	/************************************************
	�������ƣ�Backward
	��    ��:CNN_ConToFull_Layer�ķ��������������򴫵�
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_ConToFull_Layer<CNN_TYPE>::Backward(const Matrix<CNN_TYPE>* const GradeOut)
	{
		Matrix<CNN_TYPE> Grade(Input_num*Input_size*Input_size, 1);
		Grade = (*Weight)('T')*(*GradeOut); //���򴫲��ݶ�
		Column_Change(Grade, GradeIn);      //���ݶȷ�����ÿһ���������

	}

	/************************************************
	�������ƣ�Updata
	��    ��:CNN_ConToFull_Layer���µ�ǰ���ϵĲ���
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_ConToFull_Layer<CNN_TYPE>::Updata(const double LS, 
		const Matrix<CNN_TYPE>* const GradeOut, const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		Matrix<CNN_TYPE>temp(Output_num, Input_num*Input_size*Input_size);
		Matrix<CNN_TYPE> DataIn_Column(Input_num*Input_size*Input_size, 1);  //������ת��Ϊ�о���
		DataIn_Column = Column_Joint(DataIn);//���ַ�����Ȼ�˷�һ����Դ���������ڽ�����ģ����������Ƚ�

		temp = (*GradeOut)*DataIn_Column('T'); //�õ�Ȩ�ص�ƫ��

		//ofstream fout_test("check\\test_linear3_output.txt", ios::binary); //�����򿪱������ļ�
		//fout_test << "error:" << endl;
		//fout_test << temp;
		//fout_test << endl << endl << "Weight";
		//fout_test << (*Weight);
		//fout_test << endl << endl;
		//fout_test.close();

		//cout << (*Weight).Count_Ave() <<' '<< (temp*LS).Count_Ave();

		(*Weight) -= temp*LS;
		(*Bias) -= (*GradeOut)*LS;
	};

	/************************************************
	��������:API_Data_Forward
	��    ��:CNN_ConToFull_Layer ��ǰ����һ�㣩���ݵ�ǰ��Ľ��
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	const Matrix<CNN_TYPE>* const CNN_ConToFull_Layer<CNN_TYPE>::API_Data_Forward()
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
	template<class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_ConToFull_Layer<CNN_TYPE>::API_Grade_Backward()
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
	void CNN_ConToFull_Layer<CNN_TYPE>::Save_Info(std::ofstream&fout)
	{

		fout << "CNN_ConToFull_Layer" << std::endl;
		fout << "Input_num " << Input_num << std::endl;
		fout << "Output_num " << Output_num << std::endl;
		fout << "Input_size " << Input_size << std::endl;

		fout << std::endl;

		fout << "CNN_ConToFull_Weight" << std::endl;
		fout << (*Weight) << std::endl;
		fout << std::endl;

		fout << "CNN_ConToFull_Bias" << std::endl;
		fout << (*Bias) << std::endl;

		fout << std::endl << std::endl;

	}


#ifdef Check

	const double trans_1fun(const double x)
	{
		return 1;
	}
	void test_CNN_ConToFull()
	{
		CNN_ConToFull_Layer<double> test(2, 4, 3, trans_1fun);


		std::vector<Matrix<double>*>data;
		if (1)
		{
			Matrix<double>* a1 = new(std::nothrow)Matrix<double>(3, 3);
			assert(a1);
			double b1[9] = { 1,2,0,1,1,3,0,2,2 };
			(*a1).assigment(b1, sizeof(b1));

			Matrix<double>* a2 = new(std::nothrow)Matrix<double>(3, 3);
			assert(a2);
			double b2[9] = { 0,2,1,0,3,2,1,1,0 };
			(*a2).assigment(b2, sizeof(b2));

			data.push_back(a1);
			data.push_back(a2);
		}

		test.Forward(data);

		(*test.API_Data_Forward()).show();

		cout << std::endl << std::endl << "*************" << std::endl;

		Matrix<double> Grade(4, 1);
		double g[4] = { 1,2,3,4 };
		Grade.assigment(g, sizeof(g));
		test.Backward(&Grade);

		std::vector<Matrix<double>*>r = test.API_Grade_Backward();

		for (unsigned int i = 0; i < r.size(); i++)
			(*r[i]).show();
	}

#endif // Check 


#undef Check
}

#undef Check

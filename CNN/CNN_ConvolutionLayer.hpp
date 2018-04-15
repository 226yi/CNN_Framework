#pragma once

/**************************
File Name:CNN_ConvolutionLayer.hpp
Author: MiaoMiaoYoung

��־��
�ڹ�����BP�����
��һ�Σ���һ�Σ���һ�Σ��������˾��������

2017��11��15�ս���
��Ҫ���ھ�����������ʵ��
�ڱ�д��������У�ʹ�ú궨��ķ�ʽ��Ŀ��Ϊ�����ȡ���궨�������Ϊ��ģ��

�������Ĺ��캯�������������Ľ�����Ϊ�����ռ䣬
����������û�м���Ƿ���ȷ��û�м���Ƿ�����ڴ�й©������
�������������ڴ���䣬������ν��зֲ�ʽ����

2017��11��16�ո���
����˾������ǰ�򴫲��Ĺ���
����˾�����練�򴫵ݵĹ���
����˾���������˸��²����Ĺ���
����֤���Ϲ�������ȷ�ģ�������֤��

//ȱ����ֵ��

2017��11��17�ո���
�����˾����������е���ֵ
����������̼��������
û����֤��ֵ�ڴ��ݹ����е���ȷ��

2017��11��25�ո���
������Check.cpp����
��ֵ���������ڸ���

2017��11��28�ո���
������д���ļ�����
���ڱ��浱ǰѵ��ֵ������ͨ��

2017��12��5�ո���
�����˾�������������򴫲������²���ʱ��ֵ������
֮ǰ����ֵ�������ƫ��

2017��12��10�ո���
void CNN_Convolution_Layer<CNN_TYPE>::Backward(std::vector<Matrix<CNN_TYPE>*>GradeOut)��
��ǰ���ݶȽ��м���Ĺ�����
û�����³�ʼ���Ϳ�ʼ��ͣ��ʹ���ۼ� (*GradeIn[index]) += Convolution(&data, &kernel_back, Stride);
����ÿһ��ͼƬѵ�������ۼ�����һ�ε��ݶȣ����´��󰡣�����
�ش���©��������Ϊ���ڳ�ʼ����ʱ���ǽ���������ģ��ĳ�ʼ���ģ����Ծ͵������������ô��Ĵ���
����˵ʵ�������ִ���̫�����ˣ�����

ʹ��ǰ��ʼ��������ʹ��ǰ��ʼ����������Щ������̲ܽ���

�����һ�����ƵĴ���

****************************/

//#define Check

#ifdef Check
extern void test_CNN_ConvolutionLayer(); //Check.cpp ʵ����֤�Ƿ���ȷ
#endif // Check


#include"Matrix.hpp"
#include<string>

namespace yDL
{

	//#define CNN_TYPE double

	/************************************************

	���ܣ�ʵ�־��������ľ����

	������

	-   const int In_num               // ����������ȣ�������ͼƬ����ȣ�
	-   const int Out_num              // ����������ȣ������ͼƬ����ȣ�
	-   const int In_size              // �������Ĵ�С������Ϊ����
	-   const int Out_size             // �������Ĵ�С������Ϊ����
	-   const int Kernel_Matrix_size   // ����˵Ĵ�С��Ϊ���󣬾���˵�������������������ͬ������˵ĸ������������ĸ�����ͬ��
	-   const int Stride               // ����˽��о��ʱ�����ߵĲ���
	-   const T trans_func(const T x)  // ����˵ĳ�ʼ��������Ĭ��ΪNULL�������ⲿʵ�֣��������ֵ���Ծ�̬��������(���ڲ���)���и�ֵ

	˵����

	-   Forward;              // �����ǰ�򴫲���ǰ����о�� (const std::vector<Matrix<CNN_TYPE>*>DataIn)
	-   Backward;             // ����㷴�򴫵ݣ����򴫵���� (std::vector<Matrix<CNN_TYPE>*>GradeOut)
	-   Updata;               // ��������˽��в�������(const std::vector<Matrix<CNN_TYPE>*>DataIn,const std::vector<Matrix<CNN_TYPE>*>GradeOut)

	//API
	-   API_Data_Forward;     // ��ǰ���������Ľ�� -- const std::vector<Matrix<CNN_TYPE>*>
	-   API_Grade_Backward;   // ��󴫵ݵ�ǰ������ -- const std::vector<Matrix<CNN_TYPE>*>

	*************************************************/
	template<class CNN_TYPE>
	class CNN_Convolution_Layer
	{

	protected:
	public:
		std::vector<std::vector<Matrix<CNN_TYPE>*>> Kernel;      //��ǰ��ľ������
		std::vector<CNN_TYPE>Kernel_Bias;                        //��ǰ��������ֵ

		std::vector<Matrix<CNN_TYPE>*> DataOut;                  //�����������ǰ��������
		std::vector<Matrix<CNN_TYPE>*> GradeIn;                  //������ݶȣ���ǰ����ݶȣ�

		int Input_num;    //�������ĸ���������ȣ�
		int Output_num;   //�������ĸ���������ȣ�

		int Output_size;   //��������Ĵ�С
		int Input_size;    //��������Ĵ�С

		int Kernel_size;   //����˾���Ĵ�С
		int Stride;        //��������ߵĲ���

		CNN_Convolution_Layer(const CNN_Convolution_Layer&) = delete;       //����������Ҫ�и��ƹ��캯�����Ҿܾ�


	public:
		explicit CNN_Convolution_Layer(
			const int In_num, const int Out_num,
			const int In_size, const int Out_size,
			const int Kernel_Matrix_size, const int stride,
			const CNN_TYPE trans_func(const CNN_TYPE x) = NULL
		); //���캯��

		explicit CNN_Convolution_Layer(std::ifstream& fin); //���캯�� - ���ļ��ж�ȡ����


		~CNN_Convolution_Layer(); //��������

		void Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn);        //�������ǰ�򴫲�
		void Backward(const std::vector<Matrix<CNN_TYPE>*>GradeOut);     //������練�򴫵�
		void Updata(const double LS,
			const std::vector<Matrix<CNN_TYPE>*>DataIn,
			const std::vector<Matrix<CNN_TYPE>*>GradeOut);               //����������˽���Ȩֵ����

		const std::vector<Matrix<CNN_TYPE>*> API_Data_Forward();         //��ǰ������� - API
		const std::vector<Matrix<CNN_TYPE>*> API_Grade_Backward();       //���򴫵��ݶ� - API

		void Save_Info(std::ofstream& fout); //����ǰ����д���ļ���

#ifdef Check
		friend void test_convolution();
		friend void ::test_CNN_ConvolutionLayer();
#endif

	};


	/************************************************

	�������ƣ�   CNN_Convolution_Layer
	��    ��:    CNN_Convolution_Layer�Ĺ��캯��
	��    ����
	-  �����ͼƬ����ȣ������� -  �����ͼƬ����ȣ�������
	-  �����ͼƬ�Ĵ�С����X��-  �����ͼƬ�Ĵ�С����X��
	-  ����˵Ĵ�С����X��    -  ��������ߵĲ���
	-  ����˳�ʼ����������û����Ϊȫ����ʼ��Ϊ0��
	��    �أ�
	˵    ����
	-  Ϊ�����kernel�����DataOut�������ݶ�GradeIn������Ӧ��С�Ŀռ�
	-  trans_func���ڳ�ʼ��Kernel������ʵ�ֽ��������������ʹ�þ�̬���������������
	-  DataOut ֻ�Ƿ���ռ䣬���޸����������������������ģ����ʵ��
	-  GradeIn ���������˿ռ䣬Ϊ����Ϸ��򴫵ݵĲ������ڷ���ռ��ͬʱ��ȫ����ʼ��Ϊ0
	-  �����ڳ������й����У����������ڴ�Ĳ��裬���Ч��
	*************************************************/
	template<class CNN_TYPE>
	CNN_Convolution_Layer<CNN_TYPE>::CNN_Convolution_Layer(
		const int In_num, const int Out_num,
		const int In_size, const int Out_size,
		const int Kernel_Matrix_size, const int stride,
		const CNN_TYPE trans_func(const CNN_TYPE x)
	) :
		Input_num{ In_num }, Output_num{ Out_num },
		Input_size{ In_size }, Output_size{ Out_size },
		Kernel_size{ Kernel_Matrix_size }, Stride{ stride }
	{

		//Ϊ�����������Ӧ�Ŀռ�
		for (int i = 0; i < Out_num; i++) //���ľ������
		{
			std::vector<Matrix<CNN_TYPE>*>temp_kernels;
			for (int j = 0; j < In_num; j++) //�ڲ�ľ������
			{
				Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(Kernel_Matrix_size, Kernel_Matrix_size);
				assert(temp != NULL);

				if (trans_func != NULL)
					(*temp) = (*temp).transfer(trans_func);
				else
					(*temp).Initialize();//��ʼ��

				temp_kernels.push_back(temp);
			}

			Kernel.push_back(temp_kernels); //����ָ��
		}

		//Ϊ�������ֵ������Ӧ�ռ�
		for (int i = 0; i < Out_num; i++)
		{
			CNN_TYPE bias = 0; //��ֵ��ʼ��Ϊ0
			Kernel_Bias.push_back(bias);
		}

		//Ϊ���DataOut������Ӧ�Ŀռ�
		for (int i = 0; i < Out_num; i++) //���ľ������
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(Out_size, Out_size);
			assert(temp != NULL);
			DataOut.push_back(temp);
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
	�������ƣ�   CNN_Convolution_Layer
	��    ��:    CNN_Convolution_Layer�Ĺ��캯�� - ���ļ��ж�ȡ��ǰ������
	��    ����   �ļ�·��
	-  �����ͼƬ����ȣ������� -  �����ͼƬ����ȣ�������
	-  �����ͼƬ�Ĵ�С����X��-  �����ͼƬ�Ĵ�С����X��
	-  ����˵Ĵ�С����X��    -  ��������ߵĲ���
	-  ����˳�ʼ����������û����Ϊȫ����ʼ��Ϊ0��
	��    �أ�
	˵    ����
	-  Ϊ�����kernel�����DataOut�������ݶ�GradeIn������Ӧ��С�Ŀռ�
	-  trans_func���ڳ�ʼ��Kernel������ʵ�ֽ��������������ʹ�þ�̬���������������
	-  DataOut ֻ�Ƿ���ռ䣬���޸����������������������ģ����ʵ��
	-  GradeIn ���������˿ռ䣬Ϊ����Ϸ��򴫵ݵĲ������ڷ���ռ��ͬʱ��ȫ����ʼ��Ϊ0
	-  �����ڳ������й����У����������ڴ�Ĳ��裬���Ч��
	*************************************************/
	template<class CNN_TYPE>
	CNN_Convolution_Layer<CNN_TYPE>::CNN_Convolution_Layer(std::ifstream& fin)
	{

		//���в�У��
		if (1)
		{
			std::string Proofread_Layer = "";
			fin >> Proofread_Layer;
			assert(Proofread_Layer == "CNN_Convolution_Layer"); //У�� - �Ƿ��뵱ǰ������� - �д�����б���
		}

		int In_num = 0;
		int Out_num = 0;
		int In_size = 0;
		int Out_size = 0;
		int Kernel_Matrix_size = 0;
		int stride = 0;


		//����ò������Ϣ
		if (1)
		{
			std::string Check_inner = ""; //���λ

			fin >> Check_inner;
			assert(Check_inner == "Input_num");
			fin >> In_num;

			fin >> Check_inner;
			assert(Check_inner == "Output_num");
			fin >> Out_num;

			fin >> Check_inner;
			assert(Check_inner == "Input_size");
			fin >> In_size;

			fin >> Check_inner;
			assert(Check_inner == "Output_size");
			fin >> Out_size;

			fin >> Check_inner;
			assert(Check_inner == "Kernel_size");
			fin >> Kernel_Matrix_size;

			fin >> Check_inner;
			assert(Check_inner == "Stride_size");
			fin >> stride;

			//Input_num{ In_num }, Output_num{ Out_num },
			//Input_size{ In_size }, Output_size{ Out_size },
			//Kernel_size{ Kernel_Matrix_size }, Stride{ stride }

			Input_num = In_num;
			Output_num = Out_num;

			Input_size = In_size;
			Output_size = Out_size;

			Kernel_size = Kernel_Matrix_size;
			Stride = stride;

		}


		std::string Check = ""; //���λ
		fin >> Check;
		assert(Check == "CNN_Kernel_Weight");

		//Ϊ�����������Ӧ�Ŀռ� - ����ȡ����
		for (int i = 0; i < Out_num; i++) //���ľ������
		{
			std::vector<Matrix<CNN_TYPE>*>temp_kernels;
			for (int j = 0; j < In_num; j++) //�ڲ�ľ������
			{
				Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(Kernel_Matrix_size, Kernel_Matrix_size);
				assert(temp != NULL);

				fin >> (*temp);

				if (trans_func != NULL)
					(*temp) = (*temp).transfer(trans_func);
				else
					(*temp).Initialize();//��ʼ��

				temp_kernels.push_back(temp);
			}

			Kernel.push_back(temp_kernels); //����ָ��
		}

		fin >> Check;
		assert(Check == "CNN_Kernel_Bias");

		//Ϊ�������ֵ������Ӧ�ռ�
		for (int i = 0; i < Out_num; i++)
		{
			CNN_TYPE bias = 0; //��ֵ��ʼ��Ϊ0
			fin >> bias;
			Kernel_Bias.push_back(bias);
		}

		//Ϊ���DataOut������Ӧ�Ŀռ�
		for (int i = 0; i < Out_num; i++) //���ľ������
		{
			Matrix<CNN_TYPE>* temp = new(std::nothrow)Matrix<CNN_TYPE>(Out_size, Out_size);
			assert(temp != NULL);
			DataOut.push_back(temp);
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
	�������ƣ�~CNN_Convolution_Layer
	��    ��:CNN_Convolution_Layer����������
	��    ����
	��    �أ�
	˵    ����Ϊ�����kernel \ DataOut \ GradeIn �ͷ���Ӧ�ռ�
	*************************************************/
	template<class CNN_TYPE>
	CNN_Convolution_Layer<CNN_TYPE>::~CNN_Convolution_Layer()
	{
		//Ϊ������ͷ���Ӧ�Ŀռ�
		for (int i = 0; i < Output_num; i++) //���ľ������
			for (int j = 0; j < Input_num; j++) //�ڲ�ľ������
				delete Kernel[i][j];

		//Ϊ���DataOut�ͷ���Ӧ�Ŀռ�
		for (int i = 0; i < Output_num; i++) //���ľ������
			delete DataOut[i];

		//Ϊ����GradeIn�ͷ���Ӧ�Ŀռ�
		for (int i = 0; i < Input_num; i++) //���ľ������
			delete GradeIn[i];
	}

	/************************************************
	�������ƣ�Forward
	��    ��:CNN_Convolution_Layer�ľ��������ǰ�򴫲�
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Convolution_Layer<CNN_TYPE>::Forward(const std::vector<Matrix<CNN_TYPE>*>DataIn)
	{
		for (int cnt = 0; cnt < Output_num; cnt++)
			(*DataOut[cnt]) = Convolution(DataIn, Kernel[cnt], Stride) + (Kernel_Bias[cnt]);  //����ľ������
	}

	/************************************************
	�������ƣ�Backward
	��    ��:CNN_Convolution_Layer�ķ��������������򴫵�
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Convolution_Layer<CNN_TYPE>::Backward(std::vector<Matrix<CNN_TYPE>*>GradeOut)
	{
		//��GradeIn�����³�ʼ����������һ�ε��ݶȵĺۼ���
		for (int cnt = 0; cnt < Output_num; cnt++) //��ÿһ������ͼ���ݶȽ��б���
			for (int index = 0; index < Input_num; index++)
				(*GradeIn[index]).Initialize();

		for (int cnt = 0; cnt < Output_num; cnt++) //��ÿһ������ͼ���ݶȽ��б���
		{
			//����ָ�� - �����ָ�벢û������ռ䣬��Pading Rot���ɵľ��󣬻���ѭ���������Զ�������������������������delete

			const int Pad_Size = ((Input_size - 1)*Stride + Kernel_size - Output_size) / 2;
			Matrix<CNN_TYPE>data = (*GradeOut[cnt]).Pading(Pad_Size);

			for (int index = 0; index < Input_num; index++)
			{
				Matrix<CNN_TYPE> kernel_back = (*Kernel[cnt][index])('R');

				(*GradeIn[index]) += Convolution(&data, &kernel_back, Stride);

			}// for ��ÿһ����б��������л���

		}// for ��ÿһ������ͼ���ݶȽ��б���
	}

	/************************************************
	��������:Updata
	��    ��:CNN_Convolution_Layer ��ǰ��ľ����Ȩ�ظ���
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	void CNN_Convolution_Layer<CNN_TYPE>::Updata(const double LS,
		const std::vector<Matrix<CNN_TYPE>*>DataIn,
		const std::vector<Matrix<CNN_TYPE>*>GradeOut)
	{
		for (int cnt = 0; cnt < Output_num; cnt++)//for ��ÿһ�����ɵ�����ͼ����ÿһ������˽��б���
		{
			//�����Ȩֵ����

			Matrix<CNN_TYPE>* temp_kernel = GradeOut[cnt];  //ǣ���������ĵ�ǰ��

			for (int index = 0; index < Input_num; index++)  // for �Ծ�����е�ÿһ����б���
			{
				Matrix<CNN_TYPE> Error(Kernel_size, Kernel_size); //������

				Matrix<CNN_TYPE>* temp_data = DataIn[index];   //ǣ���������ĵ�ǰ��

				Error = Convolution(temp_data, temp_kernel, Stride);  //�õ���ǰ����˵ĵ�ǰ������

																	  //cout << endl << endl;
																	  //cout << "error" << endl;
																	  //Error.show();
																	  //cout << endl << endl;
																	  //(*Kernel[cnt][index]).show();

				//cout << (*Kernel[cnt][index]).Count_Ave() << " " << (LS*Error).Count_Ave()<<endl;

				(*Kernel[cnt][index]) -= LS*Error;   //����˽��и���

			} // for ��һ��������е�ÿһ����б������Ծ���˵�Ȩ�ؽ��и���


			  //�������ֵ����

			Kernel_Bias[cnt] -= LS*(*GradeOut[cnt]).Get_Determinant_Value();

		} //for ��ÿһ�����ɵ�����ͼ����ÿһ������˽��б���

	}

	/************************************************
	��������:API_Data_Forward
	��    ��:CNN_Convolution_Layer ��ǰ����һ�㣩���ݵ�ǰ��Ľ��
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	template<class CNN_TYPE>
	const std::vector<Matrix<CNN_TYPE>*> CNN_Convolution_Layer<CNN_TYPE>::API_Data_Forward()
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
	const std::vector<Matrix<CNN_TYPE>*> CNN_Convolution_Layer<CNN_TYPE>::API_Grade_Backward()
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
	void CNN_Convolution_Layer<CNN_TYPE>::Save_Info(std::ofstream&fout)
	{

		fout << "CNN_Convolution_Layer" << std::endl;
		fout << "Input_num " << Input_num << std::endl;
		fout << "Output_num " << Output_num << std::endl;
		fout << "Input_size " << Input_size << std::endl;
		fout << "Output_size " << Output_size << std::endl;
		fout << "Kernel_size " << Kernel_size << std::endl;
		fout << "Stride " << Stride << std::endl;
		fout << std::endl;


		fout << "CNN_Kernel_Weight" << std::endl;
		for (unsigned int cnt = 0; cnt < Kernel.size(); cnt++)
		{
			for (unsigned int index = 0; index < (Kernel[cnt]).size(); index++)
				fout << (*Kernel[cnt][index]) << std::endl;
			fout << std::endl;
		}

		fout << "CNN_Kernel_Bias" << std::endl;
		for (unsigned int cnt = 0; cnt < Kernel_Bias.size(); cnt++)
			fout << (Kernel_Bias[cnt]) << std::endl;

		fout << std::endl << std::endl;

	}

#undef Check
#ifdef Check

	using std::vector;

	const double trans_fun(const double x)
	{
		static double _array[24] = { 1,1,2,2,1,1,1,1,0,1,1,0,1,0,0,1,2,1,2,1,1,2,2,0 };
		static int i = -1;
		i++;
		return _array[i];
	}

	/************************************************
	��������:test_Convolution
	��    ��:���Ծ������ľ��������򴫲������򴫵ݣ������Ȩ�ظ���
	��    ����
	��    �أ�
	˵    ����
	*************************************************/
	void test_convolution()
	{
		CNN_Convolution_Layer<double> test(3, 2, 3, 2, 2, 1, trans_fun);

		cout << "Kernel:" << std::endl;
		vector<vector<Matrix<double>*>>k = test.Kernel;
		for (unsigned int i = 0; i < k.size(); i++)
			for (unsigned int j = 0; j < k[i].size(); j++)
				(*k[i][j]).show();

		cout << std::endl << "**********" << std::endl << std::endl;


		vector<Matrix<double>*>data;
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

			Matrix<double>* a3 = new(std::nothrow)Matrix<double>(3, 3);
			assert(a3);
			double b3[9] = { 1,2,1,0,1,3,3,3,2 };
			(*a3).assigment(b3, sizeof(b3));

			data.push_back(a1);
			data.push_back(a2);
			data.push_back(a3);
		}
		test.Forward(data);

		vector<Matrix<double>*>grade;
		if (1)
		{
			Matrix<double>* a1 = new(std::nothrow)Matrix<double>(2, 2);
			assert(a1);
			double b1[4] = { -1,-2,-3,-4 };
			(*a1).assigment(b1, sizeof(b1));

			Matrix<double>* a2 = new(std::nothrow)Matrix<double>(2, 2);
			assert(a2);
			double b2[4] = { -5,-6,-7,-8 };
			(*a2).assigment(b2, sizeof(b2));

			grade.push_back(a1);
			grade.push_back(a2);
		}
		test.Backward(grade);

		const vector<Matrix<double>*>result_data = test.API_Data_Forward(); //������

		cout << "DataOut:" << std::endl;
		for (unsigned int cnt = 0; cnt < result_data.size(); cnt++)
			(*result_data[cnt]).show();

		cout << std::endl << "**********" << std::endl << std::endl;

		const vector<Matrix<double>*>result_grade = test.API_Grade_Backward(); //������
		cout << "GradeIn:" << std::endl;
		for (unsigned int cnt = 0; cnt < result_grade.size(); cnt++)
			(*result_grade[cnt]).show();

		test.Updata(1.0, data, grade);

		cout << "Kernel:" << std::endl;
		vector<vector<Matrix<double>*>>k1 = test.Kernel;
		for (unsigned int i = 0; i < k1.size(); i++)
			for (unsigned int j = 0; j < k1[i].size(); j++)
				(*k1[i][j]).show();

		cout << std::endl << "**********" << std::endl << std::endl;


		test.Forward(data);
		const vector<Matrix<double>*>_result_data = test.API_Data_Forward();  //������

		cout << std::endl << "**********" << std::endl << std::endl;

		cout << "DataOut_Again:" << std::endl;
		for (unsigned int cnt = 0; cnt < _result_data.size(); cnt++)
			(*_result_data[cnt]).show();

		for (unsigned int i = 0; i < data.size(); i++)
			delete data[i];

		for (unsigned int i = 0; i < grade.size(); i++)
			delete grade[i];

	}


#endif

#undef Check

}

#undef Check
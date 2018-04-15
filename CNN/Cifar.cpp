/**************************
File Name:Cifar.cpp
Author: MiaoMiaoYoung

��־��
�ڹ�����BP�����
��һ�Σ���һ�Σ���һ�Σ��������˾��������

2017��11��18�ս���
�����˾��������Ļ������

2017��11��19�ո���
���ֵ�����ӿڵĴ���
����������Matrix������˱���

2017��11��25�ո���
���¼�������Է������������и���

2017��12��3�ո���
���¼�飬
���Բ���ʱ���֣���������ʱ����Ϊ��char�Ͷ��룬���Ի�����λ���ɷ���λ
������ֵ����125ʱ������ֵ����double�ǻ��Զ���ȥ256����Ϳӵ���
�����ڴ�ֵ��double����Ϊ(unsigned char)�� double = (unsigned char)

��Xavier����������һЩ����

2017��12��3��23:49����
������һЩ�����˵����⣡����
���ٸ��²�����ʱ��Ӧ���ǣ�Garde,Data���������д����(Grade,Grade)
ȫ�Ǹ����ǵĻ�������ȫ�Ǹ����ǵĻ�������
�Լ���Ц�Լ��أ�����������������������
�Ժ󣬸���ֻ�ܸ��ƿ�ܣ�����һ��һ��һ��һ��һ��Ҫ�Լ�д

��ʾ����ܵ����¼�飡����
ע������ļ�飬�����ǵļ�飬����һ��
****************************/


#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
//����ڴ��Ƿ�й©

#include"CNN_BottomLayer.hpp"
#include"CNN_ConvolutionLayer.hpp"
#include"CNN_ConActivationLayer.hpp"
#include"CNN_PoolingLayer.hpp"
#include"CNN_ConToFullLayer.hpp"
#include"CNN_FullActivation_Layer.hpp"
#include"CNN_FullLinearLayer.hpp"
#include"CNN_FullLossFuncLayer.hpp"

#include<random>
#include<string>

using namespace std;
using namespace yDL;

/************************************************
�������ƣ�
��    ��:Sigmoid����������䵼��
��    ����
��    �أ�
˵    ������Ϊ��������Matrix��ת����
*************************************************/
const double Sigmoid_Function(const double x)
{
	return 1.0 / (1.0 + exp(-x));
}
const double Sigmoid_Derivative(const double x)
{
	return Sigmoid_Function(x)*(1 - Sigmoid_Function(x));
}

/************************************************
�������ƣ�
��    ��:tanh����������䵼��
��    ����
��    �أ�
˵    ������Ϊ��������Matrix��ת����
*************************************************/
const double Tanh_Function(const double x)
{
	return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
}
const double Tanh_Derivative(const double x)
{
	return (1.0 - Tanh_Function(x)*Tanh_Function(x));
}

/************************************************
�������ƣ�
��    ��:SQRT����������䵼��
��    ����
��    �أ�
˵    ������Ϊ��������Matrix��ת����
*************************************************/ 
const double Sqrt_Function(const double x)
{
	if (x > 1e-20)
		return sqrt(x);
	else if (x < -1e-20)
		return -sqrt(-x);
	else
	{
		assert(1 == 0);
		return 1e-20;
	}
}
const double Sqrt_Derivative(const double x)
{
	if (x > 1e-20)
		return 1.0 / (2.0*sqrt(x));
	else if (x < -1e-20)
		return 1.0 / (2.0*sqrt(-x));
	else
	{
		assert(1 == 0);
		return 1e-20;
	}
}

/************************************************
�������ƣ�
��    ��:RELU����������䵼��
��    ����
��    �أ�
˵    ������Ϊ��������Matrix��ת����
*************************************************/
const double relu_Function(const double x)
{
	if (x > 0.0)
		return x;
	else
		return 0;
}
const double relu_Derivative(const double x)
{
	if (x > 0.0)
		return 1;
	else
		return 0;
}



void Read_Cifar_data(ifstream& in, double* data, int& label)
{
	char la = 0;
	in.read(&la, sizeof(char));
	label = (int)la;

	const int In_num = 3072;

	for (int i = 0; i < In_num; i++)
	{
		char ch = 0;
		in.read(&ch, sizeof(char));
		data[i] = (unsigned char)ch;
	}
}

//�ײ����� 3*32*32
const int bottom_layer_input_num = 3;
const int bottom_layer_input_size = 32;

//��һ�� �����-�ػ���
const int layer_1_con_output_num = 6;
const int layer_1_con_output_size = 28;
const int layer_1_con_kernel_size = 5;
const int layer_1_con_kernel_stride = 1; 

const int layer_1_pool_output_num = 6;
const int layer_1_pool_output_size = 14;
const int layer_1_pool_kernel_size = 2;
const int layer_1_pool_kernel_stride = 2;

//�ڶ��� �����-�ػ���
const int layer_2_con_output_num = 16;
const int layer_2_con_output_size = 10;
const int layer_2_con_kernel_size = 5;
const int layer_2_con_kernel_stride = 1;

const int layer_2_pool_output_num = 16;
const int layer_2_pool_output_size = 5;
const int layer_2_pool_kernel_size = 2;
const int layer_2_pool_kernel_stride = 2;

//������ ȫ���Ӳ㣨���ɣ�
const int layer_3_output_num = 120;

//���Ĳ� ȫ���Ӳ�
const int layer_4_output_num = 84;

//����� ȫ���Ӳ�
const int layer_5_output_num = 10;

/************************************************
�������ƣ�
��    ��:����������ת���㷨�����������min,max��֮��
��    ����
��    �أ�
˵    ����
-- uniform_real_distribution -- ���������ȷֲ�
-- uniform_int_distribution  -- �������ȷֲ�
*************************************************/
double uniform_rand(double min, double max)
{
	static std::mt19937 gen(rand());
	std::uniform_real_distribution<double> dst(min, max); //���������ȷֲ�
	return dst(gen);
}
const double layer_1_con_initialize_Xiaver(const double x)
{
	double max = sqrt(6.0 / (bottom_layer_input_size*bottom_layer_input_size));
	double min = -max;

	return uniform_rand(min, max);
}
const double layer_2_con_initialize_Xiaver(const double x)
{
	double max = sqrt(6.0 / (layer_1_pool_output_size*layer_1_pool_output_size));
	double min = -max;

	return uniform_rand(min, max);
}
const double layer_3_initialize_Xiaver(const double x)
{
	int input = layer_2_pool_output_size*layer_2_pool_output_size;

	double max = sqrt(6.0 / (input + layer_3_output_num));
	double min = -max;

	return uniform_rand(min, max);
}
const double layer_4_initialize_Xiaver(const double x)
{
	double max = sqrt(6.0 / (layer_3_output_num+ layer_4_output_num));
	double min = -max;

	return uniform_rand(min, max);
}
const double layer_5_initialize_Xiaver(const double x)
{
	double max = sqrt(6.0 / (layer_5_output_num+layer_4_output_num));
	double min = -max;

	return uniform_rand(min, max);
}

const double ActivationFunc(const double x)
{
	return relu_Function(x);
	//return Sqrt_Function(x);
	//return Tanh_Function(x);
}
const double ActivationDerivate(const double x)
{
	return relu_Derivative(x);
	//return Sqrt_Derivative(x);
	//return Tanh_Derivative(x);
}

void Cifar()
{
	srand(unsigned(time(NULL) + rand()));

	//��������ṹ

	CNN_Botton_Layer<double, double> Layer_bottom
	(
		bottom_layer_input_num,
		bottom_layer_input_size
	);//�ײ�װ�����ݲ�

	CNN_Convolution_Layer<double> Layer_1_con
	(
		bottom_layer_input_num, layer_1_con_output_num,
		bottom_layer_input_size, layer_1_con_output_size,
		layer_1_con_kernel_size, layer_1_con_kernel_stride,
		layer_1_con_initialize_Xiaver
	);// ��һ�� -- �����


	CNN_ConActivation_Layer<double> Layer_1_con_act
	(
		layer_1_con_output_num, layer_1_con_output_size,
		ActivationFunc, ActivationDerivate
	);// ��һ�� -- �������

	CNN_Pooling_Layer<double> Layer_1_pool
	(
		layer_1_pool_output_num, layer_1_con_output_size, layer_1_pool_output_size,
		layer_1_pool_kernel_size, layer_1_pool_kernel_stride
	);// ��һ�� -- �ػ���

	CNN_Convolution_Layer<double> Layer_2_con
	(
		layer_1_pool_output_num, layer_2_con_output_num,
		layer_1_pool_output_size, layer_2_con_output_size,
		layer_2_con_kernel_size, layer_2_con_kernel_stride,
		layer_2_con_initialize_Xiaver
	);// �ڶ��� -- �����

	CNN_ConActivation_Layer<double> Layer_2_con_act
	(
		layer_2_con_output_num, layer_2_con_output_size,
		ActivationFunc, ActivationDerivate
	);// �ڶ��� -- �������

	CNN_Pooling_Layer<double> Layer_2_pool
	(
		layer_2_pool_output_num, layer_2_con_output_size, layer_2_pool_output_size,
		layer_2_pool_kernel_size, layer_2_pool_kernel_stride
	);// �ڶ��� -- �ػ���

	CNN_ConToFull_Layer<double> Layer_3_linear
	(
		layer_2_pool_output_num, layer_3_output_num, layer_2_pool_output_size,
		layer_3_initialize_Xiaver
	);// ������ -- ȫ���Ӳ㣨���ɣ� - ���Է���

	CNN_FullActivation_Layer<double> Layer_3_activation
	(
		layer_3_output_num, ActivationFunc, ActivationDerivate
	);// ������ -- ȫ���Ӳ㣨���ɣ� - �����

	CNN_FullLinear_Layer<double> Layer_4_linear
	(
		layer_3_output_num, layer_4_output_num,
		layer_4_initialize_Xiaver
	);// ���Ĳ� -- ȫ���Ӳ� - ���Է���

	CNN_FullActivation_Layer<double> Layer_4_activation
	(
		layer_4_output_num, ActivationFunc, ActivationDerivate
	);// ���Ĳ� -- ȫ���Ӳ� - �����

	CNN_FullLinear_Layer<double> Layer_5_linear
	(
		layer_4_output_num, layer_5_output_num,
		layer_5_initialize_Xiaver
	);// ����� -- ȫ���Ӳ� - ���Է���

	CNN_FullActivation_Layer<double> Layer_5_activation
	(
		layer_5_output_num, ActivationFunc, ActivationDerivate
	);// ����� -- ȫ���Ӳ� - �����

	CNN_FullLossFunc_Softmax_Layer<double> Layer_top
	(
		layer_5_output_num
	);// ���� -- ������ʧ


	//׼������
	string database[5] = {
		"data_batch_1.bin","data_batch_2.bin",
		"data_batch_3.bin","data_batch_4.bin",
		"data_batch_5.bin"
	}; //���ݼ�����

	string input_mouse;
	cin >> input_mouse;
	cout << endl;

	string FoutFolder_Path = "SaveInfo//";
	string FoutFile_Result = "result_Relu_"+input_mouse;
	string FoutFile_SaveData = "SaveData_Relu_" + input_mouse;
	string FoutFile_suffix = ".data";

	ifstream fin_train; //����ѵ�����ļ�
	ifstream fin_check; //����������֤����֤���ȴ��ļ�
	ofstream fout_SaveInfo;   //�����򿪱��������ļ�
	ofstream fout_testResult; //�����򿪱������ļ�

	//�򿪱������ļ������浱ǰ���ȣ�
	fout_testResult.open(FoutFolder_Path + FoutFile_Result + FoutFile_suffix);
	assert(fout_testResult.is_open());

	double LS_layer_1 = 1e-3;
	double LS_layer_2 = 1e-3;
	double LS_layer_3 = 1e-5;
	double LS_layer_4 = 1e-5;
	double LS_layer_5 = 1e-5;

	int train_photo_num = 1000; //ÿһ�ε���ѵ��ͼƬ�ĸ���
	int check_photo_num = 1000; //ѵ�������ʱͼƬ�ĸ���

	//��ʼѵ��
	cout << "Learning Relu train"<<input_mouse << endl;

	for (int interation = 0;; interation++) // ����ѵ��
	{
		cout << "����ѵ��" << interation << "��  ";

        //������֤����ѵ��
		if (1)
		{
			fin_train.open(database[0], ios::binary); //�򿪵�circle�����ݼ�����ѵ��
			assert(fin_train.is_open());      //û�򿪱���

			for (int cnt = 0; cnt < train_photo_num; cnt++)
			{
				//����װ�� - Ԥ����
				int label = -1;
				if (1)
				{
					int num = bottom_layer_input_size*bottom_layer_input_size*bottom_layer_input_num;
					double* data = new(nothrow)double[num];
					Read_Cifar_data(fin_train, data, label);

					assert(label != -1); //������󱨾�

					Layer_bottom.Load_Data(data);
					Layer_bottom.Zero_Center();        //�����Ļ� - Ԥ����
					delete[] data;
				}


				//���򴫵�

				//��һ��
				Layer_1_con.Forward(Layer_bottom.API_Data_Forward());           //���
				Layer_1_con_act.Forward(Layer_1_con.API_Data_Forward());        //����
				Layer_1_pool.Forward_Ave(Layer_1_con_act.API_Data_Forward());   //�ػ�

				//�ڶ���
				Layer_2_con.Forward(Layer_1_pool.API_Data_Forward());           //���
				Layer_2_con_act.Forward(Layer_2_con.API_Data_Forward());        //����
				Layer_2_pool.Forward_Ave(Layer_2_con_act.API_Data_Forward());   //�ػ�

				//������
				Layer_3_linear.Forward(Layer_2_pool.API_Data_Forward());        //����
				Layer_3_activation.Forward(Layer_3_linear.API_Data_Forward());  //����

				//���Ĳ�
				Layer_4_linear.Forward(Layer_3_activation.API_Data_Forward());  //����
				Layer_4_activation.Forward(Layer_4_linear.API_Data_Forward());  //����

				//�����
				Layer_5_linear.Forward(Layer_4_activation.API_Data_Forward());  //����
				Layer_5_activation.Forward(Layer_5_linear.API_Data_Forward());  //����

				//������ʧֵ
				Layer_top.Cal_Score(Layer_5_activation.API_Data_Forward());     //����
				Layer_top.Cal_Loss(label);                                      //��ʧ
				Layer_top.Cal_Grade(label);                                     //�ݶ�

				//���򴫵�

				//�����
				Layer_5_activation.Backward(Layer_top.API_Grade_Backward(), Layer_5_linear.API_Data_Forward());
				Layer_5_linear.Backward(Layer_5_activation.API_Grade_Backward());

				//���Ĳ�
				Layer_4_activation.Backward(Layer_5_linear.API_Grade_Backward(), Layer_4_linear.API_Data_Forward());
				Layer_4_linear.Backward(Layer_4_activation.API_Grade_Backward());

				//������
				Layer_3_activation.Backward(Layer_4_linear.API_Grade_Backward(), Layer_3_linear.API_Data_Forward());
				Layer_3_linear.Backward(Layer_3_activation.API_Grade_Backward());

				//�ڶ���
				Layer_2_pool.Backward_Ave(Layer_3_linear.API_Grade_Backward());
				Layer_2_con_act.Backward(Layer_2_con.API_Data_Forward(), Layer_2_pool.API_Grade_Backward());
				Layer_2_con.Backward(Layer_2_con_act.API_Grade_Backward());

				//��һ��
				Layer_1_pool.Backward_Ave(Layer_2_con.API_Grade_Backward());
				Layer_1_con_act.Backward(Layer_1_con.API_Data_Forward(), Layer_1_pool.API_Grade_Backward());
				Layer_1_con.Backward(Layer_1_con_act.API_Grade_Backward());

				//���²���
				Layer_1_con.Updata(LS_layer_1, Layer_bottom.API_Data_Forward(), Layer_1_con_act.API_Grade_Backward());                 //���
				Layer_2_con.Updata(LS_layer_2, Layer_1_pool.API_Data_Forward(), Layer_2_con_act.API_Grade_Backward());                 //���

				Layer_3_linear.Updata(LS_layer_3, Layer_3_activation.API_Grade_Backward(), Layer_2_pool.API_Data_Forward());                                            //���� - ����
				Layer_4_linear.Updata(LS_layer_4, Layer_4_activation.API_Grade_Backward(), Layer_3_activation.API_Data_Forward());   //����
				Layer_5_linear.Updata(LS_layer_5, Layer_5_activation.API_Grade_Backward(), Layer_4_activation.API_Data_Forward());   //����

			}// for - train_photo_num

			fin_train.close(); //�رյ�ǰ��ѵ����ѵ����

		}

		 //���鵱ǰ���Ȳ�����
		if (1)
		{
			fin_check.open(database[0], ios::binary); //�򿪲������ݼ�
			assert(fin_check.is_open());     //�ļ��򿪴��󱨾�
			 
			int right_num = 0;

			//��ʼ����
			for (int cnt = 0; cnt < check_photo_num; cnt++)
			{
				//����װ�� - Ԥ����
				int label = 0;
				if (1)
				{
					int num = bottom_layer_input_size*bottom_layer_input_size*bottom_layer_input_num;
					double* data = new(nothrow)double[num];
					Read_Cifar_data(fin_check, data, label);

					assert(label != -1);

					Layer_bottom.Load_Data(data);
					Layer_bottom.Zero_Center();        //�����Ļ� - Ԥ����
					delete[] data;
				}

				//���򴫵�

				//��һ��
				Layer_1_con.Forward(Layer_bottom.API_Data_Forward());           //���
				Layer_1_con_act.Forward(Layer_1_con.API_Data_Forward());        //����
				Layer_1_pool.Forward_Ave(Layer_1_con_act.API_Data_Forward());   //�ػ�

																				//�ڶ���
				Layer_2_con.Forward(Layer_1_pool.API_Data_Forward());           //���
				Layer_2_con_act.Forward(Layer_2_con.API_Data_Forward());        //����
				Layer_2_pool.Forward_Ave(Layer_2_con_act.API_Data_Forward());   //�ػ�

																				//������
				Layer_3_linear.Forward(Layer_2_pool.API_Data_Forward());        //����
				Layer_3_activation.Forward(Layer_3_linear.API_Data_Forward());  //����

																				//���Ĳ�
				Layer_4_linear.Forward(Layer_3_activation.API_Data_Forward());  //����
				Layer_4_activation.Forward(Layer_4_linear.API_Data_Forward());  //����

																				//�����
				Layer_5_linear.Forward(Layer_4_activation.API_Data_Forward());  //����
				Layer_5_activation.Forward(Layer_5_linear.API_Data_Forward());  //����

																				//������ʧֵ
				Layer_top.Cal_Score(Layer_5_activation.API_Data_Forward());     //����
				Layer_top.Cal_Loss(label);                                      //��ʧ
				Layer_top.Cal_Grade(label);                                     //�ݶ�

				if (Layer_top.return_Label() == label)
					right_num++;
			}// for - check_photo_num

			fin_check.close(); //�رյ�ǰ�����ļ�

			double Accuracy = (double)right_num / (double)check_photo_num; //���㵱ǰ����

			fout_testResult << "interation: " << interation << " Accuracy: " << Accuracy << std::endl; //������д�����ļ� 
			cout << "interation: " << interation << " Accuracy: " << Accuracy << std::endl;
			//Ψһһ����ʼʱ�򿪣�����ʱ�رյ��ļ���
		}

		//���浱ǰ�������ֵ
		if (interation % 10 == 0 && interation != 0)
		{
			//���������ļ�·��
			string SaveInfo_path = FoutFolder_Path + FoutFile_SaveData + "_" + to_string(interation) + FoutFile_suffix;
			fout_SaveInfo.open(SaveInfo_path, ios::binary); //�򿪴����ļ�
			assert(fout_SaveInfo.is_open());

			fout_SaveInfo << "Layer_1_con" << std::endl << std::endl;
			Layer_1_con.Save_Info(fout_SaveInfo);
			fout_SaveInfo << "Layer_2_con" << std::endl << std::endl;
			Layer_2_con.Save_Info(fout_SaveInfo);
			fout_SaveInfo << "Layer_3_contolinear" << std::endl << std::endl;
			Layer_3_linear.Save_Info(fout_SaveInfo);
			fout_SaveInfo << "Layer_4_linear" << std::endl << std::endl;
			Layer_4_linear.Save_Info(fout_SaveInfo);
			fout_SaveInfo << "Layer_5_linear" << std::endl << std::endl;
			Layer_5_linear.Save_Info(fout_SaveInfo);

			fout_SaveInfo.close(); //�ͷŵ�ǰ�ļ�
		}

		//���������
		//if (interation != 0 && interation % 10 == 0)
		//{
		//	LS_layer_1 /= 3;
		//	LS_layer_2 /= 3;
		//	LS_layer_3 /= 3;
		//	LS_layer_4 /= 3;
		//	LS_layer_5 /= 3;
		//}


	}// for - iteration

	//�ƺ���

	fout_testResult.close();
}


int main()
{
	Cifar();

	_CrtDumpMemoryLeaks();

	return 0;
}
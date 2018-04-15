/**************************
File Name:Cifar.cpp
Author: MiaoMiaoYoung

日志：
在构建好BP网络后
再一次，再一次，再一次，继续攻克卷积神经网络

2017年11月18日建立
建立了卷积神经网络的基本框架

2017年11月19日更新
发现第三层接口的错误
改正，并在Matrix中添加了报警

2017年11月25日更新
重新检查了线性分类器，并进行更正

2017年12月3日更新
重新检查，
调试参数时发现：读入数据时，因为用char型读入，所以会把最高位当成符号位
所以数值超过125时，该数值传到double是会自动减去256，这就坑爹了
于是在传值给double是设为(unsigned char)即 double = (unsigned char)

对Xavier方法进行了一些更改

2017年12月3日23:49更新
发现了一些不得了的问题！！！
你再更新参数的时候，应该是（Garde,Data），结果你写成了(Grade,Grade)
全是复制惹的祸！！！全是复制惹的祸！！！
自己还笑自己呢！！！哈哈哈哈哈哈哈啊哈
以后，复制只能复制框架，参数一定一定一定一定一定要自己写

启示：框架的重新检查！！！
注意参数的检查，无死角的检查，再来一遍
****************************/


#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
//检查内存是否泄漏

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
函数名称：
功    能:Sigmoid激活函数，及其导数
参    数：
返    回：
说    明：作为参数带入Matrix的转换中
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
函数名称：
功    能:tanh激活函数，及其导数
参    数：
返    回：
说    明：作为参数带入Matrix的转换中
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
函数名称：
功    能:SQRT激活函数，及其导数
参    数：
返    回：
说    明：作为参数带入Matrix的转换中
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
函数名称：
功    能:RELU激活函数，及其导数
参    数：
返    回：
说    明：作为参数带入Matrix的转换中
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

//底层输入 3*32*32
const int bottom_layer_input_num = 3;
const int bottom_layer_input_size = 32;

//第一层 卷积层-池化层
const int layer_1_con_output_num = 6;
const int layer_1_con_output_size = 28;
const int layer_1_con_kernel_size = 5;
const int layer_1_con_kernel_stride = 1; 

const int layer_1_pool_output_num = 6;
const int layer_1_pool_output_size = 14;
const int layer_1_pool_kernel_size = 2;
const int layer_1_pool_kernel_stride = 2;

//第二层 卷积层-池化层
const int layer_2_con_output_num = 16;
const int layer_2_con_output_size = 10;
const int layer_2_con_kernel_size = 5;
const int layer_2_con_kernel_stride = 1;

const int layer_2_pool_output_num = 16;
const int layer_2_pool_output_size = 5;
const int layer_2_pool_kernel_size = 2;
const int layer_2_pool_kernel_stride = 2;

//第三层 全连接层（过渡）
const int layer_3_output_num = 120;

//第四层 全连接层
const int layer_4_output_num = 84;

//第五层 全连接层
const int layer_5_output_num = 10;

/************************************************
函数名称：
功    能:马特赛特旋转演算法生成随机数（min,max）之间
参    数：
返    回：
说    明：
-- uniform_real_distribution -- 浮点数均匀分布
-- uniform_int_distribution  -- 整数均匀分布
*************************************************/
double uniform_rand(double min, double max)
{
	static std::mt19937 gen(rand());
	std::uniform_real_distribution<double> dst(min, max); //浮点数均匀分布
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

	//建立网络结构

	CNN_Botton_Layer<double, double> Layer_bottom
	(
		bottom_layer_input_num,
		bottom_layer_input_size
	);//底层装载数据层

	CNN_Convolution_Layer<double> Layer_1_con
	(
		bottom_layer_input_num, layer_1_con_output_num,
		bottom_layer_input_size, layer_1_con_output_size,
		layer_1_con_kernel_size, layer_1_con_kernel_stride,
		layer_1_con_initialize_Xiaver
	);// 第一层 -- 卷积层


	CNN_ConActivation_Layer<double> Layer_1_con_act
	(
		layer_1_con_output_num, layer_1_con_output_size,
		ActivationFunc, ActivationDerivate
	);// 第一层 -- 激活函数层

	CNN_Pooling_Layer<double> Layer_1_pool
	(
		layer_1_pool_output_num, layer_1_con_output_size, layer_1_pool_output_size,
		layer_1_pool_kernel_size, layer_1_pool_kernel_stride
	);// 第一层 -- 池化层

	CNN_Convolution_Layer<double> Layer_2_con
	(
		layer_1_pool_output_num, layer_2_con_output_num,
		layer_1_pool_output_size, layer_2_con_output_size,
		layer_2_con_kernel_size, layer_2_con_kernel_stride,
		layer_2_con_initialize_Xiaver
	);// 第二层 -- 卷积层

	CNN_ConActivation_Layer<double> Layer_2_con_act
	(
		layer_2_con_output_num, layer_2_con_output_size,
		ActivationFunc, ActivationDerivate
	);// 第二层 -- 激活函数层

	CNN_Pooling_Layer<double> Layer_2_pool
	(
		layer_2_pool_output_num, layer_2_con_output_size, layer_2_pool_output_size,
		layer_2_pool_kernel_size, layer_2_pool_kernel_stride
	);// 第二层 -- 池化层

	CNN_ConToFull_Layer<double> Layer_3_linear
	(
		layer_2_pool_output_num, layer_3_output_num, layer_2_pool_output_size,
		layer_3_initialize_Xiaver
	);// 第三层 -- 全连接层（过渡） - 线性分类

	CNN_FullActivation_Layer<double> Layer_3_activation
	(
		layer_3_output_num, ActivationFunc, ActivationDerivate
	);// 第三层 -- 全连接层（过渡） - 激活函数

	CNN_FullLinear_Layer<double> Layer_4_linear
	(
		layer_3_output_num, layer_4_output_num,
		layer_4_initialize_Xiaver
	);// 第四层 -- 全连接层 - 线性分类

	CNN_FullActivation_Layer<double> Layer_4_activation
	(
		layer_4_output_num, ActivationFunc, ActivationDerivate
	);// 第四层 -- 全连接层 - 激活函数

	CNN_FullLinear_Layer<double> Layer_5_linear
	(
		layer_4_output_num, layer_5_output_num,
		layer_5_initialize_Xiaver
	);// 第五层 -- 全连接层 - 线性分类

	CNN_FullActivation_Layer<double> Layer_5_activation
	(
		layer_5_output_num, ActivationFunc, ActivationDerivate
	);// 第五层 -- 全连接层 - 激活函数

	CNN_FullLossFunc_Softmax_Layer<double> Layer_top
	(
		layer_5_output_num
	);// 顶层 -- 计算损失


	//准备工作
	string database[5] = {
		"data_batch_1.bin","data_batch_2.bin",
		"data_batch_3.bin","data_batch_4.bin",
		"data_batch_5.bin"
	}; //数据集名称

	string input_mouse;
	cin >> input_mouse;
	cout << endl;

	string FoutFolder_Path = "SaveInfo//";
	string FoutFile_Result = "result_Relu_"+input_mouse;
	string FoutFile_SaveData = "SaveData_Relu_" + input_mouse;
	string FoutFile_suffix = ".data";

	ifstream fin_train; //声明训练打开文件
	ifstream fin_check; //声明交叉验证，验证精度打开文件
	ofstream fout_SaveInfo;   //声明打开保存数据文件
	ofstream fout_testResult; //声明打开保存结果文件

	//打开保存结果文件（保存当前精度）
	fout_testResult.open(FoutFolder_Path + FoutFile_Result + FoutFile_suffix);
	assert(fout_testResult.is_open());

	double LS_layer_1 = 1e-3;
	double LS_layer_2 = 1e-3;
	double LS_layer_3 = 1e-5;
	double LS_layer_4 = 1e-5;
	double LS_layer_5 = 1e-5;

	int train_photo_num = 1000; //每一次迭代训练图片的个数
	int check_photo_num = 1000; //训练后检验时图片的个数

	//开始训练
	cout << "Learning Relu train"<<input_mouse << endl;

	for (int interation = 0;; interation++) // 迭代训练
	{
		cout << "迭代训练" << interation << "次  ";

        //交叉验证进行训练
		if (1)
		{
			fin_train.open(database[0], ios::binary); //打开第circle个数据集进行训练
			assert(fin_train.is_open());      //没打开报警

			for (int cnt = 0; cnt < train_photo_num; cnt++)
			{
				//数据装填 - 预处理
				int label = -1;
				if (1)
				{
					int num = bottom_layer_input_size*bottom_layer_input_size*bottom_layer_input_num;
					double* data = new(nothrow)double[num];
					Read_Cifar_data(fin_train, data, label);

					assert(label != -1); //输入错误报警

					Layer_bottom.Load_Data(data);
					Layer_bottom.Zero_Center();        //零中心化 - 预处理
					delete[] data;
				}


				//正向传递

				//第一层
				Layer_1_con.Forward(Layer_bottom.API_Data_Forward());           //卷积
				Layer_1_con_act.Forward(Layer_1_con.API_Data_Forward());        //激活
				Layer_1_pool.Forward_Ave(Layer_1_con_act.API_Data_Forward());   //池化

				//第二层
				Layer_2_con.Forward(Layer_1_pool.API_Data_Forward());           //卷积
				Layer_2_con_act.Forward(Layer_2_con.API_Data_Forward());        //激活
				Layer_2_pool.Forward_Ave(Layer_2_con_act.API_Data_Forward());   //池化

				//第三层
				Layer_3_linear.Forward(Layer_2_pool.API_Data_Forward());        //线性
				Layer_3_activation.Forward(Layer_3_linear.API_Data_Forward());  //激活

				//第四层
				Layer_4_linear.Forward(Layer_3_activation.API_Data_Forward());  //线性
				Layer_4_activation.Forward(Layer_4_linear.API_Data_Forward());  //激活

				//第五层
				Layer_5_linear.Forward(Layer_4_activation.API_Data_Forward());  //线性
				Layer_5_activation.Forward(Layer_5_linear.API_Data_Forward());  //激活

				//计算损失值
				Layer_top.Cal_Score(Layer_5_activation.API_Data_Forward());     //评分
				Layer_top.Cal_Loss(label);                                      //损失
				Layer_top.Cal_Grade(label);                                     //梯度

				//反向传递

				//第五层
				Layer_5_activation.Backward(Layer_top.API_Grade_Backward(), Layer_5_linear.API_Data_Forward());
				Layer_5_linear.Backward(Layer_5_activation.API_Grade_Backward());

				//第四层
				Layer_4_activation.Backward(Layer_5_linear.API_Grade_Backward(), Layer_4_linear.API_Data_Forward());
				Layer_4_linear.Backward(Layer_4_activation.API_Grade_Backward());

				//第三层
				Layer_3_activation.Backward(Layer_4_linear.API_Grade_Backward(), Layer_3_linear.API_Data_Forward());
				Layer_3_linear.Backward(Layer_3_activation.API_Grade_Backward());

				//第二层
				Layer_2_pool.Backward_Ave(Layer_3_linear.API_Grade_Backward());
				Layer_2_con_act.Backward(Layer_2_con.API_Data_Forward(), Layer_2_pool.API_Grade_Backward());
				Layer_2_con.Backward(Layer_2_con_act.API_Grade_Backward());

				//第一层
				Layer_1_pool.Backward_Ave(Layer_2_con.API_Grade_Backward());
				Layer_1_con_act.Backward(Layer_1_con.API_Data_Forward(), Layer_1_pool.API_Grade_Backward());
				Layer_1_con.Backward(Layer_1_con_act.API_Grade_Backward());

				//更新参数
				Layer_1_con.Updata(LS_layer_1, Layer_bottom.API_Data_Forward(), Layer_1_con_act.API_Grade_Backward());                 //卷积
				Layer_2_con.Updata(LS_layer_2, Layer_1_pool.API_Data_Forward(), Layer_2_con_act.API_Grade_Backward());                 //卷积

				Layer_3_linear.Updata(LS_layer_3, Layer_3_activation.API_Grade_Backward(), Layer_2_pool.API_Data_Forward());                                            //线性 - 过渡
				Layer_4_linear.Updata(LS_layer_4, Layer_4_activation.API_Grade_Backward(), Layer_3_activation.API_Data_Forward());   //线性
				Layer_5_linear.Updata(LS_layer_5, Layer_5_activation.API_Grade_Backward(), Layer_4_activation.API_Data_Forward());   //线性

			}// for - train_photo_num

			fin_train.close(); //关闭当前打开训练的训练集

		}

		 //检验当前精度并储存
		if (1)
		{
			fin_check.open(database[0], ios::binary); //打开测试数据集
			assert(fin_check.is_open());     //文件打开错误报警
			 
			int right_num = 0;

			//开始测试
			for (int cnt = 0; cnt < check_photo_num; cnt++)
			{
				//数据装填 - 预处理
				int label = 0;
				if (1)
				{
					int num = bottom_layer_input_size*bottom_layer_input_size*bottom_layer_input_num;
					double* data = new(nothrow)double[num];
					Read_Cifar_data(fin_check, data, label);

					assert(label != -1);

					Layer_bottom.Load_Data(data);
					Layer_bottom.Zero_Center();        //零中心化 - 预处理
					delete[] data;
				}

				//正向传递

				//第一层
				Layer_1_con.Forward(Layer_bottom.API_Data_Forward());           //卷积
				Layer_1_con_act.Forward(Layer_1_con.API_Data_Forward());        //激活
				Layer_1_pool.Forward_Ave(Layer_1_con_act.API_Data_Forward());   //池化

																				//第二层
				Layer_2_con.Forward(Layer_1_pool.API_Data_Forward());           //卷积
				Layer_2_con_act.Forward(Layer_2_con.API_Data_Forward());        //激活
				Layer_2_pool.Forward_Ave(Layer_2_con_act.API_Data_Forward());   //池化

																				//第三层
				Layer_3_linear.Forward(Layer_2_pool.API_Data_Forward());        //线性
				Layer_3_activation.Forward(Layer_3_linear.API_Data_Forward());  //激活

																				//第四层
				Layer_4_linear.Forward(Layer_3_activation.API_Data_Forward());  //线性
				Layer_4_activation.Forward(Layer_4_linear.API_Data_Forward());  //激活

																				//第五层
				Layer_5_linear.Forward(Layer_4_activation.API_Data_Forward());  //线性
				Layer_5_activation.Forward(Layer_5_linear.API_Data_Forward());  //激活

																				//计算损失值
				Layer_top.Cal_Score(Layer_5_activation.API_Data_Forward());     //评分
				Layer_top.Cal_Loss(label);                                      //损失
				Layer_top.Cal_Grade(label);                                     //梯度

				if (Layer_top.return_Label() == label)
					right_num++;
			}// for - check_photo_num

			fin_check.close(); //关闭当前测试文件

			double Accuracy = (double)right_num / (double)check_photo_num; //计算当前精度

			fout_testResult << "interation: " << interation << " Accuracy: " << Accuracy << std::endl; //将精度写入结果文件 
			cout << "interation: " << interation << " Accuracy: " << Accuracy << std::endl;
			//唯一一个开始时打开，结束时关闭的文件流
		}

		//储存当前各层参数值
		if (interation % 10 == 0 && interation != 0)
		{
			//构建储存文件路径
			string SaveInfo_path = FoutFolder_Path + FoutFile_SaveData + "_" + to_string(interation) + FoutFile_suffix;
			fout_SaveInfo.open(SaveInfo_path, ios::binary); //打开储存文件
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

			fout_SaveInfo.close(); //释放当前文件
		}

		//迭代后调整
		//if (interation != 0 && interation % 10 == 0)
		//{
		//	LS_layer_1 /= 3;
		//	LS_layer_2 /= 3;
		//	LS_layer_3 /= 3;
		//	LS_layer_4 /= 3;
		//	LS_layer_5 /= 3;
		//}


	}// for - iteration

	//善后处理

	fout_testResult.close();
}


int main()
{
	Cifar();

	_CrtDumpMemoryLeaks();

	return 0;
}
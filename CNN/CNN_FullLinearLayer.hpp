#pragma once

/**************************
File Name:CNN_FullLinearLayer.h
Author: MiaoMiaoYoung

日志：
在构建好BP网络后
再一次，再一次，再一次，继续攻克卷积神经网络

2017年11月17日建立
该层实现全连接层的线性分类
未验证其正确性

2017年11月18日更新
将其更新为类模板
验证其正确性

2017年11月25日更新
引用Check.cpp进行进一步检查
将updata中改为类模板更新
测试通过

2017年11月28日更新
增加了文件写出功能，将网络训练结果得以保存
测试通过
****************************/

#include"Matrix.hpp"

//#define Check

#ifdef Check
extern void test_CNN_FullLinear_Layer(); //Check.cpp 实例验证是否正确
#endif // Check

namespace yDL
{

//#define CNN_TYPE double

/************************************************
	功能：实现神经网络的线性分类层
	参数：
  	      -   const int In_num,
	      -   const int Out_num,
	      -   const double LearningSpeed,
	      -   const vector<const Matrix<double>* const> InPointer,
	      -   const vector<const int> LaPointer
	说明：
	      -   f(x)=w(T)*x+b    //线性分类
	      -   Forward(const Matrix<double>&DataIn);                                    //正向传播数据
	      -   Backward(const Matrix<double>&GradeOut);                                 //反向传递误差
	      -   Updata(const Matrix<double>& GradeOut, const Matrix<double>& DataIn);    //进行参数的更新

	//API
	      -   API_Data_Forward();   //向前传递输出层的结果--const Matrix<double>&
	      -   API_Grade_Backward(); //向后传递当前层的误差--const Matrix<double>&
	*************************************************/
	template<class CNN_TYPE>
	class CNN_FullLinear_Layer
	{

	private:

		Matrix<CNN_TYPE>* Weight;  //当前层的权重
		Matrix<CNN_TYPE>* Bias;    //当前层的阈值

		Matrix<CNN_TYPE>* DataOut;    //输出层的结果（即当前层的输出）
		Matrix<CNN_TYPE>* GradeIn;    //输入层的梯度（即当前层的梯度）

		int Input_num;       //输入节点的个数
		int Output_num;      //输出节点的个数

		CNN_FullLinear_Layer(const CNN_FullLinear_Layer&) = delete;

	public:
		explicit CNN_FullLinear_Layer(
			const int In_num,
			const int Out_num,
			const CNN_TYPE trans_func(const CNN_TYPE x) = NULL
		);//构造函数

		~CNN_FullLinear_Layer();  //析构函数

		void Forward(const Matrix<CNN_TYPE>* const DataIn);                                      //正向传播数据
		void Backward(const Matrix<CNN_TYPE>* const GradeOut);                                   //反向传递误差
		void Updata(
			const double LS,
			const Matrix<CNN_TYPE>* const GradeOut,
			const Matrix<CNN_TYPE>* const DataIn
		); //进行参数的更新

	  //提供API接口
		const Matrix<CNN_TYPE>* const API_Data_Forward();   //向前传递输出层的结果
		const Matrix<CNN_TYPE>* const API_Grade_Backward(); //向后传递当前层的误差

		void Save_Info(std::ofstream& fout); //将当前数据写入文件中

#ifdef Check
		friend void test_FullLinear();
		friend void ::test_CNN_FullLinear_Layer(); //Check.cpp 实例验证是否正确
#endif // Check

	};

	/************************************************
	函数名称：CNN_FullLinear_Layer
	功    能: CNN_FullLinear_Layer的构造函数
	参    数：
 	           -  输入节点个数
	           -  输出节点个数
			   -  初始化Weight函数（默认为NULL）
	返    回：
	说    明：为DataOut\GradeIn\Weight\Bias分配空间
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
	函数名称：~CNN_FullLinear_Layer
	功    能:  CNN_FullLinear_Layer的析构函数
	参    数：
	返    回：
	说    明：为DataOut\GradeIn\Weight\Bias释放空间
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
	函数名称：Forward
	功    能:正向传播
	参    数：API传入需要训练的数据
	返    回：
	说    明：进行正向传播运算，矩阵乘法
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullLinear_Layer<CNN_TYPE>::Forward(const Matrix<CNN_TYPE>* const DataIn)
	{
		*DataOut = (*Weight)*(*DataIn) + (*Bias);
	}

	/************************************************
	函数名称:Backward
	功    能:反向传递误差
	参    数：API传入上一层的梯度项
	返    回：
	说    明：链式法则进行反向传播
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullLinear_Layer<CNN_TYPE>::Backward(const Matrix<CNN_TYPE>* const GradeOut)
	{
		*GradeIn = (*Weight)('T')*(*GradeOut); //反向传播梯度
	}

	/************************************************
	函数名称：Updata
	功    能:更新当前层上的参数
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	void CNN_FullLinear_Layer<CNN_TYPE>::Updata(
		const double LS,
		const Matrix<CNN_TYPE>* const GradeOut,
		const Matrix<CNN_TYPE>* const DataIn
	)
	{
		Matrix<CNN_TYPE>temp(Output_num, Input_num);
		temp = (*GradeOut)*(*DataIn)('T'); //得到权重的偏置

		//if (1)
		//{
		//	ofstream fout_test("check\\test_linear5_output.txt", ios::binary); //声明打开保存结果文件
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
	函数名称：API_Data_Forward
	功    能:提供前向传播API接口
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const Matrix<CNN_TYPE>* const CNN_FullLinear_Layer<CNN_TYPE>::API_Data_Forward()
	{
		return DataOut;
	}

	/************************************************
	函数名称：API_Grade_Backward
	功    能:提供后向传递API接口
	参    数：
	返    回：
	说    明：
	*************************************************/
	template<class CNN_TYPE>
	const Matrix<CNN_TYPE>* const CNN_FullLinear_Layer<CNN_TYPE>::API_Grade_Backward()
	{
		return GradeIn;
	}

	/************************************************
	函数名称:Save_Info
	功    能:将训练成果储存在文件中，日后可以直接读取
	参    数：
	返    回：
	说    明：
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

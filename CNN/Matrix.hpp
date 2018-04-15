#pragma once //hpp文件，告诉编译器该文件只编译一次，避免重复编译产生问题

/**************************
File Name:Matrix.hpp
Author: MiaoMiaoYoung

日志：
2017年10月22日创建
基于double型的矩阵类，通过类模板实现一般情况的矩阵工具
//部分功能没有添加

2017年11月2日更新
添加矩阵的转置

2017年11月3日更新
添加外部函数对矩阵的改变 transfer
添加Hadamard乘积（同型矩阵对应元素相乘,为使用方便，添加为类函数，而不是友元函数，但这一做法尚待考量

2017年11月4日更新
添加外部函数对矩阵特定行列的查值，外部无法修改值
更新了类的说明
增加了矩阵对输入输出流的重载

2017年11月5日更新
添加返回矩阵的行列数目

2017年11月15日更新
添加了矩阵的卷积，并验证其正确性
添加了矩阵的最大池化，并验证其正确性
添加了矩阵的平均池化，并验证其正确性
并对上述进行三个函数进行函数，进行指针操作

2017年11月16日更新
添加了矩阵的180°旋转，并验证其正确性
添加了矩阵的扩充Pading,并验证其正确性
添加了最大池化的符号标记，并验证其正确性
添加了矩阵的Kronecker乘积（张量积），并验证其正确性
为Convolution重载了其他可能情况，并验证其正确性，防止内存泄漏等问题

2017年11月17日更新
添加了矩阵的列展开，并验证其正确性
添加了矩阵列向量的拼接，并验证其正确性
添加了矩阵列向量向方块矩阵的转化

2017年11月18日更新
添加了矩阵的零中心化操作，并验证其正确性

2017年11月19日更新
添加了矩阵的归一化处理
添加了复制=函数的报警

2017年11月28日更新
更改了矩阵的输出
添加换行符

2017年12月4日更新
添加了矩阵的数加

需要更新:
矩阵乘法的快速实现（）//可以考虑
矩阵转置的快速实现O(1)原地转置
矩阵的行列变换
判断是否为行向量
判断是否为列向量

//功能尚多，需要验证其正确性
//矩阵的赋值,assignment不太安全
//继续验证矩阵的正确性！！！
//矩阵的多处功能并没有完善的考虑错误处理的情况，要慎重，要增加安全设置
****************************/


#include<new>
#include<assert.h>
#include<vector>
#include<iostream>
#include<iomanip>
#include<fstream>

using std::cout;

/************************************************
功  能：矩阵类，实现矩阵的操作
参  数：
		-  const int row
		-  const int col
说  明：
		-  Initialize();               //矩阵的初始化，将矩阵重置为0（默认）
		-  show();                     //打印当前矩阵
		-  is_Valid();                 //判断当前矩阵是否有效，有效返回true
		-  assigment();                //矩阵的赋值，内存的简单搬运
		-  []                          //矩阵的下标索引
		-  Get_Value(i,j)const;        //外部函数查值，外部函数不可写
		-  = + - *  == !=              //等号赋值，加法，减法，矩阵数乘，矩阵乘法，判断是否为同型矩阵
		-  (Type)                      // Type='T' - 矩阵的转置  Type='R' - 矩阵的旋转180° 
		-  transfer                    //矩阵的变化,每一个元素由func_pointer指向的函数改变值 (func_pointer)
		-  Hadamard                    //矩阵Hadamard乘积，同型矩阵对应元素相乘              (another Matrix)
		-  Kronecker                   //矩阵Kronecker乘积（张量积），两个矩阵规定了顺序     (mul_1 Matrix,mul_2 Matrix)
		-  Convolution                 //矩阵的卷积              (vector<Matrix<Tn>*>data,vector<Matrix<Tn>*>kernel,int Stride)
		-  Max_Pooling                 //矩阵的最大池化          (Matrix data, int Pool_size, int Stride)
		-  Max_Pooling_Sign            //矩阵的最大池化标记      (Matrix data, int Pool_size, int Stride)
		-  Ave_Pooling                 //矩阵的平均池化          (Matrix data, int Pool_size, int Stride)
		-  Pading                      //矩阵的扩充              (int Pad_Size)
		-  Column_Expansion            //矩阵的列展开            (const Matrix<Tn>& data)
		-  Column_Joint                //列矩阵的拼接            (std::vector<Matrix<Tn>*>data)
		-  Column_Change               //列矩阵向一般矩阵的转换  (const Matrix<Tn>& column,std::vector<Matrix<Tn>*>data)
		-  支持对输入流、输出流的重载
		-  返回矩阵的行、列
*************************************************/
template<class T>
class Matrix
{

public:

	Matrix(const int row, const int col);  //构造函数
	Matrix(const Matrix<T>&copy);         //复制构造函数
	~Matrix();                            //析构函数

	void Initialize(const int value = 0);   //矩阵的初始化，默认初始化为0
	void show()const;                     //打印当前矩阵

	bool is_Valid()const;                   //判断是否有效
	//bool is_ColumnVector();               //判断是否为列向量
	//bool is_RowVector();                  //判断是否为行向量

	double Count_Ave();                     //求当前矩阵的平均值
	double Count_Var();                     //求当前矩阵的方差

	Matrix<T>& assigment(const void* copy, const int size); //矩阵的赋值

	const int return_row()const;                  //返回矩阵的行
	const int return_col()const;                  //返回矩阵的列

	T* operator[](const int i); //下标索引,外部函数可写
	const T Get_Value(const int row, const int col)const;   //外部函数查值，外部函数不可写
	const T Get_Determinant_Value()const;                   //矩阵的行列式求值

	Matrix<T>& operator=(const Matrix<T>& copy);            //矩阵的赋值函数
	Matrix<T>& operator+=(const Matrix<T>& copy);           //矩阵的加法
	Matrix<T>& operator-=(const Matrix<T>& copy);           //矩阵的减法
	Matrix<T>& operator*=(const T& value);                  //矩阵的数乘
	Matrix<T>  operator()(const char Type)const;            //矩阵的转置
	Matrix<T>  transfer(const T trans_func(const T x))const;//矩阵的变化 
	Matrix<T>  Hadamard(const Matrix<T>&another)const;      //矩阵Hadamard乘积，同型矩阵对应元素相乘
	Matrix<T>  Pading(const int Pad_Size)const;             //矩阵的扩充
	Matrix<T>  Zero_Center()const;                          //矩阵的零中心化
	Matrix<T>  Normalization(const T min,const T max)const;                        //矩阵的归一化处理

	template<class Tn>                                      //矩阵的卷积
	friend Matrix<Tn>  Convolution(const Matrix<Tn>* const data,const Matrix<Tn>* const kernel,const int Stride);
	template<class Tn>                                      //矩阵的卷积
	friend Matrix<Tn>  Convolution(const std::vector<Matrix<Tn>*>data, const std::vector<Matrix<Tn>*>kernel, const int Stride);
	template<class Tn>                                      //矩阵的最大池化
	friend Matrix<Tn> Max_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride);
	template<class Tn>                                      //矩阵的最大池化 - 符号标记
	friend std::vector<std::vector<bool>> Max_Pooling_Sign(const Matrix<Tn>& data, const int Pool_size, const int Stride);
	template<class Tn>                                      //矩阵的平均池化
	friend Matrix<Tn> Ave_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride);

	template<class Tn>                                      //矩阵的卷积
	friend void Convolution(std::vector<Matrix<Tn>*>data, std::vector<Matrix<Tn>*>kernel, const int Stride,Matrix<Tn>** retu);
	template<class Tn>                                      //矩阵的最大池化
	friend void Max_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride, Matrix<Tn>** retu);
	template<class Tn>                                      //矩阵的平均池化
	friend void Ave_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride, Matrix<Tn>** retu);
	template<class Tn>                                      //矩阵的按列展开
	friend Matrix<Tn> Column_Expansion(const Matrix<Tn>& data);
	template<class Tn>                                      //列矩阵的拼接
	friend Matrix<Tn> Column_Joint(std::vector<Matrix<Tn>*>data);
	template<class Tn>                                      //列矩阵向普通矩阵的转换
	friend void Column_Change(const Matrix<Tn>& column,std::vector<Matrix<Tn>*>data);


	template<class Tn>
	friend Matrix<Tn> operator+(const Matrix<Tn>& add_1, const Matrix<Tn>& add_2); //矩阵加法
	template<class Tn>
	friend Matrix<Tn> operator+(const Matrix<Tn>& add_1, const Tn& value); //矩阵加法
	template<class Tn>
	friend Matrix<Tn> operator-(const Matrix<Tn>& sub_1, const Matrix<Tn>& sub_2); //矩阵减法
	template<class Tn>
	friend Matrix<Tn> operator*(const Matrix<Tn>& mul_1, const Matrix<Tn>& mul_2); //矩阵乘法
	template<class Tn>
	friend Matrix<Tn> Kronecker(const Matrix<Tn>&mul_1, const Matrix<Tn>&mul_2);   //矩阵Kronecker乘积（张量积），规定顺序
	template<class Tn,class V>
	friend Matrix<Tn> operator*(const Matrix<Tn>& mul, const V& value);            //矩阵的数乘
	template<class Tn, class V> 
	friend Matrix<Tn> operator*(const V& value, const Matrix<Tn>& mul);            //矩阵的数乘

	//流的重载
	template<class Tn>
	friend std::ofstream& operator<<(std::ofstream&out, Matrix<Tn>& M);              //文件输出流重载
	template<class Tn>
	friend std::ifstream& operator >> (std::ifstream&in, Matrix<Tn>& M);             //文件输入流重载
	template<class Tn>
	friend std::ostream& operator<<(std::ostream&out, Matrix<Tn>& M);                //输出流重载
	template<class Tn>
	friend std::istream& operator >> (std::istream&in, Matrix<Tn>& M);               //输入流重载


	template<class Tn>
	friend bool operator==(const Matrix<Tn>& o1, const Matrix<Tn>& o2);              //判断矩阵大小是否相等
	template<class Tn>
	friend bool operator!=(const Matrix<Tn>& o1, const Matrix<Tn>& o2);              //判断矩阵大小是否相等


private:
	T* M_value;   //存储矩阵的值
	int M_row;    //矩阵的行
	int M_col;    //矩阵的列

};


//矩阵的构造函数
template<class T>
Matrix<T>::Matrix(const int row, const int col) :M_row(row), M_col(col)
{
	M_value = new(std::nothrow)T[row*col];
	assert(M_value != NULL);
}

//矩阵的复制构造函数
template<class T>
Matrix<T>::Matrix(const Matrix<T>& copy) :M_row(copy.M_row), M_col(copy.M_col)
{
	int size = copy.M_row*copy.M_col;
	M_value = new(std::nothrow)T[size];
	assert(M_value != NULL);
	memcpy(M_value, copy.M_value, size * sizeof(T));
}

//矩阵的析构函数
template<class T>
Matrix<T>::~Matrix()
{
	delete[] M_value;
	M_value = NULL;

	M_row = 0;
	M_col = 0;
}

//返回矩阵的行
template<class T>
const int Matrix<T>::return_row()const
{
	return M_row;
}

//返回矩阵的列
template<class T>
const int Matrix<T>::return_col()const
{
	return M_col;
}

//矩阵的初始化（默认初始化值为0）
template<class T>
void Matrix<T>::Initialize(const int value = 0)
{
	memset(M_value, value, M_row*M_col * sizeof(T));
}

//矩阵的行列式求值
template<class T>
const T Matrix<T>::Get_Determinant_Value()const
{
	T value = 0;

	for (int cnt = 0; cnt < M_col*M_row; cnt++)
		value += M_value[cnt];

	return value;
}

//矩阵的下表索引,对矩阵内部可写
template<class T>
T* Matrix<T>::operator[](const int i)
{
	return &M_value[i*M_col];
}

//矩阵的值传出
template<class T>
const T Matrix<T>::Get_Value(const int row, const int col)const
{
	return M_value[row*M_col + col];
}

//矩阵的赋值,矩阵大小不对时不赋值
template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& copy)
{
	if (M_col != copy.M_col || M_row != copy.M_row)
	{
		assert(M_col != copy.M_col); //报警
		return *this;
	}
		
	memcpy(M_value, copy.M_value, M_row*M_col * sizeof(T));
	return *this;
}

//求当前矩阵的平均值
template<class T>
inline double Matrix<T>::Count_Ave()
{
	int Length = M_row*M_col;
	double Value = 0;

	for (int cnt = 0; cnt < Length; cnt++)
		Value += M_value[cnt];

	double Ave = Value / (double)Length;

	return Ave;
}

template<class T>
inline double Matrix<T>::Count_Var()
{
	int Length = M_row*M_col;

	double Ave = Count_Ave(); //平均值
	double Value = 0;           //方差

	for (int cnt = 0; cnt < Length; cnt++)
		Value += (Ave - M_value[cnt])*(Ave - M_value[cnt]);

	double Var = Value / (double)Length;

	return Var;
}

//矩阵的赋值
template<class T>
Matrix<T>& Matrix<T>::assigment(const void* copy, const int size)
{
	memcpy((*this).M_value, copy, size);
	return *this;
}

//判断两个矩阵的大小是否一致
template<class Tn>
bool operator==(const Matrix<Tn>& o1, const Matrix<Tn>& o2)
{
	if (o1.M_row == o2.M_row&&o1.M_col == o2.M_col)
		return true;
	else
		return false;
}

//判断两个矩阵的大小是否一致
template<class Tn>
bool operator!=(const Matrix<Tn>& o1, const Matrix<Tn>& o2)
{
	if (o1.M_row == o2.M_row&&o1.M_col == o2.M_col)
		return false;
	else
		return true;
}

//矩阵加法
template<class T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& copy)
{
	if ((*this) != copy)
		return *this;

	int size = M_col*M_row;
	for (int i = 0; i < size; i++)
		M_value[i] += copy.M_value[i];
	return *this;
}

//矩阵加法
template<class Tn>
Matrix<Tn> operator+(const Matrix<Tn>& add_1, const Matrix<Tn>& add_2)
{
	if (add_1 != add_2)
		return Matrix<Tn>(0, 0);

	Matrix<Tn> temp(add_1.M_row, add_2.M_col);
	int size = temp.M_col*temp.M_row;
	for (int i = 0; i < size; i++)
		temp.M_value[i] = add_1.M_value[i] + add_2.M_value[i];

	return temp;
}

//矩阵数加
template<class Tn>
Matrix<Tn> operator+(const Matrix<Tn>& add_1, const Tn& value)
{
	Matrix<Tn> temp(add_1.M_row, add_1.M_col);
	int size = temp.M_col*temp.M_row;
	for (int i = 0; i < size; i++)
		temp.M_value[i] = add_1.M_value[i] + value;

	return temp;
}


//矩阵减法
template<class T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& subed)
{
	if ((*this) != subed)
		return *this;

	int size = M_col*M_row;
	for (int i = 0; i < size; i++)
		M_value[i] -= subed.M_value[i];
	return *this;
}

//矩阵减法
template<class Tn>
Matrix<Tn> operator-(const Matrix<Tn>& sub, const Matrix<Tn>& subed)
{
	if (sub != subed)
		return Matrix<Tn>(0, 0);

	Matrix<Tn> temp(sub.M_row, subed.M_col);
	int size = temp.M_col*temp.M_row;
	for (int i = 0; i < size; i++)
		temp.M_value[i] = sub.M_value[i] - subed.M_value[i];

	return temp;
}

//矩阵数乘
template<class T>
Matrix<T>& Matrix<T>::operator*=(const T& value)
{
	int size = M_col*M_row;
	for (int i = 0; i < size; i++)
		M_value[i] *= value;
	return *this;
}

//矩阵的转置
//矩阵的旋转180
template<class T>
Matrix<T> Matrix<T>::operator()(const char Type)const
{
	//矩阵的转置
	if (Type == 'T')
	{
		Matrix<T> temp(M_col, M_row);

		for (int i = 0; i < M_row; i++)
			for (int j = 0; j < M_col; j++)
				temp[j][i] = M_value[i*M_col + j];

		return temp;
	}
	else if (Type == 'R')
	{
		Matrix<T> temp(M_row, M_col);

		for (int i = 0; i < M_row; i++)
			for (int j = 0; j < M_col; j++)
				temp[i][j] = M_value[(M_row - i)*M_col - 1 - j];

		return temp;
	}

	return Matrix<T>(0, 0);
}


//矩阵的随函数的变化
template<class T>
Matrix<T> Matrix<T>::transfer(const T trans_func(const T x))const
{
	Matrix<T>temp(M_row, M_col);
	int size = temp.M_row*temp.M_col;
	for (int i = 0; i < size; i++)
		temp.M_value[i] = trans_func(M_value[i]);
	return temp;
}

//矩阵Hadamard乘积，同型矩阵对应元素相乘
template<class T>
Matrix<T> Matrix<T>::Hadamard(const Matrix<T>&another)const
{
	if (M_col != another.M_col || M_row != another.M_row)
		return Matrix<T>(0, 0);

	Matrix<T>temp(M_row, M_col);
	int size = M_row*M_col;
	for (int i = 0; i < size; i++)
		temp.M_value[i] = M_value[i] * another.M_value[i];
	return temp;
}

//矩阵Kronecker乘积,规定了前后顺序
template<class Tn>
Matrix<Tn> Kronecker(const Matrix<Tn>&mul_1, const Matrix<Tn>&mul_2) 
{

	int m1_row = mul_1.return_row();
	int m1_col = mul_1.return_col();

	int m2_row = mul_2.return_row();
	int m2_col = mul_2.return_col();

	int row = m1_row*m2_row;
	int col = m1_col*m2_col;

	Matrix<Tn>temp(row, col);

	int counter = 0;
	for (int cnt = 0; cnt < m1_row; cnt++)
		for (int index = 0; index < m2_row; index++)
		{
			for (int i = 0; i < m1_col; i++)
				for (int j = 0; j < m2_col; j++)
				{
					temp.M_value[counter] = mul_1.Get_Value(cnt, i)*mul_2.Get_Value(index, j);
					counter++;
				}
		}//for 外部的行循环

	return temp;
}

//矩阵的扩充
template<class T>
Matrix<T> Matrix<T>::Pading(const int Pad_Size)const
{
	int new_col = M_col + 2 * Pad_Size;
	int new_row = M_row + 2 * Pad_Size;
	Matrix<T> temp(new_row, new_col);
	temp.Initialize();

	for (int cnt = 0, index = Pad_Size; cnt < M_row; cnt++, index++)
		memcpy(&temp.M_value[index*new_col + Pad_Size], &M_value[cnt*M_col],sizeof(T)*M_col);

	return temp;
}

//矩阵的零中心化
template<class T>
Matrix<T> Matrix<T>::Zero_Center()const
{
	Matrix<T> temp(M_row, M_col);

	T ave= 0;
	int counter = M_row*M_col;
	for (int cnt = 0; cnt < counter; cnt++)
		ave += M_value[cnt];

	ave /= counter;
	for (int cnt = 0; cnt < counter; cnt++)
		temp.M_value[cnt] = M_value[cnt] - ave;

	return temp;
}

//矩阵的零中心化
template<class T>
Matrix<T> Matrix<T>::Normalization(const T min,const T max)const
{
	Matrix<T> temp(M_row, M_col);
	int counter = M_row*M_col;
	for (int cnt = 0; cnt < counter; cnt++)
		temp.M_value[cnt] = M_value[cnt] / (max-min);

	return temp;
}

//矩阵数乘
template<class Tn,class V>
Matrix<Tn> operator*(const Matrix<Tn>& mul, const V& value)
{
	Matrix<Tn> temp(mul.M_row, mul.M_col);
	int size = temp.M_col*temp.M_row;
	for (int i = 0; i < size; i++)
		temp.M_value[i] = mul.M_value[i] * (Tn)value;

	return temp;
}

//矩阵数乘
template<class Tn, class V>
Matrix<Tn> operator*(const V& value,const Matrix<Tn>& mul)
{
	Matrix<Tn> temp(mul.M_row, mul.M_col);
	int size = temp.M_col*temp.M_row;
	for (int i = 0; i < size; i++)
		temp.M_value[i] = mul.M_value[i] * (Tn)value;

	return temp;
}

//矩阵乘法,明确规定他的顺序
template<class Tn>
Matrix<Tn> operator*(const Matrix<Tn>& mul_1, const Matrix<Tn>& mul_2)
{
	if (mul_1.M_col != mul_2.M_row)
	{
		assert(1 == 0);
		return Matrix<Tn>(0, 0);
	}

	Matrix<Tn> temp(mul_1.M_row, mul_2.M_col);
	temp.Initialize();

	for (int k = 0; k < mul_1.M_row; k++)
		for (int i = 0; i < mul_2.M_row*mul_2.M_col; i++)
			temp.M_value[k*temp.M_col + i % mul_2.M_col] += mul_2.M_value[i] * mul_1.M_value[k*mul_1.M_col + i / mul_2.M_col];

	return temp;
}

//矩阵的卷积
template<class Tn>
Matrix<Tn>  Convolution(const std::vector<Matrix<Tn>*>data, const std::vector<Matrix<Tn>*>kernel, const int Stride)   //矩阵的卷积
{
	int kernel_row = kernel[0]->M_row;
	int kernel_col = kernel[0]->M_col;

	if (data[0]->M_row < kernel[0]->M_row)
		return Matrix<Tn>(0, 0);

	int block_row = (data[0]->M_row - kernel[0]->M_row) / Stride + 1;
	int block_col = kernel_row*kernel_col;

	int row_num = block_row*block_row; //矩阵进行卷积一行（或一列上的个数）,即大矩阵的行
	int col_num = block_col*(int)kernel.size(); //大矩阵的列的个数
	Matrix<Tn> temp_feature(row_num, col_num);

	//为大矩阵赋值
	Tn* temp_space = new(std::nothrow)Tn[block_col];
	assert(temp_space != NULL);
	for (unsigned int cnt = 0; cnt < data.size(); cnt++) //对每一张特征图进行遍历
	{
		for (int k = 0; k < row_num; k++) //对每一个行进行遍历
		{
			int s_r = k / block_row;
			int s_c = k % block_row;

			int counter = 0;
			for(int i=0;i<kernel_row;i++)
				for (int j = 0; j < kernel_col; j++)
				{
					temp_space[counter]=(*data[cnt]).Get_Value(s_r*Stride + i,s_c*Stride + j);
					counter++;
				}

			memcpy(&(temp_feature.M_value[k*col_num + cnt*block_col]), temp_space, sizeof(Tn)*block_col);
		}
	}
	delete[] temp_space;

	//为卷积核矩阵赋值
	Matrix<Tn> temp_kernel(block_col*(int)kernel.size(), 1);
	for (unsigned int cnt = 0; cnt < kernel.size(); cnt++)
		memcpy(&(temp_kernel.M_value[block_col*cnt]), kernel[cnt]->M_value, sizeof(Tn)*block_col);

	Matrix<Tn> result(row_num, 1);
	result = temp_feature*temp_kernel;

	Matrix<Tn>_return(block_row, block_row);
	_return.assigment(result.M_value, block_row*block_row*sizeof(Tn));
	return _return;
}

//矩阵的卷积
template<class Tn> 
Matrix<Tn>  Convolution(const Matrix<Tn>* const data, const Matrix<Tn>* const kernel, const int Stride)
{
	int kernel_row = (*kernel).return_row();
	int kernel_col = (*kernel).return_col();

	if ((*data).M_row < kernel_row)
		return Matrix<Tn>(0, 0);

	int block_row = (data[0].M_row - kernel[0].M_row) / Stride + 1;
	int block_col = kernel_row*kernel_col;

	int row_num = block_row*block_row; //矩阵进行卷积一行（或一列上的个数）,即大矩阵的行
	int col_num = block_col;           //大矩阵的列的个数
	Matrix<Tn> temp_feature(row_num, col_num);

	//为大矩阵赋值
	Tn* temp_space = new(std::nothrow)Tn[block_col];
	assert(temp_space != NULL);
	for (int k = 0; k < row_num; k++) //对每一个行进行遍历
	{
		int s_r = k / block_row;
		int s_c = k % block_row;

		int counter = 0;
		for (int i = 0; i < kernel_row; i++)
			for (int j = 0; j < kernel_col; j++)
			{
				temp_space[counter] = (*data).Get_Value(s_r*Stride + i, s_c*Stride + j);
				counter++;
			}

		memcpy(&(temp_feature.M_value[k*col_num]), temp_space, sizeof(Tn)*block_col);
	}
	delete[] temp_space;

	//为卷积核矩阵赋值
	Matrix<Tn> temp_kernel(block_col, 1);
	memcpy(&(temp_kernel.M_value[0]), kernel[0].M_value, sizeof(Tn)*block_col);

	Matrix<Tn> result(row_num, 1);
	result = temp_feature*temp_kernel;

	Matrix<Tn>_return(block_row, block_row);
	_return.assigment(result.M_value, block_row*block_row * sizeof(Tn));
	return _return;

}

//矩阵的卷积
template<class Tn>
void  Convolution(std::vector<Matrix<Tn>*>data, std::vector<Matrix<Tn>*>kernel, const int Stride,Matrix<Tn>** retu)   //矩阵的卷积
{
	int kernel_row = kernel[0]->M_row;
	int kernel_col = kernel[0]->M_col;

	if (data[0]->M_row < kernel[0]->M_row||(*retu)!=NULL)
		return;

	int block_row = (data[0]->M_row - kernel[0]->M_row) / Stride + 1;
	int block_col = kernel_row*kernel_col;

	int row_num = block_row*block_row; //矩阵进行卷积一行（或一列上的个数）,即大矩阵的行
	int col_num = block_col*(int)kernel.size(); //大矩阵的列的个数
	Matrix<Tn> temp_feature(row_num, col_num);

	//为大矩阵赋值
	Tn* temp_space = new(std::nothrow)Tn[block_col];
	assert(temp_space != NULL);
	for (unsigned int cnt = 0; cnt < data.size(); cnt++) //对每一张特征图进行遍历
	{
		for (int k = 0; k < row_num; k++) //对每一个行进行遍历
		{
			int s_r = k / block_row;
			int s_c = k % block_row;

			int counter = 0;
			for (int i = 0; i<kernel_row; i++)
				for (int j = 0; j < kernel_col; j++)
				{
					temp_space[counter] = (*data[cnt]).Get_Value(s_r*Stride + i, s_c*Stride + j);
					counter++;
				}

			memcpy(&(temp_feature.M_value[k*col_num + cnt*block_col]), temp_space, sizeof(Tn)*block_col);
		}
	}
	delete[] temp_space;

	//为卷积核矩阵赋值
	Matrix<Tn> temp_kernel(block_col*(int)kernel.size(), 1);
	for (unsigned int cnt = 0; cnt < kernel.size(); cnt++)
		memcpy(&(temp_kernel.M_value[block_col*cnt]), kernel[cnt]->M_value, sizeof(Tn)*block_col);

	Matrix<Tn> result(row_num, 1);
	result = temp_feature*temp_kernel;

	(*retu) = new(std::nothrow)Matrix<Tn>(block_row, block_row);
	assert((*retu) != NULL);
	(**retu).assigment(result.M_value, block_row*block_row * sizeof(Tn));
}

//矩阵的最大池化
template<class Tn>
Matrix<Tn> Max_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride)
{
	int size = (data.return_row() - Pool_size) / Stride + 1;
	Matrix<Tn> temp(size, size);

	for(int i=0;i<size;i++)
		for (int j = 0; j < size; j++)
		{
			Tn max_value = 0;
			for(int ii=0;ii<Pool_size;ii++)
				for (int jj = 0; jj < Pool_size; jj++)
				{
					if (ii == 0 && jj == 0)
					{
						max_value = data.Get_Value(i*Stride + ii, j*Stride + jj);
						continue;
					}

					Tn temp_value = data.Get_Value(i*Stride + ii, j*Stride + jj);
					if (max_value < temp_value)
						max_value = temp_value;
				}
			temp[i][j] = max_value;
		}

	return temp;
}

//矩阵的最大池化
template<class Tn>
void Max_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride, Matrix<Tn>** retu)
{
	if (*retu != NULL)
		return;

	int size = (data.return_row() - Pool_size) / Stride + 1;
	Matrix<Tn>* temp = new(std::nothrow)Matrix<Tn>(size, size);

	for (int i = 0; i<size; i++)
		for (int j = 0; j < size; j++)
		{
			Tn max_value = 0;
			for (int ii = 0; ii<Pool_size; ii++)
				for (int jj = 0; jj < Pool_size; jj++)
				{
					if (ii == 0 && jj == 0)
					{
						max_value = data.Get_Value(i*Stride + ii, j*Stride + jj);
						continue;
					}

					Tn temp_value = data.Get_Value(i*Stride + ii, j*Stride + jj);
					if (max_value < temp_value)
						max_value = temp_value;
				}
			(*temp)[i][j] = max_value;
		}
	*retu = temp;
}

//矩阵的最大池化 - 符号标记
template<class Tn>
std::vector<std::vector<bool>> Max_Pooling_Sign(const Matrix<Tn>& data, const int Pool_size, const int Stride)
{
	int size = (data.return_row() - Pool_size) / Stride + 1;

	std::vector<std::vector<bool>> sign;
	for (int cnt = 0; cnt < data.M_row; cnt++)
	{
		std::vector<bool>temp(data.M_col,false);
		sign.push_back(temp);
	}

	for (int i = 0; i<size; i++)
		for (int j = 0; j < size; j++)
		{
			Tn max_value = 0;
			int r = 0;
			int c = 0;
			for (int ii = 0; ii<Pool_size; ii++)
				for (int jj = 0; jj < Pool_size; jj++)
				{
					if (ii == 0 && jj == 0)
					{
						max_value = data.Get_Value(i*Stride + ii, j*Stride + jj);
						r = i*Stride + ii;
						c = j*Stride + jj;
						continue;
					}

					Tn temp_value = data.Get_Value(i*Stride + ii, j*Stride + jj);
					if (max_value < temp_value)
					{
						max_value = temp_value;
						r = i*Stride + ii;
						c = j*Stride + jj;
					}
				}
			sign[r][c] = true;
		}

	return sign;

}

//矩阵的平均池化
template<class Tn>
Matrix<Tn> Ave_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride)
{
	int size = (data.return_row() - Pool_size) / Stride + 1;
	Matrix<Tn> temp(size, size);

	int Pool_area = Pool_size*Pool_size;
	for (int i = 0; i<size; i++)
		for (int j = 0; j < size; j++)
		{
			Tn ave_value = 0;
			for (int ii = 0; ii < Pool_size; ii++)
				for (int jj = 0; jj < Pool_size; jj++)
					ave_value += data.Get_Value(i*Stride + ii, j*Stride + jj);

			temp[i][j] = ave_value / Pool_area;
		}

	return temp;
}

//矩阵的平均池化
template<class Tn>
void Ave_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride, Matrix<Tn>** retu)
{
	if (*retu != NULL)
		return;

	int size = (data.return_row() - Pool_size) / Stride + 1;
	Matrix<Tn>* temp = new(std::nothrow)Matrix<Tn>(size, size);

	int Pool_area = size*size;
	for (int i = 0; i<size; i++)
		for (int j = 0; j < size; j++)
		{
			Tn ave_value = 0;
			for (int ii = 0; ii < Pool_size; ii++)
				for (int jj = 0; jj < Pool_size; jj++)
					ave_value += data.Get_Value(i*Stride + ii, j*Stride + jj);

			(*temp)[i][j] = ave_value / Pool_area;
		}

	*retu = temp;
}

//矩阵的列展开
template<class Tn>
Matrix<Tn> Column_Expansion(const Matrix<Tn>& data)
{
	int num = data.return_row()*data.return_col();
	Matrix<Tn> temp(num, 1);
	memcpy(temp.M_value, data.M_value, sizeof(Tn)*num);
	return temp;
}

//列矩阵的拼接
template<class Tn>
Matrix<Tn> Column_Joint(std::vector<Matrix<Tn>*>data)
{
	int size = 0;
	for (unsigned int cnt = 0; cnt < data.size(); cnt++)
		size += (*data[cnt]).return_row()*(*data[cnt]).return_col();

	Matrix<Tn> temp(size, 1);
	for (unsigned int cnt = 0, star = 0; cnt < data.size(); cnt++)
	{
		int num = (*data[cnt]).M_row*(*data[cnt]).M_col;
		memcpy(&(temp.M_value[star]), (*data[cnt]).M_value, sizeof(Tn)*num);
		star += num;
	}

	return temp;
}

//列矩阵向普通矩阵的转换
template<class Tn>
void Column_Change(const Matrix<Tn>& column, std::vector<Matrix<Tn>*>data)
{
	int sum = 0;
	for (unsigned int cnt = 0; cnt < data.size(); cnt++)
	{
		int size = (*data[cnt]).return_row()*(*data[cnt]).return_col();
		memcpy((*data[cnt]).M_value, &column.M_value[sum], size * sizeof(Tn));
		sum += size;
	}
}



//矩阵输入输出流
//文件输出流
template<class Tn>
std::ofstream& operator<<(std::ofstream& out, Matrix<Tn>& M)
{
	const int Max_character_num = 400; //每行最多100个数字
	int size = M.M_row*M.M_col;
	for (int i = 0; i < size; i++)
	{
		if (i != 0 && i % M.M_col == 0)
			out << endl;          //每个100个进行换行

		out << M.M_value[i] << " ";
	}
	return out;
}

//文件输入流
template<class Tn>
std::ifstream& operator >> (std::ifstream& in, Matrix<Tn>& M)
{
	int size = M.M_row*M.M_col;
	for (int i = 0; i < size; i++)
		in >> M.M_value[i];
	return in;
}

//输出流
template<class Tn>
std::ostream& operator<<(std::ostream& out, Matrix<Tn>& M)
{
	for (int i = 0; i < M.M_row; i++)
	{
		for (int j = 0; j < M.M_col; j++)
			out << M.M_value[i*M.M_col + j] << " ";
		out << std::endl;
	}
	return out;
}

//输入流
template<class Tn>
std::istream& operator >> (std::istream& in, Matrix<Tn>& M)
{
	int size = M.M_row*M.M_col;
	for (int i = 0; i < size; i++)
		in >> M.M_value[i];
	return in;
}


//判断当前矩阵会否有效 (0,0)矩阵判断为无效
template<class T>
bool Matrix<T>::is_Valid()const
{
	if ((*this).M_row != 0 && (*this).M_col != 0)
		return true;
	else
		return false;
}

//矩阵的展示
template<class T>
void Matrix<T>::show()const
{
	for (int i = 0; i < M_row; i++)
	{
		for (int j = 0; j < M_col; j++)
			cout << M_value[i*M_col + j] << " ";
		cout << std::endl;
	}
}

#ifdef Check
void Check_Convolution()
{
	if (0)
	{
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

			Matrix<double>* a3 = new(std::nothrow)Matrix<double>(3, 3);
			assert(a3);
			double b3[9] = { 1,2,1,0,1,3,3,3,2 };
			(*a3).assigment(b3, sizeof(b3));

			data.push_back(a1);
			data.push_back(a2);
			data.push_back(a3);
		}

		std::vector<Matrix<double>*>kernel;

		if (1)
		{
			Matrix<double>* a1 = new(std::nothrow)Matrix<double>(2, 2);
			assert(a1);
			double b1[4] = { 1,1,2,2 };
			(*a1).assigment(b1, sizeof(b1));

			Matrix<double>* a2 = new(std::nothrow)Matrix<double>(2, 2);
			assert(a2);
			double b2[4] = { 1,1,1,1 };
			(*a2).assigment(b2, sizeof(b2));

			Matrix<double>* a3 = new(std::nothrow)Matrix<double>(2, 2);
			assert(a3);
			double b3[4] = { 0,1,1,0 };
			(*a3).assigment(b3, sizeof(b3));

			kernel.push_back(a1);
			kernel.push_back(a2);
			kernel.push_back(a3);
		}

		Matrix<double> result(2, 2);
		result = Convolution(data, kernel, 1);
		result.show();

		Matrix<double> _result_(2, 2);
		_result_ = Convolution(data[0], kernel[0], 1);
		_result_.show();

		Matrix<double>* _result = NULL;
		Convolution(data, kernel, 1, &_result);
		(*_result).show();
		delete _result;

		for (unsigned int i = 0; i < data.size(); i++)
			delete data[i];

		for (unsigned int i = 0; i < kernel.size(); i++)
			delete kernel[i];
	}

	if (1)
	{
		std::vector<Matrix<double>*>data;

		if (1)
		{
			Matrix<double>* a1 = new(std::nothrow)Matrix<double>(4, 4);
			assert(a1);
			double b1[16] = { 0,0,0,0,0,14,20,0,0,15,24,0,0,0,0,0};
			(*a1).assigment(b1, sizeof(b1));

			data.push_back(a1);
		}

		std::vector<Matrix<double>*>kernel;

		if (1)
		{
			Matrix<double>* a1 = new(std::nothrow)Matrix<double>(2, 2);
			assert(a1);
			double b1[4] = { 2,2,1,1 };
			(*a1).assigment(b1, sizeof(b1));

			kernel.push_back(a1);
		}

		Matrix<double> result(3, 3);
		result = Convolution(data, kernel, 1);
		result.show();

		Matrix<double> _result_(3, 3);
		_result_ = Convolution(data[0], kernel[0], 1);
		_result_.show();

		Matrix<double>* _result = NULL;
		Convolution(data, kernel, 1, &_result);
		(*_result).show();
		delete _result;

		for (unsigned int i = 0; i < data.size(); i++)
			delete data[i];

		for (unsigned int i = 0; i < kernel.size(); i++)
			delete kernel[i];

	}


}

void Check_MaxPooling()
{
	Matrix<double> a(3, 3);
	double b[9] = { 1,2,9,1,1,3,0,2,2 };
	a.assigment(b, sizeof(b));

	Matrix<double> result(2, 2);
	result = Max_Pooling(a, 2, 1);
	result.show();

	Matrix<double>* _result = NULL;
	Max_Pooling(a, 2, 1,&_result);
	(*_result).show();
	delete _result;

	Matrix<int> test(4, 4);

	int a[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,1,15,16 };
	test.assigment(a, sizeof(a));

	vector<vector<bool>> as = Max_Pooling_Sign(test, 2, 2);

	Max_Pooling(test, 2, 2).show();
	cout << endl;

	for (unsigned int i = 0; i < as.size(); i++)
	{
		for (unsigned int j = 0; j < as.size(); j++)
			cout << as[i][j] << " ";
		cout << endl;
	}

}

void Check_AvePooling()
{
	Matrix<double> a(3, 3);
	double b[9] = { 1,2,9,1,1,3,0,2,2 };
	a.assigment(b, sizeof(b));

	Matrix<double> result(2, 2);
	result = Ave_Pooling(a, 2, 1);
	result.show();

	Matrix<double>* _result = NULL;
	Ave_Pooling(a, 2, 1, &_result);
	(*_result).show();
	delete _result;
}

void Check_Rot180()
{
	Matrix<int> test(2, 3);

	int a[6] = { 1,2,3,4,5,6 };
	test.assigment(a, sizeof(a));

	test.show();

	test('R').show();
}

void Check_Kronecker()
{
	Matrix<int> a(2, 2), b(3, 3);

	int a0[4] = { 1,1,1,1 };
	int b0[9] = { 0,3,2,1,4,5,6,7,8 };

	a.assigment(a0, sizeof(a0));
	b.assigment(b0, sizeof(b0));

	Kronecker(b, a).show();

}

void Check_Column_Expansion_Joint()
{
	Matrix<double> test(3, 3);
	Matrix<double> k(3, 3);

	double a[9] = { 1,2,3,4,5,6,7,8,9 };
	double b[9] = { 11,22,33,44,55,66,77,88,99 };

	test.assigment(a, sizeof(a));
	k.assigment(b, sizeof(b));

	Column_Expansion(test).show();

	vector<Matrix<double>*> as;
	as.push_back(&test);
	as.push_back(&k);

	cout << std::endl << "****" << std::endl;

	Column_Joint(as).show();
}

void Check_Column_Change()
{
	Matrix<double> test(3, 3);
	Matrix<double> k(3, 3);

	vector<Matrix<double>*> as;
	as.push_back(&test);
	as.push_back(&k);

	Matrix<double> a(18, 1);
	double ka[18] = { 1,2,3,4,5,6,7,8,9,0,11,12,13,14,15,16,17,18 };
	a.assigment(ka, sizeof(ka));

	Column_Change(a, as);

	test.show();

	cout << std::endl;

	k.show();
}

#endif

#undef Check

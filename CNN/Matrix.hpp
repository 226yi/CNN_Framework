#pragma once //hpp�ļ������߱��������ļ�ֻ����һ�Σ������ظ������������

/**************************
File Name:Matrix.hpp
Author: MiaoMiaoYoung

��־��
2017��10��22�մ���
����double�͵ľ����࣬ͨ����ģ��ʵ��һ������ľ��󹤾�
//���ֹ���û�����

2017��11��2�ո���
��Ӿ����ת��

2017��11��3�ո���
����ⲿ�����Ծ���ĸı� transfer
���Hadamard�˻���ͬ�;����ӦԪ�����,Ϊʹ�÷��㣬���Ϊ�ຯ������������Ԫ����������һ�����д�����

2017��11��4�ո���
����ⲿ�����Ծ����ض����еĲ�ֵ���ⲿ�޷��޸�ֵ
���������˵��
�����˾�������������������

2017��11��5�ո���
��ӷ��ؾ����������Ŀ

2017��11��15�ո���
����˾���ľ��������֤����ȷ��
����˾�������ػ�������֤����ȷ��
����˾����ƽ���ػ�������֤����ȷ��
�����������������������к���������ָ�����

2017��11��16�ո���
����˾����180����ת������֤����ȷ��
����˾��������Pading,����֤����ȷ��
��������ػ��ķ��ű�ǣ�����֤����ȷ��
����˾����Kronecker�˻�����������������֤����ȷ��
ΪConvolution�����������������������֤����ȷ�ԣ���ֹ�ڴ�й©������

2017��11��17�ո���
����˾������չ��������֤����ȷ��
����˾�����������ƴ�ӣ�����֤����ȷ��
����˾����������򷽿�����ת��

2017��11��18�ո���
����˾���������Ļ�����������֤����ȷ��

2017��11��19�ո���
����˾���Ĺ�һ������
����˸���=�����ı���

2017��11��28�ո���
�����˾�������
��ӻ��з�

2017��12��4�ո���
����˾��������

��Ҫ����:
����˷��Ŀ���ʵ�֣���//���Կ���
����ת�õĿ���ʵ��O(1)ԭ��ת��
��������б任
�ж��Ƿ�Ϊ������
�ж��Ƿ�Ϊ������

//�����ж࣬��Ҫ��֤����ȷ��
//����ĸ�ֵ,assignment��̫��ȫ
//������֤�������ȷ�ԣ�����
//����Ķദ���ܲ�û�����ƵĿ��Ǵ�����������Ҫ���أ�Ҫ���Ӱ�ȫ����
****************************/


#include<new>
#include<assert.h>
#include<vector>
#include<iostream>
#include<iomanip>
#include<fstream>

using std::cout;

/************************************************
��  �ܣ������࣬ʵ�־���Ĳ���
��  ����
		-  const int row
		-  const int col
˵  ����
		-  Initialize();               //����ĳ�ʼ��������������Ϊ0��Ĭ�ϣ�
		-  show();                     //��ӡ��ǰ����
		-  is_Valid();                 //�жϵ�ǰ�����Ƿ���Ч����Ч����true
		-  assigment();                //����ĸ�ֵ���ڴ�ļ򵥰���
		-  []                          //������±�����
		-  Get_Value(i,j)const;        //�ⲿ������ֵ���ⲿ��������д
		-  = + - *  == !=              //�ȺŸ�ֵ���ӷ����������������ˣ�����˷����ж��Ƿ�Ϊͬ�;���
		-  (Type)                      // Type='T' - �����ת��  Type='R' - �������ת180�� 
		-  transfer                    //����ı仯,ÿһ��Ԫ����func_pointerָ��ĺ����ı�ֵ (func_pointer)
		-  Hadamard                    //����Hadamard�˻���ͬ�;����ӦԪ�����              (another Matrix)
		-  Kronecker                   //����Kronecker�˻���������������������涨��˳��     (mul_1 Matrix,mul_2 Matrix)
		-  Convolution                 //����ľ��              (vector<Matrix<Tn>*>data,vector<Matrix<Tn>*>kernel,int Stride)
		-  Max_Pooling                 //��������ػ�          (Matrix data, int Pool_size, int Stride)
		-  Max_Pooling_Sign            //��������ػ����      (Matrix data, int Pool_size, int Stride)
		-  Ave_Pooling                 //�����ƽ���ػ�          (Matrix data, int Pool_size, int Stride)
		-  Pading                      //���������              (int Pad_Size)
		-  Column_Expansion            //�������չ��            (const Matrix<Tn>& data)
		-  Column_Joint                //�о����ƴ��            (std::vector<Matrix<Tn>*>data)
		-  Column_Change               //�о�����һ������ת��  (const Matrix<Tn>& column,std::vector<Matrix<Tn>*>data)
		-  ֧�ֶ��������������������
		-  ���ؾ�����С���
*************************************************/
template<class T>
class Matrix
{

public:

	Matrix(const int row, const int col);  //���캯��
	Matrix(const Matrix<T>&copy);         //���ƹ��캯��
	~Matrix();                            //��������

	void Initialize(const int value = 0);   //����ĳ�ʼ����Ĭ�ϳ�ʼ��Ϊ0
	void show()const;                     //��ӡ��ǰ����

	bool is_Valid()const;                   //�ж��Ƿ���Ч
	//bool is_ColumnVector();               //�ж��Ƿ�Ϊ������
	//bool is_RowVector();                  //�ж��Ƿ�Ϊ������

	double Count_Ave();                     //��ǰ�����ƽ��ֵ
	double Count_Var();                     //��ǰ����ķ���

	Matrix<T>& assigment(const void* copy, const int size); //����ĸ�ֵ

	const int return_row()const;                  //���ؾ������
	const int return_col()const;                  //���ؾ������

	T* operator[](const int i); //�±�����,�ⲿ������д
	const T Get_Value(const int row, const int col)const;   //�ⲿ������ֵ���ⲿ��������д
	const T Get_Determinant_Value()const;                   //���������ʽ��ֵ

	Matrix<T>& operator=(const Matrix<T>& copy);            //����ĸ�ֵ����
	Matrix<T>& operator+=(const Matrix<T>& copy);           //����ļӷ�
	Matrix<T>& operator-=(const Matrix<T>& copy);           //����ļ���
	Matrix<T>& operator*=(const T& value);                  //���������
	Matrix<T>  operator()(const char Type)const;            //�����ת��
	Matrix<T>  transfer(const T trans_func(const T x))const;//����ı仯 
	Matrix<T>  Hadamard(const Matrix<T>&another)const;      //����Hadamard�˻���ͬ�;����ӦԪ�����
	Matrix<T>  Pading(const int Pad_Size)const;             //���������
	Matrix<T>  Zero_Center()const;                          //����������Ļ�
	Matrix<T>  Normalization(const T min,const T max)const;                        //����Ĺ�һ������

	template<class Tn>                                      //����ľ��
	friend Matrix<Tn>  Convolution(const Matrix<Tn>* const data,const Matrix<Tn>* const kernel,const int Stride);
	template<class Tn>                                      //����ľ��
	friend Matrix<Tn>  Convolution(const std::vector<Matrix<Tn>*>data, const std::vector<Matrix<Tn>*>kernel, const int Stride);
	template<class Tn>                                      //��������ػ�
	friend Matrix<Tn> Max_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride);
	template<class Tn>                                      //��������ػ� - ���ű��
	friend std::vector<std::vector<bool>> Max_Pooling_Sign(const Matrix<Tn>& data, const int Pool_size, const int Stride);
	template<class Tn>                                      //�����ƽ���ػ�
	friend Matrix<Tn> Ave_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride);

	template<class Tn>                                      //����ľ��
	friend void Convolution(std::vector<Matrix<Tn>*>data, std::vector<Matrix<Tn>*>kernel, const int Stride,Matrix<Tn>** retu);
	template<class Tn>                                      //��������ػ�
	friend void Max_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride, Matrix<Tn>** retu);
	template<class Tn>                                      //�����ƽ���ػ�
	friend void Ave_Pooling(const Matrix<Tn>& data, const int Pool_size, const int Stride, Matrix<Tn>** retu);
	template<class Tn>                                      //����İ���չ��
	friend Matrix<Tn> Column_Expansion(const Matrix<Tn>& data);
	template<class Tn>                                      //�о����ƴ��
	friend Matrix<Tn> Column_Joint(std::vector<Matrix<Tn>*>data);
	template<class Tn>                                      //�о�������ͨ�����ת��
	friend void Column_Change(const Matrix<Tn>& column,std::vector<Matrix<Tn>*>data);


	template<class Tn>
	friend Matrix<Tn> operator+(const Matrix<Tn>& add_1, const Matrix<Tn>& add_2); //����ӷ�
	template<class Tn>
	friend Matrix<Tn> operator+(const Matrix<Tn>& add_1, const Tn& value); //����ӷ�
	template<class Tn>
	friend Matrix<Tn> operator-(const Matrix<Tn>& sub_1, const Matrix<Tn>& sub_2); //�������
	template<class Tn>
	friend Matrix<Tn> operator*(const Matrix<Tn>& mul_1, const Matrix<Tn>& mul_2); //����˷�
	template<class Tn>
	friend Matrix<Tn> Kronecker(const Matrix<Tn>&mul_1, const Matrix<Tn>&mul_2);   //����Kronecker�˻��������������涨˳��
	template<class Tn,class V>
	friend Matrix<Tn> operator*(const Matrix<Tn>& mul, const V& value);            //���������
	template<class Tn, class V> 
	friend Matrix<Tn> operator*(const V& value, const Matrix<Tn>& mul);            //���������

	//��������
	template<class Tn>
	friend std::ofstream& operator<<(std::ofstream&out, Matrix<Tn>& M);              //�ļ����������
	template<class Tn>
	friend std::ifstream& operator >> (std::ifstream&in, Matrix<Tn>& M);             //�ļ�����������
	template<class Tn>
	friend std::ostream& operator<<(std::ostream&out, Matrix<Tn>& M);                //���������
	template<class Tn>
	friend std::istream& operator >> (std::istream&in, Matrix<Tn>& M);               //����������


	template<class Tn>
	friend bool operator==(const Matrix<Tn>& o1, const Matrix<Tn>& o2);              //�жϾ����С�Ƿ����
	template<class Tn>
	friend bool operator!=(const Matrix<Tn>& o1, const Matrix<Tn>& o2);              //�жϾ����С�Ƿ����


private:
	T* M_value;   //�洢�����ֵ
	int M_row;    //�������
	int M_col;    //�������

};


//����Ĺ��캯��
template<class T>
Matrix<T>::Matrix(const int row, const int col) :M_row(row), M_col(col)
{
	M_value = new(std::nothrow)T[row*col];
	assert(M_value != NULL);
}

//����ĸ��ƹ��캯��
template<class T>
Matrix<T>::Matrix(const Matrix<T>& copy) :M_row(copy.M_row), M_col(copy.M_col)
{
	int size = copy.M_row*copy.M_col;
	M_value = new(std::nothrow)T[size];
	assert(M_value != NULL);
	memcpy(M_value, copy.M_value, size * sizeof(T));
}

//�������������
template<class T>
Matrix<T>::~Matrix()
{
	delete[] M_value;
	M_value = NULL;

	M_row = 0;
	M_col = 0;
}

//���ؾ������
template<class T>
const int Matrix<T>::return_row()const
{
	return M_row;
}

//���ؾ������
template<class T>
const int Matrix<T>::return_col()const
{
	return M_col;
}

//����ĳ�ʼ����Ĭ�ϳ�ʼ��ֵΪ0��
template<class T>
void Matrix<T>::Initialize(const int value = 0)
{
	memset(M_value, value, M_row*M_col * sizeof(T));
}

//���������ʽ��ֵ
template<class T>
const T Matrix<T>::Get_Determinant_Value()const
{
	T value = 0;

	for (int cnt = 0; cnt < M_col*M_row; cnt++)
		value += M_value[cnt];

	return value;
}

//������±�����,�Ծ����ڲ���д
template<class T>
T* Matrix<T>::operator[](const int i)
{
	return &M_value[i*M_col];
}

//�����ֵ����
template<class T>
const T Matrix<T>::Get_Value(const int row, const int col)const
{
	return M_value[row*M_col + col];
}

//����ĸ�ֵ,�����С����ʱ����ֵ
template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& copy)
{
	if (M_col != copy.M_col || M_row != copy.M_row)
	{
		assert(M_col != copy.M_col); //����
		return *this;
	}
		
	memcpy(M_value, copy.M_value, M_row*M_col * sizeof(T));
	return *this;
}

//��ǰ�����ƽ��ֵ
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

	double Ave = Count_Ave(); //ƽ��ֵ
	double Value = 0;           //����

	for (int cnt = 0; cnt < Length; cnt++)
		Value += (Ave - M_value[cnt])*(Ave - M_value[cnt]);

	double Var = Value / (double)Length;

	return Var;
}

//����ĸ�ֵ
template<class T>
Matrix<T>& Matrix<T>::assigment(const void* copy, const int size)
{
	memcpy((*this).M_value, copy, size);
	return *this;
}

//�ж���������Ĵ�С�Ƿ�һ��
template<class Tn>
bool operator==(const Matrix<Tn>& o1, const Matrix<Tn>& o2)
{
	if (o1.M_row == o2.M_row&&o1.M_col == o2.M_col)
		return true;
	else
		return false;
}

//�ж���������Ĵ�С�Ƿ�һ��
template<class Tn>
bool operator!=(const Matrix<Tn>& o1, const Matrix<Tn>& o2)
{
	if (o1.M_row == o2.M_row&&o1.M_col == o2.M_col)
		return false;
	else
		return true;
}

//����ӷ�
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

//����ӷ�
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

//��������
template<class Tn>
Matrix<Tn> operator+(const Matrix<Tn>& add_1, const Tn& value)
{
	Matrix<Tn> temp(add_1.M_row, add_1.M_col);
	int size = temp.M_col*temp.M_row;
	for (int i = 0; i < size; i++)
		temp.M_value[i] = add_1.M_value[i] + value;

	return temp;
}


//�������
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

//�������
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

//��������
template<class T>
Matrix<T>& Matrix<T>::operator*=(const T& value)
{
	int size = M_col*M_row;
	for (int i = 0; i < size; i++)
		M_value[i] *= value;
	return *this;
}

//�����ת��
//�������ת180
template<class T>
Matrix<T> Matrix<T>::operator()(const char Type)const
{
	//�����ת��
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


//������溯���ı仯
template<class T>
Matrix<T> Matrix<T>::transfer(const T trans_func(const T x))const
{
	Matrix<T>temp(M_row, M_col);
	int size = temp.M_row*temp.M_col;
	for (int i = 0; i < size; i++)
		temp.M_value[i] = trans_func(M_value[i]);
	return temp;
}

//����Hadamard�˻���ͬ�;����ӦԪ�����
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

//����Kronecker�˻�,�涨��ǰ��˳��
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
		}//for �ⲿ����ѭ��

	return temp;
}

//���������
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

//����������Ļ�
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

//����������Ļ�
template<class T>
Matrix<T> Matrix<T>::Normalization(const T min,const T max)const
{
	Matrix<T> temp(M_row, M_col);
	int counter = M_row*M_col;
	for (int cnt = 0; cnt < counter; cnt++)
		temp.M_value[cnt] = M_value[cnt] / (max-min);

	return temp;
}

//��������
template<class Tn,class V>
Matrix<Tn> operator*(const Matrix<Tn>& mul, const V& value)
{
	Matrix<Tn> temp(mul.M_row, mul.M_col);
	int size = temp.M_col*temp.M_row;
	for (int i = 0; i < size; i++)
		temp.M_value[i] = mul.M_value[i] * (Tn)value;

	return temp;
}

//��������
template<class Tn, class V>
Matrix<Tn> operator*(const V& value,const Matrix<Tn>& mul)
{
	Matrix<Tn> temp(mul.M_row, mul.M_col);
	int size = temp.M_col*temp.M_row;
	for (int i = 0; i < size; i++)
		temp.M_value[i] = mul.M_value[i] * (Tn)value;

	return temp;
}

//����˷�,��ȷ�涨����˳��
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

//����ľ��
template<class Tn>
Matrix<Tn>  Convolution(const std::vector<Matrix<Tn>*>data, const std::vector<Matrix<Tn>*>kernel, const int Stride)   //����ľ��
{
	int kernel_row = kernel[0]->M_row;
	int kernel_col = kernel[0]->M_col;

	if (data[0]->M_row < kernel[0]->M_row)
		return Matrix<Tn>(0, 0);

	int block_row = (data[0]->M_row - kernel[0]->M_row) / Stride + 1;
	int block_col = kernel_row*kernel_col;

	int row_num = block_row*block_row; //������о��һ�У���һ���ϵĸ�����,����������
	int col_num = block_col*(int)kernel.size(); //�������еĸ���
	Matrix<Tn> temp_feature(row_num, col_num);

	//Ϊ�����ֵ
	Tn* temp_space = new(std::nothrow)Tn[block_col];
	assert(temp_space != NULL);
	for (unsigned int cnt = 0; cnt < data.size(); cnt++) //��ÿһ������ͼ���б���
	{
		for (int k = 0; k < row_num; k++) //��ÿһ���н��б���
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

	//Ϊ����˾���ֵ
	Matrix<Tn> temp_kernel(block_col*(int)kernel.size(), 1);
	for (unsigned int cnt = 0; cnt < kernel.size(); cnt++)
		memcpy(&(temp_kernel.M_value[block_col*cnt]), kernel[cnt]->M_value, sizeof(Tn)*block_col);

	Matrix<Tn> result(row_num, 1);
	result = temp_feature*temp_kernel;

	Matrix<Tn>_return(block_row, block_row);
	_return.assigment(result.M_value, block_row*block_row*sizeof(Tn));
	return _return;
}

//����ľ��
template<class Tn> 
Matrix<Tn>  Convolution(const Matrix<Tn>* const data, const Matrix<Tn>* const kernel, const int Stride)
{
	int kernel_row = (*kernel).return_row();
	int kernel_col = (*kernel).return_col();

	if ((*data).M_row < kernel_row)
		return Matrix<Tn>(0, 0);

	int block_row = (data[0].M_row - kernel[0].M_row) / Stride + 1;
	int block_col = kernel_row*kernel_col;

	int row_num = block_row*block_row; //������о��һ�У���һ���ϵĸ�����,����������
	int col_num = block_col;           //�������еĸ���
	Matrix<Tn> temp_feature(row_num, col_num);

	//Ϊ�����ֵ
	Tn* temp_space = new(std::nothrow)Tn[block_col];
	assert(temp_space != NULL);
	for (int k = 0; k < row_num; k++) //��ÿһ���н��б���
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

	//Ϊ����˾���ֵ
	Matrix<Tn> temp_kernel(block_col, 1);
	memcpy(&(temp_kernel.M_value[0]), kernel[0].M_value, sizeof(Tn)*block_col);

	Matrix<Tn> result(row_num, 1);
	result = temp_feature*temp_kernel;

	Matrix<Tn>_return(block_row, block_row);
	_return.assigment(result.M_value, block_row*block_row * sizeof(Tn));
	return _return;

}

//����ľ��
template<class Tn>
void  Convolution(std::vector<Matrix<Tn>*>data, std::vector<Matrix<Tn>*>kernel, const int Stride,Matrix<Tn>** retu)   //����ľ��
{
	int kernel_row = kernel[0]->M_row;
	int kernel_col = kernel[0]->M_col;

	if (data[0]->M_row < kernel[0]->M_row||(*retu)!=NULL)
		return;

	int block_row = (data[0]->M_row - kernel[0]->M_row) / Stride + 1;
	int block_col = kernel_row*kernel_col;

	int row_num = block_row*block_row; //������о��һ�У���һ���ϵĸ�����,����������
	int col_num = block_col*(int)kernel.size(); //�������еĸ���
	Matrix<Tn> temp_feature(row_num, col_num);

	//Ϊ�����ֵ
	Tn* temp_space = new(std::nothrow)Tn[block_col];
	assert(temp_space != NULL);
	for (unsigned int cnt = 0; cnt < data.size(); cnt++) //��ÿһ������ͼ���б���
	{
		for (int k = 0; k < row_num; k++) //��ÿһ���н��б���
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

	//Ϊ����˾���ֵ
	Matrix<Tn> temp_kernel(block_col*(int)kernel.size(), 1);
	for (unsigned int cnt = 0; cnt < kernel.size(); cnt++)
		memcpy(&(temp_kernel.M_value[block_col*cnt]), kernel[cnt]->M_value, sizeof(Tn)*block_col);

	Matrix<Tn> result(row_num, 1);
	result = temp_feature*temp_kernel;

	(*retu) = new(std::nothrow)Matrix<Tn>(block_row, block_row);
	assert((*retu) != NULL);
	(**retu).assigment(result.M_value, block_row*block_row * sizeof(Tn));
}

//��������ػ�
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

//��������ػ�
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

//��������ػ� - ���ű��
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

//�����ƽ���ػ�
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

//�����ƽ���ػ�
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

//�������չ��
template<class Tn>
Matrix<Tn> Column_Expansion(const Matrix<Tn>& data)
{
	int num = data.return_row()*data.return_col();
	Matrix<Tn> temp(num, 1);
	memcpy(temp.M_value, data.M_value, sizeof(Tn)*num);
	return temp;
}

//�о����ƴ��
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

//�о�������ͨ�����ת��
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



//�������������
//�ļ������
template<class Tn>
std::ofstream& operator<<(std::ofstream& out, Matrix<Tn>& M)
{
	const int Max_character_num = 400; //ÿ�����100������
	int size = M.M_row*M.M_col;
	for (int i = 0; i < size; i++)
	{
		if (i != 0 && i % M.M_col == 0)
			out << endl;          //ÿ��100�����л���

		out << M.M_value[i] << " ";
	}
	return out;
}

//�ļ�������
template<class Tn>
std::ifstream& operator >> (std::ifstream& in, Matrix<Tn>& M)
{
	int size = M.M_row*M.M_col;
	for (int i = 0; i < size; i++)
		in >> M.M_value[i];
	return in;
}

//�����
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

//������
template<class Tn>
std::istream& operator >> (std::istream& in, Matrix<Tn>& M)
{
	int size = M.M_row*M.M_col;
	for (int i = 0; i < size; i++)
		in >> M.M_value[i];
	return in;
}


//�жϵ�ǰ��������Ч (0,0)�����ж�Ϊ��Ч
template<class T>
bool Matrix<T>::is_Valid()const
{
	if ((*this).M_row != 0 && (*this).M_col != 0)
		return true;
	else
		return false;
}

//�����չʾ
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

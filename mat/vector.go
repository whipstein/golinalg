package mat

import (
	"bytes"
	"encoding/gob"
	"math"
	"math/cmplx"
)

type Vector struct {
	Size int
	Data []float64
}

func (v *Vector) Get(n int) float64 {
	return v.Data[n]
}
func (v *Vector) GetPtr(n int) *float64 {
	return &v.Data[n]
}
func (v *Vector) GetCmplx(n int) complex128 {
	return complex(v.Data[n], 0)
}
func (v *Vector) GetMag(n int) float64 {
	return math.Abs(v.Data[n])
}
func (v *Vector) Set(n int, x float64) {
	v.Data[n] = x
}
func (v *Vector) SetAll(x float64) {
	for i := range v.Data {
		v.Data[i] = x
	}
}
func (v *Vector) Copy(idx int) *Vector {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(v)
	d := gob.NewDecoder(&b)
	result := Vector{}
	_ = d.Decode(&result)
	result.Data = v.Data[idx:]
	return &result
}
func (v *Vector) DeepCopy() *Vector {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(v)
	d := gob.NewDecoder(&b)
	result := Vector{}
	if len(v.Data) == 0 {
		result.Data = make([]float64, 0)
	}
	_ = d.Decode(&result)
	return &result
}
func (v *Vector) Off(idx int) *Vector {
	return &Vector{Size: v.Size, Data: v.Data[idx:]}
}
func (v *Vector) CVector() *CVector {
	cvf := CVectorFactory()
	y := cvf(v.Size)
	for i, val := range v.Data {
		y.Set(i, complex(val, 0))
	}
	return y
}

// Matrix references same data in memory
func (v *Vector) Matrix(r int, opts *MatOpts) *Matrix {
	var c int

	mf := MatrixFactory()
	if r == 0 {
		c = len(v.Data)
	} else {
		c = len(v.Data) / r
	}
	mat := mf(r, c, opts)
	mat.Data = v.Data
	return mat
}

// MatrixOff references same data in memory
func (v *Vector) MatrixOff(idx, r int, opts *MatOpts) *Matrix {
	var c int

	mf := MatrixFactory()
	if r == 0 {
		c = len(v.Data)
	} else {
		c = len(v.Data) / r
	}
	mat := mf(r, c, opts)
	mat.Data = v.Data[idx:]
	return mat
}

func VectorFactory() func(int) *Vector {
	return func(n int) *Vector {
		return &Vector{Size: n, Data: make([]float64, n)}
	}
}

// NewDataVectorFactory creates a new Vectorer with same data memory location
func VectorDataFactory() func([]float64) *Vector {
	return func(d []float64) *Vector {
		return &Vector{Size: len(d), Data: d}
	}
}

type CVector struct {
	Size int
	Data []complex128
}

func (v *CVector) Get(n int) complex128 {
	return v.Data[n]
}
func (v *CVector) GetConj(n int) complex128 {
	return cmplx.Conj(v.Data[n])
}
func (v *CVector) GetMag(n int) float64 {
	return cmplx.Abs(v.Data[n])
}
func (v *CVector) GetArg(n int) float64 {
	return cmplx.Phase(v.Data[n])
}
func (v *CVector) GetDeg(n int) float64 {
	return cmplx.Phase(v.Data[n]) * 180 / math.Pi
}
func (v *CVector) GetRe(n int) float64 {
	return real(v.Data[n])
}
func (v *CVector) GetIm(n int) float64 {
	return imag(v.Data[n])
}
func (v *CVector) GetReCmplx(n int) complex128 {
	return complex(real(v.Data[n]), 0)
}
func (v *CVector) GetImCmplx(n int) complex128 {
	return complex(0, imag(v.Data[n]))
}
func (v *CVector) GetConjProd(n int) float64 {
	return real(cmplx.Conj(v.Data[n]) * v.Data[n])
}
func (v *CVector) GetPtr(n int) *complex128 {
	return &v.Data[n]
}
func (v *CVector) Set(n int, x complex128) {
	v.Data[n] = x
}
func (v *CVector) SetAll(x complex128) {
	for i := range v.Data {
		v.Data[i] = x
	}
}
func (v *CVector) SetRe(n int, x float64) {
	v.Data[n] = complex(x, 0)
}
func (v *CVector) SetIm(n int, x float64) {
	v.Data[n] = complex(0, x)
}
func (v *CVector) SetReAll(x float64) {
	for i := range v.Data {
		v.Data[i] = complex(x, 0)
	}
}
func (v *CVector) SetImAll(x float64) {
	for i := range v.Data {
		v.Data[i] = complex(0, x)
	}
}
func (v *CVector) Copy(idx int) *CVector {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(v)
	d := gob.NewDecoder(&b)
	result := CVector{}
	_ = d.Decode(&result)
	result.Data = v.Data[idx:]
	return &result
}
func (v *CVector) DeepCopy() *CVector {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(v)
	d := gob.NewDecoder(&b)
	result := CVector{}
	if len(v.Data) == 0 {
		result.Data = make([]complex128, 0)
	}
	_ = d.Decode(&result)
	return &result
}
func (v *CVector) Off(idx int) *CVector {
	return &CVector{Size: v.Size, Data: v.Data[idx:]}
}

// CMatrix references same data in memory
func (v *CVector) CMatrix(r int, opts *MatOpts) *CMatrix {
	var c int

	mf := CMatrixFactory()
	if r == 0 {
		c = len(v.Data)
	} else {
		c = len(v.Data) / r
	}
	mat := mf(r, c, opts)
	mat.Data = v.Data
	return mat
}

// CMatrixOff references same data in memory
func (v *CVector) CMatrixOff(idx, r int, opts *MatOpts) *CMatrix {
	var c int

	mf := CMatrixFactory()
	if r == 0 {
		c = len(v.Data)
	} else {
		c = len(v.Data) / r
	}
	mat := mf(r, c, opts)
	mat.Data = v.Data[idx:]
	return mat
}

func CVectorFactory() func(int) *CVector {
	return func(n int) *CVector {
		return &CVector{Size: n, Data: make([]complex128, n)}
	}
}

// NewDataVectorFactory creates a new Vectorer with same data memory location
func CVectorDataFactory() func([]complex128) *CVector {
	return func(d []complex128) *CVector {
		return &CVector{Size: len(d), Data: d}
	}
}

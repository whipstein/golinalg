package mat

import (
	"bytes"
	"encoding/gob"
	"math"
	"math/cmplx"
)

type Vector struct {
	Size int
	Inc  int
	Data []float64
}

func (v *Vector) Append(x float64) {
	v.Size++
	v.Data = append(v.Data, x)
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
	for i := 0; i < v.Size; i++ {
		v.Set(i, x)
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
func (v *Vector) Off(idx int, n ...int) *Vector {
	inc := v.Inc
	if n != nil {
		inc = n[0]
	}
	return &Vector{Size: v.Size - idx, Inc: inc, Data: v.Data[idx:]}
}
func (v *Vector) CVector() *CVector {
	cvf := CVectorFactory()
	y := cvf(v.Size, v.Inc)
	for i := 0; i < v.Size; i++ {
		y.Set(i, complex(v.Get(i), 0))
	}
	return y
}
func (v *Vector) Iter(n int) []int {
	iter := make([]int, 0)

	if v.Inc > 0 {
		for i := 0; i < n; i++ {
			if i*v.Inc < v.Size {
				iter = append(iter, i*v.Inc)
			}
		}
	} else {
		for i := 0; i < n; i++ {
			if (-n+1+i)*v.Inc < v.Size {
				iter = append(iter, (-n+1+i)*v.Inc)
			}
		}
	}
	return iter
}

// Matrix references same data in memory
func (v *Vector) Matrix(r int, opts MatOpts) *Matrix {
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
func (v *Vector) MatrixOff(idx, r int, opts MatOpts) *Matrix {
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

func VectorFactory() func(...int) *Vector {
	return func(n ...int) *Vector {
		size := n[0]
		inc := 1
		if len(n) > 1 {
			if n[1] == 0 {
				panic("VectorFactory: cannot have an increment of 0")
			}
			inc = n[1]
			return &Vector{Size: n[0], Inc: inc, Data: make([]float64, size)}
		}
		return &Vector{Size: n[0], Inc: inc, Data: make([]float64, size)}
	}
}

// NewDataVectorFactory creates a new Vectorer with same data memory location
func VectorDataFactory() func([]float64, ...int) *Vector {
	return func(d []float64, n ...int) *Vector {
		size := len(d)
		if n == nil {
			return &Vector{Size: size, Inc: 1, Data: d}
		}
		if n[0] == 0 {
			panic("VectorDataFactory: cannot have an increment of 0")
		}
		inc := n[0]
		return &Vector{Size: size, Inc: inc, Data: d}
	}
}

type CVector struct {
	Size int
	Inc  int
	Data []complex128
}

func (v *CVector) Append(x complex128) {
	v.Size++
	v.Data = append(v.Data, x)
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
func (v *CVector) Off(idx int, n ...int) *CVector {
	inc := v.Inc
	if n != nil {
		inc = n[0]
	}
	return &CVector{Size: v.Size - idx, Inc: inc, Data: v.Data[idx:]}
}
func (v *CVector) Iter(n int) []int {
	iter := make([]int, 0)

	if v.Inc > 0 {
		for i := 0; i < n; i++ {
			if i*v.Inc < v.Size {
				iter = append(iter, i*v.Inc)
			}
		}
	} else {
		for i := 0; i < n; i++ {
			if (-n+1+i)*v.Inc < v.Size {
				iter = append(iter, (-n+1+i)*v.Inc)
			}
		}
	}
	return iter
}

// CMatrix references same data in memory
func (v *CVector) CMatrix(r int, opts MatOpts) *CMatrix {
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
func (v *CVector) CMatrixOff(idx, r int, opts MatOpts) *CMatrix {
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

func CVectorFactory() func(...int) *CVector {
	return func(n ...int) *CVector {
		size := n[0]
		if len(n) > 1 {
			if n[1] == 0 {
				panic("CVectorFactory: cannot have an increment of 0")
			}
			inc := n[1]
			if inc < 0 {
				size = (-size+1)*inc + 1
			}
			return &CVector{Size: n[0], Inc: inc, Data: make([]complex128, size)}
		}
		return &CVector{Size: n[0], Inc: 1, Data: make([]complex128, size)}
	}
}

// NewDataVectorFactory creates a new Vectorer with same data memory location
func CVectorDataFactory() func([]complex128, ...int) *CVector {
	return func(d []complex128, n ...int) *CVector {
		size := len(d)
		if n == nil {
			return &CVector{Size: size, Inc: 1, Data: d}
		}
		if n[0] == 0 {
			panic("VectorDataFactory: cannot have an increment of 0")
		}
		inc := n[0]
		if inc < 0 {
			size = 1 - (size-1)/inc
		}
		return &CVector{Size: size, Inc: inc, Data: d}
	}
}

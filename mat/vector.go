package mat

import (
	"fmt"
	"math"
	"math/cmplx"
)

func getVIdx(size, inc, idx int) (idxOut int) {
	if inc < 0 {
		idxOut = (size - 1) - idx*abs(inc)
	} else {
		idxOut = idx * inc
	}

	if idxOut < 0 {
		idxOut = 0
	} else if idxOut >= size {
		idxOut = size - 1
	}
	return
}

// dcabs1 computes |Re(.)| + |Im(.)| of a double complex number
func dcabs1(z complex128) (dcabs1Return float64) {

	dcabs1Return = math.Abs(real(z)) + math.Abs(imag(z))
	return
}

type Vector struct {
	size int
	data []float64
}

func (v *Vector) Append(x ...float64) {
	for _, xx := range x {
		v.size++
		v.data = append(v.data, xx)
	}
}
func (v *Vector) Data() []float64 {
	return v.data
}
func (v *Vector) Get(n int) float64 {
	return v.data[n]
}
func (v *Vector) GetPtr(n int) *float64 {
	return &v.data[n]
}
func (v *Vector) GetCmplx(n int) complex128 {
	return complex(v.data[n], 0)
}
func (v *Vector) GetMag(n int) float64 {
	return math.Abs(v.data[n])
}
func (v *Vector) Set(n int, x float64) {
	v.data[n] = x
}
func (v *Vector) SetAll(x float64) {
	for i := 0; i < v.size; i++ {
		v.Set(i, x)
	}
}
func (v *Vector) Size() int {
	return v.size
}
func (v *Vector) DeepCopy() *Vector {
	result := &Vector{size: v.size, data: make([]float64, v.size, 2*v.size)}

	copy(result.data, v.data)

	return result
}
func (v *Vector) Off(idx int) *Vector {
	return &Vector{size: v.size - idx, data: v.data[idx:]}
}
func (v *Vector) CVector() *CVector {
	cvf := CVectorFactory()
	y := cvf(v.size)
	for i := 0; i < v.size; i++ {
		y.Set(i, complex(v.Get(i), 0))
	}
	return y
}
func (v *Vector) Iter(n, inc int) []int {
	iter := make([]int, n)

	for i := 0; i < n; i++ {
		iter[i] = getVIdx(v.size, inc, i)
	}

	return iter
}

// Matrix references same data in memory
func (v *Vector) Matrix(r int, opts MatOpts) *Matrix {
	var c int

	mf := MatrixFactory()
	if r == 0 {
		c = len(v.data)
	} else {
		c = len(v.data) / r
	}
	mat := mf(r, c, opts)
	mat.Data = v.data
	return mat
}

// Asum takes the sum of the absolute values
func (v *Vector) Asum(n, inc int) (dasumReturn float64) {
	var iter []int
	if n <= 0 {
		return
	} else {
		iter = v.Iter(n, inc)
	}

	for _, ix := range iter {
		dasumReturn += math.Abs(v.Get(ix))
	}
	return
}

// Axpy constant times a vector plus a vector.
func (v *Vector) Axpy(n int, da float64, dx *Vector, incx, inc int) {
	if n <= 0 || da == 0 {
		return
	}

	ix := dx.Iter(n, incx)
	iy := v.Iter(n, inc)

	for i := 0; i < n; i++ {
		v.Set(iy[i], v.Get(iy[i])+da*dx.Get(ix[i]))
	}
}

// Copy copies a vector, dx, into vector
func (v *Vector) Copy(n int, dx *Vector, incx, inc int) {
	if n <= 0 {
		return
	}

	ix := dx.Iter(n, incx)
	iy := v.Iter(n, inc)

	for i := 0; i < n; i++ {
		v.Set(iy[i], dx.Get(ix[i]))
	}
}

// Dot forms the dot product of two vectors.
func (v *Vector) Dot(n int, dx *Vector, incx, inc int) (ddotReturn float64) {
	if n <= 0 {
		return
	}

	ix := dx.Iter(n, incx)
	iy := v.Iter(n, inc)

	for i := 0; i < n; i++ {
		ddotReturn += dx.Get(ix[i]) * v.Get(iy[i])
	}

	return
}

// Nrm2 returns the euclidean norm of a vector via the function
// name, so that Dnrm2 := sqrt( x'*x )
func (v *Vector) Nrm2(n, inc int) (dnrm2Return float64) {
	var absxi, scale, ssq float64
	var ix int

	if n < 1 || inc < 1 {
		return
	} else if n == 1 {
		return v.GetMag(0)
	} else {
		ssq = 1
		//        The following loop is equivalent to this call to the LAPACK
		//        auxiliary routine:
		//        CALL DLASSQ( N, X, INCX, SCALE, SSQ )
		//
		for _, ix = range v.Iter(n, inc) {
			if v.Get(ix) != 0 {
				absxi = v.GetMag(ix)
				if scale < absxi {
					ssq = 1 + ssq*math.Pow(scale/absxi, 2)
					scale = absxi
				} else {
					ssq += math.Pow(absxi/scale, 2)
				}
			}
		}
		dnrm2Return = scale * math.Sqrt(ssq)
	}

	return
}

// Rot applies a plane rotation
func (v *Vector) Rot(n int, dx *Vector, incx, inc int, c, s float64) {
	var dtemp float64

	if n <= 0 {
		return
	}

	ix := dx.Iter(n, incx)
	iy := v.Iter(n, inc)

	for i := 0; i < n; i++ {
		dtemp = c*dx.Get(ix[i]) + s*v.Get(iy[i])
		v.Set(iy[i], c*v.Get(iy[i])-s*dx.Get(ix[i]))
		dx.Set(ix[i], dtemp)
	}
}

// Rotm applies the modified Givens transformation, H, to the 2 x n matrix
//    (DX**T) , where **T indicates transpose. The elements of dx are in
//    (DY**T)
//
//    DX(LX+I*INCX), I = 0 TO N-1, WHERE LX = 1 IF INCX .GE. 0, ELSE
//    LX = (-INCX)*N, AND SIMILARLY FOR SY USING LY AND INCY.
//    WITH DPARAM(1)=DFLAG, H HAS ONE OF THE FOLLOWING FORMS..
//
//    DFLAG=-1.D0     DFLAG=0.D0        DFLAG=1.D0     DFLAG=-2.D0
//
//      (DH11  DH12)    (1.D0  DH12)    (DH11  1.D0)    (1.D0  0.D0)
//    H=(          )    (          )    (          )    (          )
//      (DH21  DH22),   (DH21  1.D0),   (-1.D0 DH22),   (0.D0  1.D0).
//    SEE DROTMG FOR A DESCRIPTION OF DATA STORAGE IN DPARAM.
func (v *Vector) Rotm(n int, dx *Vector, incx, inc int, dparam *DrotMatrix) {
	var dh11, dh12, dh21, dh22, w, z float64
	var dflag, i int

	dflag = dparam.Flag
	if n <= 0 || (dflag+2 == 0) {
		return
	}
	if incx == inc && incx > 0 {

		if dflag < 0 {
			dh11 = dparam.H11
			dh12 = dparam.H12
			dh21 = dparam.H21
			dh22 = dparam.H22
			for _, i = range dx.Iter(n, incx) {
				w = dx.Get(i)
				z = v.Get(i)
				dx.Set(i, w*dh11+z*dh12)
				v.Set(i, w*dh21+z*dh22)
			}
		} else if dflag == 0 {
			dh12 = dparam.H12
			dh21 = dparam.H21
			for _, i = range dx.Iter(n, incx) {
				w = dx.Get(i)
				z = v.Get(i)
				dx.Set(i, w+z*dh12)
				v.Set(i, w*dh21+z)
			}
		} else {
			dh11 = dparam.H11
			dh22 = dparam.H22
			for _, i = range dx.Iter(n, incx) {
				w = dx.Get(i)
				z = v.Get(i)
				dx.Set(i, w*dh11+z)
				v.Set(i, -w+dh22*z)
			}
		}
	} else {
		kx := dx.Iter(n, incx)
		ky := v.Iter(n, inc)

		if dflag < 0 {
			dh11 = dparam.H11
			dh12 = dparam.H12
			dh21 = dparam.H21
			dh22 = dparam.H22
			for i = 0; i < n; i++ {
				w = dx.Get(kx[i])
				z = v.Get(ky[i])
				dx.Set(kx[i], w*dh11+z*dh12)
				v.Set(ky[i], w*dh21+z*dh22)
			}
		} else if dflag == 0 {
			dh12 = dparam.H12
			dh21 = dparam.H21
			for i = 0; i < n; i++ {
				w = dx.Get(kx[i])
				z = v.Get(ky[i])
				dx.Set(kx[i], w+z*dh12)
				v.Set(ky[i], w*dh21+z)
			}
		} else {
			dh11 = dparam.H11
			dh22 = dparam.H22
			for i = 0; i < n; i++ {
				w = dx.Get(kx[i])
				z = v.Get(ky[i])
				dx.Set(kx[i], w*dh11+z)
				v.Set(ky[i], -w+dh22*z)
			}
		}
	}
	return
}

// Scal scales a vector by a constant
func (v *Vector) Scal(n int, da float64, inc int) {
	if n <= 0 || inc <= 0 {
		return
	}

	for _, i := range v.Iter(n, inc) {
		v.Set(i, da*v.Get(i))
	}
}

// Swap interchanges two vectors
func (v *Vector) Swap(n int, dx *Vector, incx, inc int) {
	if n <= 0 {
		return
	}

	var dtemp float64

	ix := dx.Iter(n, incx)
	iy := v.Iter(n, inc)

	for i := 0; i < n; i++ {
		dtemp = dx.Get(ix[i])
		dx.Set(ix[i], v.Get(iy[i]))
		v.Set(iy[i], dtemp)
	}
}

// Iamax finds the index of the first element having maximum absolute value
func (v *Vector) Iamax(n, inc int) (iamaxReturn int) {
	var dmax float64

	if n < 1 || inc <= 0 {
		return 0
	}
	if n == 1 {
		return 1
	}

	iamaxReturn = 1
	dmax = v.GetMag(0)
	for i, ix := range v.Iter(n, inc) {
		if v.GetMag(ix) > dmax {
			iamaxReturn = i + 1
			dmax = v.GetMag(ix)
		}
	}

	return
}

// Gemv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n matrix.
func (v *Vector) Gemv(trans MatTrans, m, n int, alpha float64, a *Matrix, x *Vector, incx int, beta float64, inc int) (err error) {
	var one, temp, zero float64
	var i, ix, iy, j, jx, jy, lenx, leny int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if m == 0 || n == 0 || (alpha == zero && beta == one) {
		return
	}

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == NoTrans {
		lenx = n
		leny = m
	} else {
		lenx = m
		leny = n
	}

	xiter := x.Iter(lenx, incx)
	yiter := v.Iter(leny, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy = range yiter {
				v.Set(iy, zero)
			}
		} else {
			for _, iy = range yiter {
				v.Set(iy, beta*v.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	if trans == NoTrans {
		//        Form  y := alpha*A*x + y.
		for j, jx = range xiter {
			temp = alpha * x.Get(jx)
			for i, iy = range yiter {
				v.Set(iy, v.Get(iy)+temp*a.Get(i, j))
			}
		}
	} else {
		for j, jy = range yiter {
			temp = zero
			for i, ix = range xiter {
				temp += a.Get(i, j) * x.Get(ix)
			}
			v.Set(jy, v.Get(jy)+alpha*temp)
		}
	}

	return
}

// Gbmv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n band matrix, with kl sub-diagonals and ku super-diagonals.
func (v *Vector) Gbmv(trans MatTrans, m, n, kl, ku int, alpha float64, a *Matrix, x *Vector, incx int, beta float64, inc int) (err error) {
	var one, temp, zero float64
	var i, j, k, kup1, lenx, leny int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if kl < 0 {
		err = fmt.Errorf("kl invalid: %v", kl)
	} else if ku < 0 {
		err = fmt.Errorf("ku invalid: %v", ku)
	} else if a.Rows < (kl + ku + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, kl+ku+1)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if m == 0 || n == 0 || (alpha == zero && beta == one) {
		return
	}

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == NoTrans {
		lenx = n
		leny = m
	} else {
		lenx = m
		leny = n
	}
	xiter := x.Iter(lenx, incx)
	yiter := v.Iter(leny, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the band part of A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, i = range v.Iter(leny, inc) {
				v.Set(i, zero)
			}
		} else {
			for _, i = range v.Iter(leny, inc) {
				v.Set(i, beta*v.Get(i))
			}
		}
	}
	if alpha == zero {
		return
	}
	kup1 = ku + 1
	if trans == NoTrans {
		//        Form  y := alpha*A*x + y.
		for j = 0; j < n; j++ {
			temp = alpha * x.Get(xiter[j])
			k = kup1 - (j + 1)
			for i = max(0, j-ku); i < min(m, (j+1)+kl); i++ {
				v.Set(yiter[i], v.Get(yiter[i])+temp*a.Get(k+i, j))
			}
		}
	} else {
		//        Form  y := alpha*A**T*x + y.
		for j = 0; j < n; j++ {
			temp = zero
			k = kup1 - (j + 1)
			for i = max(0, j-ku); i < min(m, (j+1)+kl); i++ {
				temp += a.Get(k+i, j) * x.Get(xiter[i])
			}
			v.Set(yiter[j], v.Get(yiter[j])+alpha*temp)
		}
	}

	return
}

// Symv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric matrix.
func (v *Vector) Symv(uplo MatUplo, n int, alpha float64, a *Matrix, x *Vector, incx int, beta float64, inc int) (err error) {
	var one, temp1, temp2, zero float64
	var i, j int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	xiter := x.Iter(n, incx)
	yiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy := range yiter {
				v.Set(iy, zero)
			}
		} else {
			for _, iy := range yiter {
				v.Set(iy, beta*v.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	if uplo == Upper {
		//        Form  y  when A is stored in upper triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			for i = 0; i < j; i++ {
				v.Set(yiter[i], v.Get(yiter[i])+temp1*a.Get(i, j))
				temp2 += a.Get(i, j) * x.Get(xiter[i])
			}
			v.Set(yiter[j], v.Get(yiter[j])+temp1*a.Get(j, j)+alpha*temp2)
		}
	} else {
		//        Form  y  when A is stored in lower triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			v.Set(yiter[j], v.Get(yiter[j])+temp1*a.Get(j, j))
			for i = j + 1; i < n; i++ {
				v.Set(yiter[i], v.Get(yiter[i])+temp1*a.Get(i, j))
				temp2 += a.Get(i, j) * x.Get(xiter[i])
			}
			v.Set(yiter[j], v.Get(yiter[j])+alpha*temp2)
		}
	}

	return
}

// Sbmv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric band matrix, with k super-diagonals.
func (v *Vector) Sbmv(uplo MatUplo, n, k int, alpha float64, a *Matrix, x *Vector, incx int, beta float64, inc int) (err error) {
	var one, temp1, temp2, zero float64
	var i, j, kplus1, l int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < (k + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	xiter := x.Iter(n, incx)
	yiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of the array A
	//     are accessed sequentially with one pass through A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, i = range yiter {
				v.Set(i, zero)
			}
		} else {
			for _, i = range yiter {
				v.Set(i, beta*v.Get(i))
			}
		}
	}
	if alpha == zero {
		return
	}
	if uplo == Upper {
		//        Form  y  when upper triangle of A is stored.
		kplus1 = k + 1
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			l = kplus1 - (j + 1)
			for i = max(0, j-k); i < j; i++ {
				v.Set(yiter[i], v.Get(yiter[i])+temp1*a.Get(l+i, j))
				temp2 += a.Get(l+i, j) * x.Get(xiter[i])
			}
			v.Set(yiter[j], v.Get(yiter[j])+temp1*a.Get(kplus1-1, j)+alpha*temp2)
		}
	} else {
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			v.Set(yiter[j], v.Get(yiter[j])+temp1*a.Get(0, j))
			l = 1 - (j + 1)
			for i = j + 1; i < min(n, j+1+k); i++ {
				v.Set(yiter[i], v.Get(yiter[i])+temp1*a.Get(l+i, j))
				temp2 += a.Get(l+i, j) * x.Get(xiter[i])
			}
			v.Set(yiter[j], v.Get(yiter[j])+alpha*temp2)
		}
	}

	return
}

// Spmv performs the matrix-vector operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n symmetric matrix, supplied in packed form.
func (v *Vector) Spmv(uplo MatUplo, n int, alpha float64, ap, x *Vector, incx int, beta float64, inc int) (err error) {
	var one, temp1, temp2, zero float64
	var i, ix, iy, j, k, kk int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	xiter := x.Iter(n, incx)
	yiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, i = range yiter {
				v.Set(i, zero)
			}
		} else {
			for _, i = range yiter {
				v.Set(i, beta*v.Get(i))
			}
		}
	}
	if alpha == zero {
		return
	}
	kk = 1
	if uplo == Upper {
		//        Form  y  when AP contains the upper triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			for k, ix, iy = kk-1, xiter[0], yiter[0]; k < kk+j-1; k, ix, iy = k+1, ix+incx, iy+inc {
				v.Set(iy, v.Get(iy)+temp1*ap.Get(k))
				temp2 += ap.Get(k) * x.Get(ix)
			}
			v.Set(yiter[j], v.Get(yiter[j])+temp1*ap.Get(kk+j-1)+alpha*temp2)
			kk += j + 1
		}
	} else {
		//        Form  y  when AP contains the lower triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			v.Set(yiter[j], v.Get(yiter[j])+temp1*ap.Get(kk-1))
			for k, ix, iy = kk, xiter[j]+incx, yiter[j]+inc; k < kk+n-(j+1); k, ix, iy = k+1, ix+incx, iy+inc {
				v.Set(iy, v.Get(iy)+temp1*ap.Get(k))
				temp2 += ap.Get(k) * x.Get(ix)
			}
			v.Set(yiter[j], v.Get(yiter[j])+alpha*temp2)
			kk += n - (j + 1) + 1
		}
	}

	return
}

// Trmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix.
func (v *Vector) Trmv(uplo MatUplo, trans MatTrans, diag MatDiag, n int, a *Matrix, inc int) (err error) {
	var nounit bool
	var temp, zero float64
	var i, j int

	zero = 0.0
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := A*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					for i = 0; i < j; i++ {
						v.Set(xiter[i], v.Get(xiter[i])+temp*a.Get(i, j))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*a.Get(j, j))
					}
				}
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					for i = n - 1; i >= j+1; i-- {
						v.Set(xiter[i], v.Get(xiter[i])+temp*a.Get(i, j))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*a.Get(j, j))
					}
				}
			}
		}
	} else {
		//        Form  x := A**T*x.
		if uplo == Upper {
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				if nounit {
					temp *= a.Get(j, j)
				}
				for i = j - 1; i >= 0; i-- {
					temp += a.Get(i, j) * v.Get(xiter[i])
				}
				v.Set(xiter[j], temp)
			}
		} else {
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				if nounit {
					temp *= a.Get(j, j)
				}
				for i = j + 1; i < n; i++ {
					temp += a.Get(i, j) * v.Get(xiter[i])
				}
				v.Set(xiter[j], temp)
			}
		}
	}

	return
}

// Tbmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular band matrix, with ( k + 1 ) diagonals.
func (v *Vector) Tbmv(uplo MatUplo, trans MatTrans, diag MatDiag, n, k int, a *Matrix, inc int) (err error) {
	var nounit bool
	var temp, zero float64
	var i, j, kplus1, l int

	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < (k + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX   too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//         Form  x := A*x.
		if uplo == Upper {
			kplus1 = k + 1
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					l = kplus1 - (j + 1)
					for i = max(0, j-k); i < j; i++ {
						v.Set(xiter[i], v.Get(xiter[i])+temp*a.Get(l+i, j))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*a.Get(kplus1-1, j))
					}
				}
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					l = 1 - (j + 1)
					for i = min(n-1, j+k); i > j; i-- {
						v.Set(xiter[i], v.Get(xiter[i])+temp*a.Get(l+i, j))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*a.Get(0, j))
					}
				}
			}
		}
	} else {
		//        Form  x := A**T*x.
		if uplo == Upper {
			kplus1 = k + 1
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				l = kplus1 - (j + 1)
				if nounit {
					temp *= a.Get(kplus1-1, j)
				}
				for i = j - 1; i >= max(0, j-k); i-- {
					temp += a.Get(l+i, j) * v.Get(xiter[i])
				}
				v.Set(xiter[j], temp)
			}
		} else {
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				l = 1 - (j + 1)
				if nounit {
					temp *= a.Get(0, j)
				}
				for i = j + 1; i < min(n, j+k+1); i++ {
					temp += a.Get(l+i, j) * v.Get(xiter[i])
				}
				v.Set(xiter[j], temp)
			}
		}
	}

	return
}

// Tpmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix, supplied in packed form.
func (v *Vector) Tpmv(uplo MatUplo, trans MatTrans, diag MatDiag, n int, ap *Vector, inc int) (err error) {
	var nounit bool
	var temp, zero float64
	var i, j, k, kk int

	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == NoTrans {
		//        Form  x:= A*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
						v.Set(xiter[i], v.Get(xiter[i])+temp*ap.Get(k))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*ap.Get(kk+j))
					}
				}
				kk += j + 1
			}
		} else {
			kk = (n * (n + 1)) / 2
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					for i, k = n-1, kk-1; k > kk-n+j; i, k = i-1, k-1 {
						v.Set(xiter[i], v.Get(xiter[i])+temp*ap.Get(k))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*ap.Get(kk-n+j))
					}
				}
				kk -= (n - j)
			}
		}
	} else {
		//        Form  x := A**T*x.
		if uplo == Upper {
			kk = (n * (n + 1)) / 2
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				if nounit {
					temp *= ap.Get(kk - 1)
				}
				for i, k = j-1, kk-2; k >= kk-j-1; i, k = i-1, k-1 {
					temp += ap.Get(k) * v.Get(xiter[i])
				}
				v.Set(xiter[j], temp)
				kk -= (j + 1)
			}
		} else {
			kk = 1
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				if nounit {
					temp *= ap.Get(kk - 1)
				}
				for i, k = j+1, kk; k < kk+n-(j+1); i, k = i+1, k+1 {
					temp += ap.Get(k) * v.Get(xiter[i])
				}
				v.Set(xiter[j], temp)
				kk += (n - j)
			}
		}
	}

	return
}

// Trsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular matrix.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (v *Vector) Trsv(uplo MatUplo, trans MatTrans, diag MatDiag, n int, a *Matrix, inc int) (err error) {
	var nounit bool
	var temp, zero float64
	var i, j int

	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/a.Get(j, j))
					}
					temp = v.Get(xiter[j])
					for i = j - 1; i >= 0; i-- {
						v.Set(xiter[i], v.Get(xiter[i])-temp*a.Get(i, j))
					}
				}
			}
		} else {
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/a.Get(j, j))
					}
					temp = v.Get(xiter[j])
					for i = j + 1; i < n; i++ {
						v.Set(xiter[i], v.Get(xiter[i])-temp*a.Get(i, j))
					}
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				for i = 0; i < j; i++ {
					temp -= a.Get(i, j) * v.Get(xiter[i])
				}
				if nounit {
					temp /= a.Get(j, j)
				}
				v.Set(xiter[j], temp)
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				for i = n - 1; i >= j+1; i-- {
					temp -= a.Get(i, j) * v.Get(xiter[i])
				}
				if nounit {
					temp /= a.Get(j, j)
				}
				v.Set(xiter[j], temp)
			}
		}
	}

	return
}

// Tbsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular band matrix, with ( k + 1 )
// diagonals.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (v *Vector) Tbsv(uplo MatUplo, trans MatTrans, diag MatDiag, n, k int, a *Matrix, inc int) (err error) {
	var nounit bool
	var temp, zero float64
	var i, j, kplus1, l int

	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < (k + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed by sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			kplus1 = k
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					l = kplus1 - j
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/a.Get(kplus1, j))
					}
					temp = v.Get(xiter[j])
					for i = j - 1; i >= max(0, j-k); i-- {
						v.Set(xiter[i], v.Get(xiter[i])-temp*a.Get(l+i, j))
					}
				}
			}
		} else {
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					l = -j
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/a.Get(0, j))
					}
					temp = v.Get(xiter[j])
					for i = j + 1; i < min(n, j+k+1); i++ {
						v.Set(xiter[i], v.Get(xiter[i])-temp*a.Get(l+i, j))
					}
				}
			}
		}
	} else {
		//        Form  x := inv( A**T)*x.
		if uplo == Upper {
			kplus1 = k
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				l = kplus1 - j
				for i = max(0, j-k); i < j; i++ {
					temp -= a.Get(l+i, j) * v.Get(xiter[i])
				}
				if nounit {
					temp /= a.Get(kplus1, j)
				}
				v.Set(xiter[j], temp)
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				l = 1 - j - 1
				for i = min(n-1, j+k); i > j; i-- {
					temp -= a.Get(l+i, j) * v.Get(xiter[i])
				}
				if nounit {
					temp /= a.Get(0, j)
				}
				v.Set(xiter[j], temp)
			}
		}
	}

	return
}

// Tpsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular matrix, supplied in packed form.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (v *Vector) Tpsv(uplo MatUplo, trans MatTrans, diag MatDiag, n int, ap *Vector, inc int) (err error) {
	var nounit bool
	var temp, zero float64
	var i, j, k, kk int

	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			kk = (n*(n+1))/2 - 1
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/ap.Get(kk))
					}
					temp = v.Get(xiter[j])
					for i, k = j-1, kk-1; k >= kk-j; i, k = i-1, k-1 {
						v.Set(xiter[i], v.Get(xiter[i])-temp*ap.Get(k))
					}
				}
				kk -= (j + 1)
			}
		} else {
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/ap.Get(kk))
					}
					temp = v.Get(xiter[j])
					for i, k = j+1, kk+1; k <= kk+n-j-1; i, k = i+1, k+1 {
						v.Set(xiter[i], v.Get(xiter[i])-temp*ap.Get(k))
					}
				}
				kk += (n - j)
			}
		}
	} else {
		//        Form  x := inv( A**T )*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
					temp -= ap.Get(k) * v.Get(xiter[i])
				}
				if nounit {
					temp /= ap.Get(kk + j)
				}
				v.Set(xiter[j], temp)
				kk += j + 1
			}
		} else {
			kk = (n * (n + 1)) / 2
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				for i, k = n-1, kk-1; k >= kk-n+j+1; i, k = i-1, k-1 {
					temp -= ap.Get(k) * v.Get(xiter[i])
				}
				if nounit {
					temp /= ap.Get(kk - n + j)
				}
				v.Set(xiter[j], temp)
				kk -= (n - j)
			}
		}
	}

	return
}

// Spr performs the symmetric rank 1 operation
//
//    A := alpha*x*x**T + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n symmetric matrix, supplied in packed form.
func (v *Vector) Spr(uplo MatUplo, n int, alpha float64, x *Vector, incx int) (err error) {
	var temp, zero float64
	var i, j, k, kk int

	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || (alpha == zero) {
		return
	}

	//     Set the start point in X if the increment is not unity.
	xiter := x.Iter(n, incx)

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	if uplo == Upper {
		for j = 0; j < n; j++ {
			if x.Get(xiter[j]) != zero {
				temp = alpha * x.Get(xiter[j])
				for i, k = 0, kk; k <= kk+j; i, k = i+1, k+1 {
					v.Set(k, v.Get(k)+x.Get(xiter[i])*temp)
				}
			}
			kk += j + 1
		}
	} else {
		for j = 0; j < n; j++ {
			if x.Get(xiter[j]) != zero {
				temp = alpha * x.Get(xiter[j])
				for i, k = j, kk; k < kk+n-j; i, k = i+1, k+1 {
					v.Set(k, v.Get(k)+x.Get(xiter[i])*temp)
				}
			}
			kk += n - j
		}
	}

	return
}

// Spr2 performs the symmetric rank 2 operation
//
//    A := alpha*x*y**T + alpha*y*x**T + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an
// n by n symmetric matrix, supplied in packed form.
func (v *Vector) Spr2(uplo MatUplo, n int, alpha float64, x *Vector, incx int, y *Vector, incy int) (err error) {
	var temp1, temp2, zero float64
	var i, j, k, kk int

	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || (alpha == zero) {
		return
	}

	//     Set up the start points in X and Y if the increments are not both
	//     unity.
	xiter := x.Iter(n, incx)
	yiter := y.Iter(n, incy)

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	kk = 1
	if uplo == Upper {
		//        Form  A  when upper triangle is stored in AP.
		for j = 0; j < n; j++ {
			if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
				temp1 = alpha * y.Get(yiter[j])
				temp2 = alpha * x.Get(xiter[j])
				for i, k = 0, kk-1; k < kk+j; i, k = i+1, k+1 {
					v.Set(k, v.Get(k)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
				}
			}
			kk += j + 1
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		for j = 0; j < n; j++ {
			if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
				temp1 = alpha * y.Get(yiter[j])
				temp2 = alpha * x.Get(xiter[j])
				for i, k = j, kk-1; k < kk+n-j-1; i, k = i+1, k+1 {
					v.Set(k, v.Get(k)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
				}
			}
			kk += n - j
		}
	}

	return
}

func VectorFactory() func(int) *Vector {
	return func(n int) *Vector {
		return &Vector{size: n, data: make([]float64, n, 2*n)}
	}
}

// VectorDataFactory creates a new Vectorer with same data memory location
func VectorDataFactory() func([]float64) *Vector {
	return func(d []float64) *Vector {
		return &Vector{size: len(d), data: d}
	}
}

// func VectorDataFactory() func([]float64, ...int) *Vector {
// 	return func(d []float64, n ...int) *Vector {
// 		size := len(d)
// 		if n == nil {
// 			newd := make([]float64, size, 2*size)
// 			copy(newd, d)
// 			return &Vector{size: size, inc: 1, Data: newd}
// 		}
// 		if n[0] == 0 {
// 			panic("VectorDataFactory: cannot have an increment of 0")
// 		}
// 		inc := n[0]
// 		newd := make([]float64, size*abs(inc), 2*size*abs(inc))
// 		if inc < 0 {
// 			for i := 0; i < size; i++ {
// 				newd[(size-i)*abs(inc)-1] = d[i]
// 			}
// 		} else {
// 			for i := 0; i < size; i++ {
// 				newd[i*inc] = d[i]
// 			}
// 		}
// 		return &Vector{size: size, inc: inc, Data: newd}
// 	}
// }

type CVector struct {
	size int
	data []complex128
}

func (v *CVector) Append(x ...complex128) {
	for _, xx := range x {
		v.size++
		v.data = append(v.data, xx)
	}
}
func (v *CVector) Data() []complex128 {
	return v.data
}
func (v *CVector) Get(n int) complex128 {
	return v.data[n]
}
func (v *CVector) GetConj(n int) complex128 {
	return cmplx.Conj(v.data[n])
}
func (v *CVector) GetMag(n int) float64 {
	return cmplx.Abs(v.data[n])
}
func (v *CVector) GetArg(n int) float64 {
	return cmplx.Phase(v.data[n])
}
func (v *CVector) GetDeg(n int) float64 {
	return cmplx.Phase(v.data[n]) * 180 / math.Pi
}
func (v *CVector) GetRe(n int) float64 {
	return real(v.data[n])
}
func (v *CVector) GetIm(n int) float64 {
	return imag(v.data[n])
}
func (v *CVector) GetReCmplx(n int) complex128 {
	return complex(real(v.data[n]), 0)
}
func (v *CVector) GetImCmplx(n int) complex128 {
	return complex(0, imag(v.data[n]))
}
func (v *CVector) GetConjProd(n int) float64 {
	return real(cmplx.Conj(v.data[n]) * v.data[n])
}
func (v *CVector) GetPtr(n int) *complex128 {
	return &v.data[n]
}
func (v *CVector) Set(n int, x complex128) {
	v.data[n] = x
}
func (v *CVector) SetAll(x complex128) {
	for i := 0; i < v.size; i++ {
		v.Set(i, x)
	}
}
func (v *CVector) SetRe(n int, x float64) {
	v.data[n] = complex(x, 0)
}
func (v *CVector) SetIm(n int, x float64) {
	v.data[n] = complex(0, x)
}
func (v *CVector) SetReAll(x float64) {
	for i := range v.data {
		v.data[i] = complex(x, 0)
	}
}
func (v *CVector) SetImAll(x float64) {
	for i := range v.data {
		v.data[i] = complex(0, x)
	}
}
func (v *CVector) Size() int {
	return v.size
}
func (v *CVector) DeepCopy() *CVector {
	result := &CVector{size: v.size, data: make([]complex128, v.size, 2*v.size)}

	copy(result.data, v.data)

	return result
}
func (v *CVector) Off(idx int) *CVector {
	return &CVector{size: v.size - idx, data: v.data[idx:]}
}
func (v *CVector) Iter(n, inc int) []int {
	iter := make([]int, n)

	for i := 0; i < n; i++ {
		iter[i] = getVIdx(v.size, inc, i)
	}

	return iter
}

// CMatrix references same data in memory
func (v *CVector) CMatrix(r int, opts MatOpts) *CMatrix {
	var c int

	mf := CMatrixFactory()
	if r == 0 {
		c = len(v.data)
	} else {
		c = len(v.data) / r
	}
	mat := mf(r, c, opts)
	mat.Data = v.data
	return mat
}

// Asum takes the sum of the (|Re(.)| + |Im(.)|)'s of a complex vector and
//    returns a single precision result.
func (v *CVector) Asum(n, inc int) (dzasumReturn float64) {
	if n <= 0 || inc <= 0 {
		return 0
	}

	for _, ix := range v.Iter(n, inc) {
		dzasumReturn += dcabs1(v.Get(ix))
	}

	return
}

// Nrm2 returns the euclidean norm of a vector via the function
// name, so that
//
//    DZNRM2 := sqrt( x**H*x )
func (v *CVector) Nrm2(n, inc int) float64 {
	var scale, ssq, temp float64
	var ix int

	xiter := v.Iter(n, inc)
	ssq = 1
	//        The following loop is equivalent to this call to the LAPACK
	//        auxiliary routine:
	//        CALL ZLASSQ( N, X, INCX, SCALE, SSQ )
	for _, ix = range xiter {
		if real(v.Get(ix)) != 0 {
			temp = cmplx.Abs(v.GetReCmplx(ix))
			if scale < temp {
				ssq = 1 + ssq*math.Pow(scale/temp, 2)
				scale = temp
			} else {
				ssq += math.Pow(temp/scale, 2)
			}
		}
		if imag(v.Get(ix)) != 0 {
			temp = cmplx.Abs(v.GetImCmplx(ix))
			if scale < temp {
				ssq = 1 + ssq*math.Pow(scale/temp, 2)
				scale = temp
			} else {
				ssq += math.Pow(temp/scale, 2)
			}
		}
	}

	return scale * math.Sqrt(ssq)
}

// Iamax finds the index of the first element having maximum |Re(.)| + |Im(.)|
func (v *CVector) Iamax(n, inc int) (izamaxReturn int) {
	var dmax float64

	if n < 1 || inc <= 0 {
		return 0
	} else if n == 1 {
		return 1
	}

	//        code for increment not equal to 1
	izamaxReturn = 1
	dmax = dcabs1(v.Get(0))
	for i, ix := range v.Iter(n, inc) {
		if dcabs1(v.Get(ix)) > dmax {
			izamaxReturn = i + 1
			dmax = dcabs1(v.Get(ix))
		}
	}
	return
}

// Dscal scales a vector by a constant.
func (v *CVector) Dscal(n int, da float64, inc int) {
	if n <= 0 || inc < 0 {
		return
	}

	for _, ix := range v.Iter(n, inc) {
		v.Set(ix, complex(da, 0)*v.Get(ix))
	}
}

// Zscal scales a vector by a constant.
func (v *CVector) Scal(n int, za complex128, inc int) {
	if n <= 0 || inc < 0 {
		return
	}

	for _, ix := range v.Iter(n, inc) {
		v.Set(ix, za*v.Get(ix))
	}
}

// Axpy constant times a vector plus a vector.
func (v *CVector) Axpy(n int, za complex128, zx *CVector, incx, inc int) {
	if n <= 0 || dcabs1(za) == 0.0 {
		return
	}

	ix := zx.Iter(n, incx)
	iy := v.Iter(n, inc)

	for i := 0; i < n; i++ {
		v.Set(iy[i], v.Get(iy[i])+za*zx.Get(ix[i]))
	}
}

// Copy copies a vector, x, to a vector, y.
func (v *CVector) Copy(n int, zx *CVector, incx, inc int) {
	if n <= 0 {
		return
	}

	ix := zx.Iter(n, incx)
	iy := v.Iter(n, inc)

	for i := 0; i < n; i++ {
		v.Set(iy[i], zx.Get(ix[i]))
	}
}

// Dotc forms the dot product of two complex vectors
//      ZDOTC = X^H * Y
func (v *CVector) Dotc(n int, zx *CVector, incx, inc int) (zdotcReturn complex128) {
	if n <= 0 {
		return
	}

	ix := zx.Iter(n, incx)
	iy := v.Iter(n, inc)

	for i := 0; i < n; i++ {
		zdotcReturn += zx.GetConj(ix[i]) * v.Get(iy[i])
	}

	return
}

// Dotu forms the dot product of two complex vectors
//      ZDOTU = X^T * Y
func (v *CVector) Dotu(n int, zx *CVector, incx, inc int) (zdotuReturn complex128) {
	if n <= 0 {
		return
	}

	ix := zx.Iter(n, incx)
	iy := v.Iter(n, inc)

	for i := 0; i < n; i++ {
		zdotuReturn += zx.Get(ix[i]) * v.Get(iy[i])
	}

	return
}

// Swap interchanges two vectors.
func (v *CVector) Swap(n int, zx *CVector, incx, inc int) {
	if n <= 0 {
		return
	}

	var ztemp complex128

	ix := zx.Iter(n, incx)
	iy := v.Iter(n, inc)

	for i := 0; i < n; i++ {
		ztemp = zx.Get(ix[i])
		zx.Set(ix[i], v.Get(iy[i]))
		v.Set(iy[i], ztemp)
	}
}

// Drot Applies a plane rotation, where the cos and sin (c and s) are real
// and the vectors cx and cy are complex.
// jack dongarra, linpack, 3/11/78.
func (v *CVector) Drot(n int, cx *CVector, incx, inc int, c, s float64) {
	if n <= 0 {
		return
	}

	var ctemp complex128

	ix := cx.Iter(n, incx)
	iy := v.Iter(n, inc)

	for i := 0; i < n; i++ {
		ctemp = complex(c, 0)*cx.Get(ix[i]) + complex(s, 0)*v.Get(iy[i])
		v.Set(iy[i], complex(c, 0)*v.Get(iy[i])-complex(s, 0)*cx.Get(ix[i]))
		cx.Set(ix[i], ctemp)
	}
}

// Gbmv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
//
//    y := alpha*A**H*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n band matrix, with kl sub-diagonals and ku super-diagonals.
func (v *CVector) Gbmv(trans MatTrans, m, n, kl, ku int, alpha complex128, a *CMatrix, x *CVector, incx int, beta complex128, inc int) (err error) {
	var noconj bool
	var one, temp, zero complex128
	var i, info, iy, j, k, kup1, lenx, leny int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	info = 0
	if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if kl < 0 {
		err = fmt.Errorf("kl invalid: %v", kl)
	} else if ku < 0 {
		err = fmt.Errorf("ku invalid: %v", ku)
	} else if a.Rows < (kl + ku + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, (kl + ku + 1))
	}
	if info != 0 {
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	noconj = trans == Trans

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == NoTrans {
		lenx = n
		leny = m
	} else {
		lenx = m
		leny = n
	}

	xiter := x.Iter(lenx, incx)
	yiter := v.Iter(leny, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the band part of A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy = range yiter {
				v.Set(iy, zero)
			}
		} else {
			for _, iy = range yiter {
				v.Set(iy, beta*v.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	kup1 = ku + 1
	if trans == NoTrans {
		//        Form  y := alpha*A*x + y.
		for j = 0; j < n; j++ {
			temp = alpha * x.Get(xiter[j])
			k = kup1 - j - 1
			for i = max(0, j-ku); i < min(m, j+kl+1); i++ {
				v.Set(yiter[i], v.Get(yiter[i])+temp*a.Get(k+i, j))
			}
		}
	} else {
		//        Form  y := alpha*A**T*x + y  or  y := alpha*A**H*x + y.
		for j = 0; j < n; j++ {
			temp = zero
			k = kup1 - j - 1
			if noconj {
				for i = max(0, j-ku); i < min(m, j+kl+1); i++ {
					temp += a.Get(k+i, j) * x.Get(xiter[i])
				}
			} else {
				for i = max(0, j-ku); i < min(m, j+kl+1); i++ {
					temp += a.GetConj(k+i, j) * x.Get(xiter[i])
				}
			}
			v.Set(yiter[j], v.Get(yiter[j])+alpha*temp)
		}
	}

	return
}

// Gemv performs one of the matrix-vector operations
//
//    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
//
//    y := alpha*A**H*x + beta*y,
//
// where alpha and beta are scalars, x and y are vectors and A is an
// m by n matrix.
func (v *CVector) Gemv(trans MatTrans, m, n int, alpha complex128, a *CMatrix, x *CVector, incx int, beta complex128, inc int) (err error) {
	var noconj bool
	var one, temp, zero complex128
	var i, ix, iy, j, jx, jy, lenx, leny int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	noconj = trans == Trans

	//     Set  LENX  and  LENY, the lengths of the vectors x and y, and set
	//     up the start points in  X  and  Y.
	if trans == NoTrans {
		lenx = n
		leny = m
	} else {
		lenx = m
		leny = n
	}

	xiter := x.Iter(lenx, incx)
	yiter := v.Iter(leny, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy = range yiter {
				v.Set(iy, zero)
			}
		} else {
			for _, iy = range yiter {
				v.Set(iy, beta*v.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	if trans == NoTrans {
		//        Form  y := alpha*A*x + y.
		for j, jx = range xiter {
			temp = alpha * x.Get(jx)
			for i, iy = range yiter {
				v.Set(iy, v.Get(iy)+temp*a.Get(i, j))
			}
		}
	} else {
		//        Form  y := alpha*A**T*x + y  or  y := alpha*A**H*x + y.
		for j, jy = range yiter {
			temp = zero
			if noconj {
				for i, ix = range xiter {
					temp += a.Get(i, j) * x.Get(ix)
				}
			} else {
				for i, ix = range xiter {
					temp += a.GetConj(i, j) * x.Get(ix)
				}
			}
			v.Set(jy, v.Get(jy)+alpha*temp)
		}
	}

	return
}

// Hbmv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n hermitian band matrix, with k super-diagonals.
func (v *CVector) Hbmv(uplo MatUplo, n, k int, alpha complex128, a *CMatrix, x *CVector, incx int, beta complex128, inc int) (err error) {
	var one, temp1, temp2, zero complex128
	var i, iy, j, kplus1, l int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < (k + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     Set up the start points in  X  and  Y.
	xiter := x.Iter(n, incx)
	yiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of the array A
	//     are accessed sequentially with one pass through A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy = range yiter {
				v.Set(iy, zero)
			}
		} else {
			for _, iy = range yiter {
				v.Set(iy, beta*v.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	if uplo == Upper {
		//        Form  y  when upper triangle of A is stored.
		kplus1 = k
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			l = kplus1 - j
			for i = max(0, j-k); i < j; i++ {
				v.Set(yiter[i], v.Get(yiter[i])+temp1*a.Get(l+i, j))
				temp2 += a.GetConj(l+i, j) * x.Get(xiter[i])
			}
			v.Set(yiter[j], v.Get(yiter[j])+temp1*a.GetReCmplx(kplus1, j)+alpha*temp2)
		}
	} else {
		//        Form  y  when lower triangle of A is stored.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			v.Set(yiter[j], v.Get(yiter[j])+temp1*a.GetReCmplx(0, j))
			l = 1 - j - 1
			for i = j + 1; i < min(n, j+k+1); i++ {
				v.Set(yiter[i], v.Get(yiter[i])+temp1*a.Get(l+i, j))
				temp2 += a.GetConj(l+i, j) * x.Get(xiter[i])
			}
			v.Set(yiter[j], v.Get(yiter[j])+alpha*temp2)
		}
	}

	return
}

// Hemv performs the matrix-vector  operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n hermitian matrix.
func (v *CVector) Hemv(uplo MatUplo, n int, alpha complex128, a *CMatrix, x *CVector, incx int, beta complex128, inc int) (err error) {
	var one, temp1, temp2, zero complex128
	var i, iy, j int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if uplo != Upper && uplo != Lower {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     Set up the start points in  X  and  Y.
	xiter := x.Iter(n, incx)
	yiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy = range yiter {
				v.Set(iy, zero)
			}
		} else {
			for _, iy = range yiter {
				v.Set(iy, beta*v.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	if uplo == Upper {
		//        Form  y  when A is stored in upper triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			for i = 0; i < j; i++ {
				v.Set(yiter[i], v.Get(yiter[i])+temp1*a.Get(i, j))
				temp2 += a.GetConj(i, j) * x.Get(xiter[i])
			}
			v.Set(yiter[j], v.Get(yiter[j])+temp1*a.GetReCmplx(j, j)+alpha*temp2)
		}
	} else {
		//        Form  y  when A is stored in lower triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			v.Set(yiter[j], v.Get(yiter[j])+temp1*a.GetReCmplx(j, j))
			for i = j + 1; i < n; i++ {
				v.Set(yiter[i], v.Get(yiter[i])+temp1*a.Get(i, j))
				temp2 += a.GetConj(i, j) * x.Get(xiter[i])
			}
			v.Set(yiter[j], v.Get(yiter[j])+alpha*temp2)
		}
	}

	return
}

// Hpmv performs the matrix-vector operation
//
//    y := alpha*A*x + beta*y,
//
// where alpha and beta are scalars, x and y are n element vectors and
// A is an n by n hermitian matrix, supplied in packed form.
func (v *CVector) Hpmv(uplo MatUplo, n int, alpha complex128, ap, x *CVector, incx int, beta complex128, inc int) (err error) {
	var one, temp1, temp2, zero complex128
	var i, iy, j, k, kk int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     Set up the start points in  X  and  Y.
	xiter := x.Iter(n, incx)
	yiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	//
	//     First form  y := beta*y.
	if beta != one {
		if beta == zero {
			for _, iy = range yiter {
				v.Set(iy, zero)
			}
		} else {
			for _, iy = range yiter {
				v.Set(iy, beta*v.Get(iy))
			}
		}
	}
	if alpha == zero {
		return
	}
	if uplo == Upper {
		//        Form  y  when AP contains the upper triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
				v.Set(yiter[i], v.Get(yiter[i])+temp1*ap.Get(k))
				temp2 += ap.GetConj(k) * x.Get(xiter[i])
			}
			v.Set(yiter[j], v.Get(yiter[j])+temp1*ap.GetReCmplx(kk+j)+alpha*temp2)
			kk += j + 1
		}
	} else {
		//        Form  y  when AP contains the lower triangle.
		for j = 0; j < n; j++ {
			temp1 = alpha * x.Get(xiter[j])
			temp2 = zero
			v.Set(yiter[j], v.Get(yiter[j])+temp1*ap.GetReCmplx(kk))
			for i, k = j+1, kk+1; k < kk+n-j; i, k = i+1, k+1 {
				v.Set(yiter[i], v.Get(yiter[i])+temp1*ap.Get(k))
				temp2 += ap.GetConj(k) * x.Get(xiter[i])
			}
			v.Set(yiter[j], v.Get(yiter[j])+alpha*temp2)
			kk += (n - j)
		}
	}

	return
}

// Tbmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular band matrix, with ( k + 1 ) diagonals.
func (v *CVector) Tbmv(uplo MatUplo, trans MatTrans, diag MatDiag, n, k int, a *CMatrix, inc int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j, kplus1, l int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < (k + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX   too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//         Form  x := A*x.
		if uplo == Upper {
			kplus1 = k + 1
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					l = kplus1 - j - 1
					for i = max(0, j-k); i < j; i++ {
						v.Set(xiter[i], v.Get(xiter[i])+temp*a.Get(l+i, j))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*a.Get(kplus1-1, j))
					}
				}
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					l = 1 - j - 1
					for i = min(n-1, j+k); i >= j+1; i-- {
						v.Set(xiter[i], v.Get(xiter[i])+temp*a.Get(l+i, j))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*a.Get(0, j))
					}
				}
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			kplus1 = k
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				l = kplus1 - j
				if noconj {
					if nounit {
						temp *= a.Get(kplus1, j)
					}
					for i = j - 1; i >= max(0, j-k); i-- {
						temp += a.Get(l+i, j) * v.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= a.GetConj(kplus1, j)
					}
					for i = j - 1; i >= max(0, j-k); i-- {
						temp += a.GetConj(l+i, j) * v.Get(xiter[i])
					}
				}
				v.Set(xiter[j], temp)
			}
		} else {
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				l = 1 - j - 1
				if noconj {
					if nounit {
						temp *= a.Get(0, j)
					}
					for i = j + 1; i < min(n, j+k+1); i++ {
						temp += a.Get(l+i, j) * v.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= a.GetConj(0, j)
					}
					for i = j + 1; i < min(n, j+k+1); i++ {
						temp += a.GetConj(l+i, j) * v.Get(xiter[i])
					}
				}
				v.Set(xiter[j], temp)
			}
		}
	}

	return
}

// Tbsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular band matrix, with ( k + 1 )
// diagonals.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (v *CVector) Tbsv(uplo MatUplo, trans MatTrans, diag MatDiag, n, k int, a *CMatrix, inc int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j, kplus1, l int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < (k + 1) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, k+1)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed by sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			kplus1 = k
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					l = kplus1 - j
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/a.Get(kplus1, j))
					}
					temp = v.Get(xiter[j])
					for i = j - 1; i >= max(0, j-k); i-- {
						v.Set(xiter[i], v.Get(xiter[i])-temp*a.Get(l+i, j))
					}
				}
			}
		} else {
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					l = 1 - j - 1
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/a.Get(0, j))
					}
					temp = v.Get(xiter[j])
					for i = j + 1; i < min(n, j+k+1); i++ {
						v.Set(xiter[i], v.Get(xiter[i])-temp*a.Get(l+i, j))
					}
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			kplus1 = k
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				l = kplus1 - j
				if noconj {
					for i = max(0, j-k); i < j; i++ {
						temp -= a.Get(l+i, j) * v.Get(xiter[i])
					}
					if nounit {
						temp /= a.Get(kplus1, j)
					}
				} else {
					for i = max(0, j-k); i < j; i++ {
						temp -= a.GetConj(l+i, j) * v.Get(xiter[i])
					}
					if nounit {
						temp /= a.GetConj(kplus1, j)
					}
				}
				v.Set(xiter[j], temp)
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				l = -j
				if noconj {
					for i = min(n-1, j+k); i >= j+1; i-- {
						temp -= a.Get(l+i, j) * v.Get(xiter[i])
					}
					if nounit {
						temp /= a.Get(0, j)
					}
				} else {
					for i = min(n-1, j+k); i >= j+1; i-- {
						temp -= a.GetConj(l+i, j) * v.Get(xiter[i])
					}
					if nounit {
						temp /= a.GetConj(0, j)
					}
				}
				v.Set(xiter[j], temp)
			}
		}
	}

	return
}

// Tpmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix, supplied in packed form.
func (v *CVector) Tpmv(uplo MatUplo, trans MatTrans, diag MatDiag, n int, ap *CVector, inc int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j, k, kk int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == NoTrans {
		//        Form  x:= A*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
						v.Set(xiter[i], v.Get(xiter[i])+temp*ap.Get(k))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*ap.Get(kk+j))
					}
				}
				kk += j + 1
			}
		} else {
			kk = (n * (n + 1)) / 2
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					for i, k = n-1, kk-1; k >= kk-n+j+1; i, k = i-1, k-1 {
						v.Set(xiter[i], v.Get(xiter[i])+temp*ap.Get(k))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*ap.Get(kk-n+j))
					}
				}
				kk -= (n - j)
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			kk = (n*(n+1))/2 - 1
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				if noconj {
					if nounit {
						temp *= ap.Get(kk)
					}
					for i, k = j-1, kk-1; k >= kk-j; i, k = i-1, k-1 {
						temp += ap.Get(k) * v.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= ap.GetConj(kk)
					}
					for i, k = j-1, kk-1; k >= kk-j; i, k = i-1, k-1 {
						temp += ap.GetConj(k) * v.Get(xiter[i])
					}
				}
				v.Set(xiter[j], temp)
				kk -= j + 1
			}
		} else {
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				if noconj {
					if nounit {
						temp *= ap.Get(kk)
					}
					for i, k = j+1, kk+1; k < kk+n-j; i, k = i+1, k+1 {
						temp += ap.Get(k) * v.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= ap.GetConj(kk)
					}
					for i, k = j+1, kk+1; k < kk+n-j; i, k = i+1, k+1 {
						temp += ap.GetConj(k) * v.Get(xiter[i])
					}
				}
				v.Set(xiter[j], temp)
				kk += (n - j)
			}
		}
	}

	return
}

// Tpsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular matrix, supplied in packed form.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (v *CVector) Tpsv(uplo MatUplo, trans MatTrans, diag MatDiag, n int, ap *CVector, inc int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j, k, kk int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of AP are
	//     accessed sequentially with one pass through AP.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			kk = (n*(n+1))/2 - 1
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/ap.Get(kk))
					}
					temp = v.Get(xiter[j])
					for i, k = j-1, kk-1; k >= kk-j; i, k = i-1, k-1 {
						v.Set(xiter[i], v.Get(xiter[i])-temp*ap.Get(k))
					}
				}
				kk -= j + 1
			}
		} else {
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/ap.Get(kk))
					}
					temp = v.Get(xiter[j])
					for i, k = j+1, kk+1; k < kk+n-j; i, k = i+1, k+1 {
						v.Set(xiter[i], v.Get(xiter[i])-temp*ap.Get(k))
					}
				}
				kk += (n - j)
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				if noconj {
					for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
						temp -= ap.Get(k) * v.Get(xiter[i])
					}
					if nounit {
						temp /= ap.Get(kk + j)
					}
				} else {
					for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
						temp -= ap.GetConj(k) * v.Get(xiter[i])
					}
					if nounit {
						temp /= ap.GetConj(kk + j)
					}
				}
				v.Set(xiter[j], temp)
				kk += j + 1
			}
		} else {
			kk = (n * (n + 1)) / 2
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				if noconj {
					for i, k = n-1, kk-1; k >= kk-n+j+1; i, k = i-1, k-1 {
						temp -= ap.Get(k) * v.Get(xiter[i])
					}
					if nounit {
						temp /= ap.Get(kk - n + j)
					}
				} else {
					for i, k = n-1, kk-1; k >= kk-n+j+1; i, k = i-1, k-1 {
						temp -= ap.GetConj(k) * v.Get(xiter[i])
					}
					if nounit {
						temp /= ap.GetConj(kk - n + j)
					}
				}
				v.Set(xiter[j], temp)
				kk -= (n - j)
			}
		}
	}

	return
}

// Trmv performs one of the matrix-vector operations
//
//    x := A*x,   or   x := A**T*x,   or   x := A**H*x,
//
// where x is an n element vector and  A is an n by n unit, or non-unit,
// upper or lower triangular matrix.
func (v *CVector) Trmv(uplo MatUplo, trans MatTrans, diag MatDiag, n int, a *CMatrix, inc int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := A*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					for i = 0; i < j; i++ {
						v.Set(xiter[i], v.Get(xiter[i])+temp*a.Get(i, j))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*a.Get(j, j))
					}
				}
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					temp = v.Get(xiter[j])
					for i = n - 1; i >= j+1; i-- {
						v.Set(xiter[i], v.Get(xiter[i])+temp*a.Get(i, j))
					}
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])*a.Get(j, j))
					}
				}
			}
		}
	} else {
		//        Form  x := A**T*x  or  x := A**H*x.
		if uplo == Upper {
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				if noconj {
					if nounit {
						temp *= a.Get(j, j)
					}
					for i = j - 1; i >= 0; i-- {
						temp += a.Get(i, j) * v.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= a.GetConj(j, j)
					}
					for i = j - 1; i >= 0; i-- {
						temp += a.GetConj(i, j) * v.Get(xiter[i])
					}
				}
				v.Set(xiter[j], temp)
			}
		} else {
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				if noconj {
					if nounit {
						temp *= a.Get(j, j)
					}
					for i = j + 1; i < n; i++ {
						temp += a.Get(i, j) * v.Get(xiter[i])
					}
				} else {
					if nounit {
						temp *= a.GetConj(j, j)
					}
					for i = j + 1; i < n; i++ {
						temp += a.GetConj(i, j) * v.Get(xiter[i])
					}
				}
				v.Set(xiter[j], temp)
			}
		}
	}

	return
}

// Trsv solves one of the systems of equations
//
//    A*x = b,   or   A**T*x = b,   or   A**H*x = b,
//
// where b and x are n element vectors and A is an n by n unit, or
// non-unit, upper or lower triangular matrix.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func (v *CVector) Trsv(uplo MatUplo, trans MatTrans, diag MatDiag, n int, a *CMatrix, inc int) (err error) {
	var noconj, nounit bool
	var temp, zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	noconj = trans == Trans
	nounit = diag == NonUnit

	//     Set up the start point in X if the increment is not unity. This
	//     will be  ( N - 1 )*INCX  too small for descending loops.
	xiter := v.Iter(n, inc)

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	if trans == NoTrans {
		//        Form  x := inv( A )*x.
		if uplo == Upper {
			for j = n - 1; j >= 0; j-- {
				if v.Get(xiter[j]) != zero {
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/a.Get(j, j))
					}
					temp = v.Get(xiter[j])
					for i = j - 1; i >= 0; i-- {
						v.Set(xiter[i], v.Get(xiter[i])-temp*a.Get(i, j))
					}
				}
			}
		} else {
			for j = 0; j < n; j++ {
				if v.Get(xiter[j]) != zero {
					if nounit {
						v.Set(xiter[j], v.Get(xiter[j])/a.Get(j, j))
					}
					temp = v.Get(xiter[j])
					for i = j + 1; i < n; i++ {
						v.Set(xiter[i], v.Get(xiter[i])-temp*a.Get(i, j))
					}
				}
			}
		}
	} else {
		//        Form  x := inv( A**T )*x  or  x := inv( A**H )*x.
		if uplo == Upper {
			for j = 0; j < n; j++ {
				temp = v.Get(xiter[j])
				if noconj {
					for i = 0; i < j; i++ {
						temp -= a.Get(i, j) * v.Get(xiter[i])
					}
					if nounit {
						temp /= a.Get(j, j)
					}
				} else {
					for i = 0; i < j; i++ {
						temp -= a.GetConj(i, j) * v.Get(xiter[i])
					}
					if nounit {
						temp /= a.GetConj(j, j)
					}
				}
				v.Set(xiter[j], temp)
			}
		} else {
			for j = n - 1; j >= 0; j-- {
				temp = v.Get(xiter[j])
				if noconj {
					for i = n - 1; i >= j+1; i-- {
						temp -= a.Get(i, j) * v.Get(xiter[i])
					}
					if nounit {
						temp /= a.Get(j, j)
					}
				} else {
					for i = n - 1; i >= j+1; i-- {
						temp -= a.GetConj(i, j) * v.Get(xiter[i])
					}
					if nounit {
						temp /= a.GetConj(j, j)
					}
				}
				v.Set(xiter[j], temp)
			}
		}
	}

	return
}

// Hpr performs the hermitian rank 1 operation
//
//    A := alpha*x*x**H + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n hermitian matrix, supplied in packed form.
func (v *CVector) Hpr(uplo MatUplo, n int, alpha float64, x *CVector, incx int) (err error) {
	var temp, zero complex128
	var i, j, k, kk int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || (alpha == real(zero)) {
		return
	}

	//     Set the start point in X if the increment is not unity.
	xiter := x.Iter(n, incx)

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	if uplo == Upper {
		//        Form  A  when upper triangle is stored in AP.
		for j = 0; j < n; j++ {
			if x.Get(xiter[j]) != zero {
				temp = complex(alpha, 0) * x.GetConj(xiter[j])
				for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
					v.Set(k, v.Get(k)+x.Get(xiter[i])*temp)
				}
				v.Set(kk+j, v.GetReCmplx(kk+j)+complex(real(x.Get(xiter[j])*temp), 0))
			} else {
				v.Set(kk+j, v.GetReCmplx(kk+j))
			}
			kk += j + 1
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		for j = 1; j <= n; j++ {
			if x.Get(xiter[j-1]) != zero {
				temp = complex(alpha, 0) * x.GetConj(xiter[j-1])
				v.Set(kk, v.GetReCmplx(kk)+complex(real(temp*x.Get(xiter[j-1])), 0))
				for i, k = j, kk+2; k <= kk+n-j+1; i, k = i+1, k+1 {
					v.Set(k-1, v.Get(k-1)+x.Get(xiter[i])*temp)
				}
			} else {
				v.Set(kk, v.GetReCmplx(kk))
			}
			kk += (n - j + 1)
		}
	}

	return
}

// Hpr2 performs the hermitian rank 2 operation
//
//    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an
// n by n hermitian matrix, supplied in packed form.
func (v *CVector) Hpr2(uplo MatUplo, n int, alpha complex128, x *CVector, incx int, y *CVector, incy int) (err error) {
	var temp1, temp2, zero complex128
	var i, j, k, kk int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || (alpha == zero) {
		return
	}

	//     Set up the start points in X and Y if the increments are not both
	//     unity.
	xiter := x.Iter(n, incx)
	yiter := y.Iter(n, incy)

	//     Start the operations. In this version the elements of the array AP
	//     are accessed sequentially with one pass through AP.
	if uplo == Upper {
		//        Form  A  when upper triangle is stored in AP.
		for j = 0; j < n; j++ {
			if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
				temp1 = alpha * y.GetConj(yiter[j])
				temp2 = cmplx.Conj(alpha * x.Get(xiter[j]))
				for i, k = 0, kk; k < kk+j; i, k = i+1, k+1 {
					v.Set(k, v.Get(k)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
				}
				v.Set(kk+j, v.GetReCmplx(kk+j)+complex(real(x.Get(xiter[j])*temp1+y.Get(yiter[j])*temp2), 0))
			} else {
				v.Set(kk+j, v.GetReCmplx(kk+j))
			}
			kk += j + 1
		}
	} else {
		//        Form  A  when lower triangle is stored in AP.
		for j = 0; j < n; j++ {
			if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
				temp1 = alpha * y.GetConj(yiter[j])
				temp2 = cmplx.Conj(alpha * x.Get(xiter[j]))
				v.Set(kk, v.GetReCmplx(kk)+complex(real(x.Get(xiter[j])*temp1+y.Get(yiter[j])*temp2), 0))
				for i, k = j+1, kk+1; k < kk+n-j; i, k = i+1, k+1 {
					v.Set(k, v.Get(k)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
				}
			} else {
				v.Set(kk, v.GetReCmplx(kk))
			}
			kk += n - j
		}
	}

	return
}

func CVectorFactory() func(int) *CVector {
	return func(n int) *CVector {
		return &CVector{size: n, data: make([]complex128, n, 2*n)}
	}
}

// NewDataVectorFactory creates a new Vectorer with same data memory location
func CVectorDataFactory() func([]complex128) *CVector {
	return func(d []complex128) *CVector {
		return &CVector{size: len(d), data: d}
	}
}

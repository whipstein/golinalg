package mat

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"math/cmplx"
)

func getIdx(m MatMajor, rows, cols, r, c int) int {
	if m == Row {
		return c + cols*r
	}
	return r + rows*c
}

type MatStyle int

const (
	General MatStyle = iota
	Symmetric
	Hermitian
	Triangular
)

func (m MatStyle) String() string {
	switch m {
	case General:
		return "General"
	case Symmetric:
		return "Symmetric"
	case Hermitian:
		return "Hermitian"
	case Triangular:
		return "Triangular"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(m))
}
func (m MatStyle) IsValid() bool {
	if m == General || m == Symmetric || m == Hermitian || m == Triangular {
		return true
	}
	return false
}
func IterMatStyle() []MatStyle {
	return []MatStyle{General, Symmetric, Hermitian, Triangular}
}

type MatStorage int

const (
	Dense MatStorage = iota
	Banded
	Packed
)

func (m MatStorage) String() string {
	switch m {
	case Dense:
		return "Dense"
	case Banded:
		return "Banded"
	case Packed:
		return "Packed"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(m))
}
func (m MatStorage) IsValid() bool {
	if m == Dense || m == Banded || m == Packed {
		return true
	}
	return false
}
func IterMatStorage() []MatStorage {
	return []MatStorage{Dense, Banded, Packed}
}

type MatMajor int

const (
	Row MatMajor = iota
	Col
)

func (m MatMajor) String() string {
	switch m {
	case Row:
		return "Row"
	case Col:
		return "Col"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(m))
}
func (m MatMajor) IsValid() bool {
	if m == Row || m == Col {
		return true
	}
	return false
}
func IterMatMajor() []MatMajor {
	return []MatMajor{Row, Col}
}

type MatTrans int

const (
	NoTrans MatTrans = iota
	Trans
	ConjTrans
)

func (t MatTrans) Byte() byte {
	switch t {
	case NoTrans:
		return byte('N')
	case Trans:
		return byte('T')
	case ConjTrans:
		return byte('C')
	}
	return byte(' ')
}
func (t MatTrans) String() string {
	switch t {
	case NoTrans:
		return "NoTrans"
	case Trans:
		return "Trans"
	case ConjTrans:
		return "ConjTrans"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(t))
}
func (t MatTrans) IsValid() bool {
	if t == NoTrans || t == Trans || t == ConjTrans {
		return true
	}
	return false
}
func (t MatTrans) IsTrans() bool {
	if t == Trans || t == ConjTrans {
		return true
	}
	return false
}
func TransByte(b byte) MatTrans {
	switch b {
	case 'n':
		return NoTrans
	case 'N':
		return NoTrans
	case 't':
		return Trans
	case 'T':
		return Trans
	case 'c':
		return ConjTrans
	case 'C':
		return ConjTrans
	default:
		return -1
	}
}
func IterMatTrans(conj ...bool) []MatTrans {
	if conj == nil || conj[0] == true {
		return []MatTrans{NoTrans, Trans, ConjTrans}
	}

	return []MatTrans{NoTrans, Trans}
}

type MatUplo int

const (
	Full MatUplo = iota
	Lower
	Upper
)

func (u MatUplo) Byte() byte {
	switch u {
	case Full:
		return byte('F')
	case Lower:
		return byte('L')
	case Upper:
		return byte('U')
	}
	return byte(' ')
}
func (u MatUplo) String() string {
	switch u {
	case Lower:
		return "Lower"
	case Upper:
		return "Upper"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(u))
}
func (u MatUplo) IsValid() bool {
	if u == Lower || u == Upper || u == Full {
		return true
	}
	return false
}
func UploByte(b byte) MatUplo {
	switch b {
	case 'u':
		return Upper
	case 'U':
		return Upper
	case 'l':
		return Lower
	case 'L':
		return Lower
	default:
		return Full
	}
}
func IterMatUplo(full ...bool) []MatUplo {
	if full == nil || full[0] == true {
		return []MatUplo{Upper, Lower, Full}
	}

	return []MatUplo{Upper, Lower}
}

type MatDiag int

const (
	NonUnit MatDiag = iota
	Unit
)

func (d MatDiag) Byte() byte {
	switch d {
	case NonUnit:
		return byte('N')
	case Unit:
		return byte('U')
	}
	return byte(' ')
}
func (d MatDiag) String() string {
	switch d {
	case NonUnit:
		return "NonUnit"
	case Unit:
		return "Unit"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(d))
}
func (d MatDiag) IsValid() bool {
	if d == NonUnit || d == Unit {
		return true
	}
	return false
}
func DiagByte(b byte) MatDiag {
	switch b {
	case 'n':
		return NonUnit
	case 'N':
		return NonUnit
	case 'u':
		return Unit
	case 'U':
		return Unit
	default:
		return -1
	}
}
func IterMatDiag() []MatDiag {
	return []MatDiag{NonUnit, Unit}
}

type MatSide int

const (
	Left MatSide = iota
	Right
	Both
)

func (s MatSide) Byte() byte {
	switch s {
	case Left:
		return byte('L')
	case Right:
		return byte('R')
	case Both:
		return byte('B')
	}
	return byte(' ')
}
func (s MatSide) String() string {
	switch s {
	case Left:
		return "Left"
	case Right:
		return "Right"
	case Both:
		return "Both"
	}
	return fmt.Sprintf("Unrecognized: %c", byte(s))
}
func (s MatSide) IsValid() bool {
	if s == Left || s == Right || s == Both {
		return true
	}
	return false
}
func SideByte(b byte) MatSide {
	switch b {
	case 'l':
		return Left
	case 'L':
		return Left
	case 'r':
		return Right
	case 'R':
		return Right
	case 'b':
		return Both
	case 'B':
		return Both
	default:
		return -1
	}
}
func IterMatSide() []MatSide {
	return []MatSide{Left, Right}
}

type MatOpts struct {
	Style   MatStyle
	Storage MatStorage
	Kl, Ku  int
	Uplo    MatUplo
	Diag    MatDiag
	Side    MatSide
	Major   MatMajor
}

func NewMatOpts() MatOpts {
	return MatOpts{
		Style:   General,
		Storage: Dense,
		Uplo:    Full,
		Diag:    NonUnit,
		Side:    Left,
		Major:   Row,
	}
}
func NewMatOptsCol() MatOpts {
	return MatOpts{
		Style:   General,
		Storage: Dense,
		Uplo:    Full,
		Diag:    NonUnit,
		Side:    Left,
		Major:   Col,
	}
}
func (m MatOpts) DeepCopy() MatOpts {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := MatOpts{}
	_ = d.Decode(&result)
	return result
}
func (m MatOpts) Iter() map[string]interface{} {
	opts := make(map[string]interface{})
	opts["style"] = m.Style
	opts["storage"] = m.Storage
	opts["kl"] = m.Kl
	opts["ku"] = m.Ku
	opts["uplo"] = m.Uplo
	opts["diag"] = m.Diag
	opts["side"] = m.Side
	opts["major"] = m.Major
	return opts
}

type TransBuilder struct {
	rows, cols int
	data       []float64
}

func NewTransBuilder(r, c int, x []float64) *TransBuilder {
	return &TransBuilder{rows: r, cols: c, data: x}
}
func (t *TransBuilder) T() []float64 {
	newmat := make([]float64, len(t.data))
	for i := 0; i < t.rows; i++ {
		for j := 0; j < t.cols; j++ {
			newmat[j+t.cols*i] = t.data[i+t.rows*j]
		}
	}
	copy(t.data, newmat)
	return t.data
}

type Matrix struct {
	Rows, Cols int
	Data       []float64
	Opts       MatOpts
}

func (m *Matrix) AppendCol(x []float64) {
	if m.Opts.Major == Row {
		m.Data = append(m.Data, make([]float64, m.Rows)...)
		m.Cols++
		iter := 0
		j := 1
		for i := 1; i < m.Rows*(m.Cols-1)-2; i++ {
			a := m.Rows*(m.Cols-1) - i
			b := m.Rows*m.Cols - 1*j - i
			m.Data[b] = m.Data[a]
			iter++
			if iter >= m.Cols-1 {
				iter = 0
				j++
			}
		}
		for i := 0; i < m.Rows; i++ {
			m.Set(i, m.Cols-1, x[i])
		}
	} else {
		m.Data = append(m.Data, x...)
		m.Cols++
	}
}
func (m *Matrix) AppendRow(x []float64) {
	if m.Opts.Major == Col {
		m.Data = append(m.Data, make([]float64, m.Cols)...)
		m.Rows++
		iter := 0
		j := 1
		for i := 1; i < m.Cols*(m.Rows-1)-2; i++ {
			a := m.Cols*(m.Rows-1) - i
			b := m.Cols*m.Rows - 1*j - i
			m.Data[b] = m.Data[a]
			iter++
			if iter >= m.Rows-1 {
				iter = 0
				j++
			}
		}
		for i := 0; i < m.Cols; i++ {
			m.Set(m.Rows-1, i, x[i])
		}
	} else {
		m.Data = append(m.Data, x...)
		m.Rows++
	}
}

// func (m *Matrix) Rows() int {
// 	return m.Rows
// }
// func (m *Matrix) Cols() int {
// 	return m.Cols
// }
func (m *Matrix) Shape() (int, int) {
	return m.Rows, m.Cols
}

// func (m *Matrix) Data() []float64 {
// 	return m.Data
// }
func (m *Matrix) Get(r, c int) float64 {
	return m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)]
}
func (m *Matrix) GetPtr(r, c int) *float64 {
	return &m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)]
}
func (m *Matrix) GetIdx(idx int) float64 {
	return m.Data[idx]
}
func (m *Matrix) GetIdxPtr(idx int) *float64 {
	return &m.Data[idx]
}
func (m *Matrix) GetCmplx(r, c int) complex128 {
	return complex(m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)], 0)
}
func (m *Matrix) GetMag(r, c int) float64 {
	return math.Abs(m.Get(r, c))
}
func (m *Matrix) GetCmplxIdx(idx int) complex128 {
	return complex(m.Data[idx], 0)
}
func (m *Matrix) GetMagIdx(idx int) float64 {
	return math.Abs(m.Data[idx])
}
func (m *Matrix) Set(r, c int, x float64) {
	m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)] = x
}
func (m *Matrix) SetAll(x float64) {
	for i := range m.Data {
		m.Data[i] = x
	}
}
func (m *Matrix) SetCol(c int, x float64) {
	for i := 0; i < m.Rows; i++ {
		m.Set(i, c, x)
	}
}
func (m *Matrix) SetRow(r int, x float64) {
	for i := 0; i < m.Cols; i++ {
		m.Set(r, i, x)
	}
}
func (m *Matrix) SetIdx(idx int, x float64) {
	m.Data[idx] = x
}
func (m *Matrix) Swap(r1, c1, r2, c2 int) {
	a := getIdx(m.Opts.Major, m.Rows, m.Cols, r1, c1)
	b := getIdx(m.Opts.Major, m.Rows, m.Cols, r2, c2)
	m.Data[a], m.Data[b] = m.Data[b], m.Data[a]
}
func (m *Matrix) T() []float64 {
	t := NewTransBuilder(m.Rows, m.Cols, m.Data)
	return t.T()
}
func (m *Matrix) Copy(r, c int) *Matrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := Matrix{}
	_ = d.Decode(&result)
	result.Data = m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c):]
	return &result
}

// CopyIdx copies Matrix starting at idx index to a new Matrix
func (m *Matrix) CopyIdx(idx int) *Matrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := Matrix{}
	_ = d.Decode(&result)
	result.Data = m.Data[idx:]
	return &result
}
func (m *Matrix) DeepCopy() *Matrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	if err := e.Encode(m); err != nil {
		panic(err)
	}
	d := gob.NewDecoder(&b)
	result := &Matrix{}
	if err := d.Decode(result); err != nil {
		panic(err)
	}

	if len(m.Data) == 0 {
		result.Data = make([]float64, 0)
	} else {
		copy(result.Data, m.Data)
	}

	return result
}
func (m *Matrix) Off(r, c int) *Matrix {
	if getIdx(m.Opts.Major, m.Rows, m.Cols, r, c) >= m.Rows*m.Cols {
		log.Panicf("\n\ntrying to refernce start value greater than total array size! r=%v, c=%v, rows=%v, cols=%v, got %v, have %v\n\n\n", r, c, m.Rows, m.Cols, getIdx(m.Opts.Major, m.Rows, m.Cols, r, c), m.Rows*m.Cols)
	}
	return &Matrix{Rows: m.Rows, Cols: m.Cols, Opts: m.Opts, Data: m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c):]}
}
func (m *Matrix) OffIdx(idx int) *Matrix {
	return &Matrix{Rows: m.Rows, Cols: m.Cols, Opts: m.Opts, Data: m.Data[idx:]}
}
func (m *Matrix) Vector() *Vector {
	return &Vector{size: len(m.Data), data: m.Data}
}
func (m *Matrix) UpdateSize(r, c int) *Matrix {
	m.Rows, m.Cols = r, c
	return m
}
func (m *Matrix) UpdateCols(c int) *Matrix {
	m.Cols = c
	m.Rows = len(m.Data) / m.Cols
	return m
}
func (m *Matrix) UpdateRows(r int) *Matrix {
	m.Rows = r
	m.Cols = len(m.Data) / m.Rows
	return m
}
func (m *Matrix) ToColMajor() {
	if m.Opts.Major == Col {
		return
	}

	optsNew := m.Opts.DeepCopy()
	optsNew.Major = Col
	a := &Matrix{Rows: m.Rows, Cols: m.Cols, Opts: optsNew, Data: newMatrixData(m.Rows, m.Cols)}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			a.Set(i, j, m.Get(i, j))
		}
	}
	m.Opts.Major = Col
	m.Data = a.Data
}
func (m *Matrix) ToRowMajor() {
	if m.Opts.Major == Row {
		return
	}

	optsNew := m.Opts.DeepCopy()
	optsNew.Major = Row
	a := &Matrix{Rows: m.Rows, Cols: m.Cols, Opts: optsNew, Data: newMatrixData(m.Rows, m.Cols)}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			a.Set(i, j, m.Get(i, j))
		}
	}
	m.Opts.Major = Row
	m.Data = a.Data
}

// Ger performs the rank 1 operation
//
//    A := alpha*x*y**T + A,
//
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix.
func (mx *Matrix) Ger(m, n int, alpha float64, x *Vector, incx int, y *Vector, incy int) (err error) {
	var temp, zero float64
	var i, j int

	zero = 0.0
	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || (alpha == zero) {
		return
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	xiter := x.Iter(m, incx)
	yiter := y.Iter(n, incy)
	for j = 0; j < n; j++ {
		if y.Get(yiter[j]) != zero {
			temp = alpha * y.Get(yiter[j])
			for i = 0; i < m; i++ {
				mx.Set(i, j, mx.Get(i, j)+x.Get(xiter[i])*temp)
			}
		}
	}

	return
}

// Syr performs the symmetric rank 1 operation
//
//    A := alpha*x*x**T + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n symmetric matrix.
func (mx *Matrix) Syr(uplo MatUplo, n int, alpha float64, x *Vector, incx int) (err error) {
	var temp, zero float64
	var i, j int

	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if mx.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", mx.Rows, max(1, n))
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

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == Upper {
		for j = 0; j < n; j++ {
			if x.Get(xiter[j]) != zero {
				temp = alpha * x.Get(xiter[j])
				for i = 0; i < j+1; i++ {
					mx.Set(i, j, mx.Get(i, j)+x.Get(xiter[i])*temp)
				}
			}
		}
	} else {
		for j = 0; j < n; j++ {
			if x.Get(xiter[j]) != zero {
				temp = alpha * x.Get(xiter[j])
				for i = j; i < n; i++ {
					mx.Set(i, j, mx.Get(i, j)+x.Get(xiter[i])*temp)
				}
			}
		}
	}

	return
}

// Syr2 performs the symmetric rank 2 operation
//
//    A := alpha*x*y**T + alpha*y*x**T + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an n
// by n symmetric matrix.
func (mx *Matrix) Syr2(uplo MatUplo, n int, alpha float64, x *Vector, incx int, y *Vector, incy int) (err error) {
	var temp1, temp2, zero float64
	var i, j int

	zero = 0.0

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if mx.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", mx.Rows, max(1, n))
	}
	if err != nil {
		Xerbla2([]byte("Dsyr2"), err)
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

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == Upper {
		//        Form  A  when A is stored in the upper triangle.
		for j = 1; j <= n; j++ {
			if (x.Get(xiter[j-1]) != zero) || (y.Get(yiter[j-1]) != zero) {
				temp1 = alpha * y.Get(yiter[j-1])
				temp2 = alpha * x.Get(xiter[j-1])
				for i = 1; i <= j; i++ {
					mx.Set(i-1, j-1, mx.Get(i-1, j-1)+x.Get(xiter[i-1])*temp1+y.Get(yiter[i-1])*temp2)
				}
			}
		}
	} else {
		//        Form  A  when A is stored in the lower triangle.
		for j = 1; j <= n; j++ {
			if (x.Get(xiter[j-1]) != zero) || (y.Get(yiter[j-1]) != zero) {
				temp1 = alpha * y.Get(yiter[j-1])
				temp2 = alpha * x.Get(xiter[j-1])
				for i = j; i <= n; i++ {
					mx.Set(i-1, j-1, mx.Get(i-1, j-1)+x.Get(xiter[i-1])*temp1+y.Get(yiter[i-1])*temp2)
				}
			}
		}
	}

	return
}

// Gemm performs one of the matrix-matrix operations
//    C := alpha*op( A )*op( B ) + beta*C,
// where  op( X ) is one of
//    op( X ) = X   or   op( X ) = X**T,
// alpha and beta are scalars, and A, B and C are matrices, with op( A )
// an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
func (mx *Matrix) Gemm(transa, transb MatTrans, m, n, k int, alpha float64, a, b *Matrix, beta float64) (err error) {
	var nota, notb bool
	var one, temp, zero float64
	var i, j, l, nrowa, nrowb int

	one = 1.0
	zero = 0.0

	//     Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
	//     transposed and set  NROWA, NCOLA and  NROWB  as the number of rows
	//     and  columns of  A  and the  number of  rows  of  B  respectively.
	nota = transa == NoTrans
	notb = transb == NoTrans
	if nota {
		nrowa = m
	} else {
		nrowa = k
	}
	if notb {
		nrowb = k
	} else {
		nrowb = n
	}

	//     Test the input parameters.
	if !transa.IsValid() {
		err = fmt.Errorf("transa invalid: %v", transa.String())
	} else if !transb.IsValid() {
		err = fmt.Errorf("transb invalid: %v", transb.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, nrowb) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, nrowb))
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And if  alpha.eq.zero.
	if alpha == zero {
		if beta == zero {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, zero)
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
				}
			}
		}
		return
	}

	//     Start the operations.
	if notb {
		if nota {
			//           Form  C := alpha*A*B + beta*C.
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					temp = alpha * b.Get(l-1, j-1)
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
					}
				}
			}
		} else {
			//           Form  C := alpha*A**T*B + beta*C
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * b.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		}
	} else {
		if nota {
			//           Form  C := alpha*A*B**T + beta*C
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					temp = alpha * b.Get(j-1, l-1)
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
					}
				}
			}
		} else {
			//           Form  C := alpha*A**T*B**T + beta*C
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * b.Get(j-1, l-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		}
	}

	return
}

// Dsymm performs one of the matrix-matrix operations
//    C := alpha*A*B + beta*C,
// or
//    C := alpha*B*A + beta*C,
// where alpha and beta are scalars,  A is a symmetric matrix and  B and
// C are  m by n matrices.
func (mx *Matrix) Symm(side MatSide, uplo MatUplo, m, n int, alpha float64, a, b *Matrix, beta float64) (err error) {
	var upper bool
	var one, temp1, temp2, zero float64
	var i, j, k, nrowa int

	one = 1.0
	zero = 0.0

	//     Set NROWA as the number of rows of A.
	if side == Left {
		nrowa = m
	} else {
		nrowa = n
	}
	upper = uplo == Upper

	//     Test the input parameters.
	if !side.IsValid() {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, m))
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if beta == zero {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, zero)
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
				}
			}
		}
		return
	}

	//     Start the operations.
	if side == Left {
		//        Form  C := alpha*A*B + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp1 = alpha * b.Get(i-1, j-1)
					temp2 = zero
					for k = 1; k <= i-1; k++ {
						mx.Set(k-1, j-1, mx.Get(k-1, j-1)+temp1*a.Get(k-1, i-1))
						temp2 += b.Get(k-1, j-1) * a.Get(k-1, i-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, temp1*a.Get(i-1, i-1)+alpha*temp2)
					} else {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+temp1*a.Get(i-1, i-1)+alpha*temp2)
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = m; i >= 1; i-- {
					temp1 = alpha * b.Get(i-1, j-1)
					temp2 = zero
					for k = i + 1; k <= m; k++ {
						mx.Set(k-1, j-1, mx.Get(k-1, j-1)+temp1*a.Get(k-1, i-1))
						temp2 += b.Get(k-1, j-1) * a.Get(k-1, i-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, temp1*a.Get(i-1, i-1)+alpha*temp2)
					} else {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+temp1*a.Get(i-1, i-1)+alpha*temp2)
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*B*A + beta*C.
		for j = 1; j <= n; j++ {
			temp1 = alpha * a.Get(j-1, j-1)
			if beta == zero {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, temp1*b.Get(i-1, j-1))
				}
			} else {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+temp1*b.Get(i-1, j-1))
				}
			}
			for k = 1; k <= j-1; k++ {
				if upper {
					temp1 = alpha * a.Get(k-1, j-1)
				} else {
					temp1 = alpha * a.Get(j-1, k-1)
				}
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp1*b.Get(i-1, k-1))
				}
			}
			for k = j + 1; k <= n; k++ {
				if upper {
					temp1 = alpha * a.Get(j-1, k-1)
				} else {
					temp1 = alpha * a.Get(k-1, j-1)
				}
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp1*b.Get(i-1, k-1))
				}
			}
		}
	}

	return
}

// Trmm performs one of the matrix-matrix operations
//    B := alpha*op( A )*B,   or   B := alpha*B*op( A ),
// where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
// non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
//    op( A ) = A   or   op( A ) = A**T.
func (mx *Matrix) Trmm(side MatSide, uplo MatUplo, transa MatTrans, diag MatDiag, m, n int, alpha float64, a *Matrix) (err error) {
	var lside, nounit, upper bool
	var one, temp, zero float64
	var i, j, k, nrowa int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	lside = side == Left
	if lside {
		nrowa = m
	} else {
		nrowa = n
	}
	nounit = diag == NonUnit
	upper = uplo == Upper

	if !side.IsValid() {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !transa.IsValid() {
		err = fmt.Errorf("transa invalid: %v", transa.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if m == 0 || n == 0 {
		return
	}
	//
	//     And when  alpha.eq.zero.
	//
	if alpha == zero {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				mx.Set(i-1, j-1, zero)
			}
		}
		return
	}

	//     Start the operations.
	if lside {
		if transa == NoTrans {
			//           Form  B := alpha*A*B.
			if upper {
				for j = 1; j <= n; j++ {
					for k = 1; k <= m; k++ {
						if mx.Get(k-1, j-1) != zero {
							temp = alpha * mx.Get(k-1, j-1)
							for i = 1; i <= k-1; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, k-1))
							}
							if nounit {
								temp *= a.Get(k-1, k-1)
							}
							mx.Set(k-1, j-1, temp)
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for k = m; k >= 1; k-- {
						if mx.Get(k-1, j-1) != zero {
							temp = alpha * mx.Get(k-1, j-1)
							mx.Set(k-1, j-1, temp)
							if nounit {
								mx.Set(k-1, j-1, mx.Get(k-1, j-1)*a.Get(k-1, k-1))
							}
							for i = k + 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, k-1))
							}
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*A**T*B.
			if upper {
				for j = 1; j <= n; j++ {
					for i = m; i >= 1; i-- {
						temp = mx.Get(i-1, j-1)
						if nounit {
							temp *= a.Get(i-1, i-1)
						}
						for k = 1; k <= i-1; k++ {
							temp += a.Get(k-1, i-1) * mx.Get(k-1, j-1)
						}
						mx.Set(i-1, j-1, alpha*temp)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= m; i++ {
						temp = mx.Get(i-1, j-1)
						if nounit {
							temp *= a.Get(i-1, i-1)
						}
						for k = i + 1; k <= m; k++ {
							temp += a.Get(k-1, i-1) * mx.Get(k-1, j-1)
						}
						mx.Set(i-1, j-1, alpha*temp)
					}
				}
			}
		}
	} else {
		if transa == NoTrans {
			//           Form  B := alpha*B*A.
			if upper {
				for j = n; j >= 1; j-- {
					temp = alpha
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, temp*mx.Get(i-1, j-1))
					}
					for k = 1; k <= j-1; k++ {
						if a.Get(k-1, j-1) != zero {
							temp = alpha * a.Get(k-1, j-1)
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*mx.Get(i-1, k-1))
							}
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					temp = alpha
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, temp*mx.Get(i-1, j-1))
					}
					for k = j + 1; k <= n; k++ {
						if a.Get(k-1, j-1) != zero {
							temp = alpha * a.Get(k-1, j-1)
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*mx.Get(i-1, k-1))
							}
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*B*A**T.
			if upper {
				for k = 1; k <= n; k++ {
					for j = 1; j <= k-1; j++ {
						if a.Get(j-1, k-1) != zero {
							temp = alpha * a.Get(j-1, k-1)
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*mx.Get(i-1, k-1))
							}
						}
					}
					temp = alpha
					if nounit {
						temp *= a.Get(k-1, k-1)
					}
					if temp != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, temp*mx.Get(i-1, k-1))
						}
					}
				}
			} else {
				for k = n; k >= 1; k-- {
					for j = k + 1; j <= n; j++ {
						if a.Get(j-1, k-1) != zero {
							temp = alpha * a.Get(j-1, k-1)
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*mx.Get(i-1, k-1))
							}
						}
					}
					temp = alpha
					if nounit {
						temp *= a.Get(k-1, k-1)
					}
					if temp != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, temp*mx.Get(i-1, k-1))
						}
					}
				}
			}
		}
	}

	return
}

// Trsm solves one of the matrix equations
//    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
// where alpha is a scalar, X and B are m by n matrices, A is a unit, or
// non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
//    op( A ) = A   or   op( A ) = A**T.
// The matrix X is overwritten on B.
func (mx *Matrix) Trsm(side MatSide, uplo MatUplo, transa MatTrans, diag MatDiag, m, n int, alpha float64, a *Matrix) (err error) {
	var lside, nounit, upper bool
	var one, temp, zero float64
	var i, j, k, nrowa int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	lside = side == Left
	if lside {
		nrowa = m
	} else {
		nrowa = n
	}
	nounit = diag == NonUnit
	upper = uplo == Upper

	if !side.IsValid() {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !transa.IsValid() {
		err = fmt.Errorf("transa invalid: %v", transa.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				mx.Set(i-1, j-1, zero)
			}
		}
		return
	}

	//     Start the operations.
	if lside {
		if transa == NoTrans {
			//           Form  B := alpha*inv( A )*B.
			if upper {
				for j = 1; j <= n; j++ {
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, alpha*mx.Get(i-1, j-1))
						}
					}
					for k = m; k >= 1; k-- {
						if mx.Get(k-1, j-1) != zero {
							if nounit {
								mx.Set(k-1, j-1, mx.Get(k-1, j-1)/a.Get(k-1, k-1))
							}
							for i = 1; i <= k-1; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-mx.Get(k-1, j-1)*a.Get(i-1, k-1))
							}
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, alpha*mx.Get(i-1, j-1))
						}
					}
					for k = 1; k <= m; k++ {
						if mx.Get(k-1, j-1) != zero {
							if nounit {
								mx.Set(k-1, j-1, mx.Get(k-1, j-1)/a.Get(k-1, k-1))
							}
							for i = k + 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-mx.Get(k-1, j-1)*a.Get(i-1, k-1))
							}
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*inv( A**T )*B.
			if upper {
				for j = 1; j <= n; j++ {
					for i = 1; i <= m; i++ {
						temp = alpha * mx.Get(i-1, j-1)
						for k = 1; k <= i-1; k++ {
							temp -= a.Get(k-1, i-1) * mx.Get(k-1, j-1)
						}
						if nounit {
							temp /= a.Get(i-1, i-1)
						}
						mx.Set(i-1, j-1, temp)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = m; i >= 1; i-- {
						temp = alpha * mx.Get(i-1, j-1)
						for k = i + 1; k <= m; k++ {
							temp -= a.Get(k-1, i-1) * mx.Get(k-1, j-1)
						}
						if nounit {
							temp /= a.Get(i-1, i-1)
						}
						mx.Set(i-1, j-1, temp)
					}
				}
			}
		}
	} else {
		if transa == NoTrans {
			//           Form  B := alpha*B*inv( A ).
			if upper {
				for j = 1; j <= n; j++ {
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, alpha*mx.Get(i-1, j-1))
						}
					}
					for k = 1; k <= j-1; k++ {
						if a.Get(k-1, j-1) != zero {
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-a.Get(k-1, j-1)*mx.Get(i-1, k-1))
							}
						}
					}
					if nounit {
						temp = one / a.Get(j-1, j-1)
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, temp*mx.Get(i-1, j-1))
						}
					}
				}
			} else {
				for j = n; j >= 1; j-- {
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, alpha*mx.Get(i-1, j-1))
						}
					}
					for k = j + 1; k <= n; k++ {
						if a.Get(k-1, j-1) != zero {
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-a.Get(k-1, j-1)*mx.Get(i-1, k-1))
							}
						}
					}
					if nounit {
						temp = one / a.Get(j-1, j-1)
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, temp*mx.Get(i-1, j-1))
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*B*inv( A**T ).
			if upper {
				for k = n; k >= 1; k-- {
					if nounit {
						temp = one / a.Get(k-1, k-1)
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, temp*mx.Get(i-1, k-1))
						}
					}
					for j = 1; j <= k-1; j++ {
						if a.Get(j-1, k-1) != zero {
							temp = a.Get(j-1, k-1)
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-temp*mx.Get(i-1, k-1))
							}
						}
					}
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, alpha*mx.Get(i-1, k-1))
						}
					}
				}
			} else {
				for k = 1; k <= n; k++ {
					if nounit {
						temp = one / a.Get(k-1, k-1)
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, temp*mx.Get(i-1, k-1))
						}
					}
					for j = k + 1; j <= n; j++ {
						if a.Get(j-1, k-1) != zero {
							temp = a.Get(j-1, k-1)
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-temp*mx.Get(i-1, k-1))
							}
						}
					}
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, alpha*mx.Get(i-1, k-1))
						}
					}
				}
			}
		}
	}

	return
}

// Syrk performs one of the symmetric rank k operations
//    C := alpha*A*A**T + beta*C,
// or
//    C := alpha*A**T*A + beta*C,
// where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
// and  A  is an  n by k  matrix in the first case and a  k by n  matrix
// in the second case.
func (mx *Matrix) Syrk(uplo MatUplo, trans MatTrans, n, k int, alpha float64, a *Matrix, beta float64) (err error) {
	var upper bool
	var one, temp, zero float64
	var i, j, l, nrowa int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if trans == NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if mx.Rows < max(1, n) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", mx.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if upper {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
			}
		} else {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
			}
		}
		return
	}

	//     Start the operations.
	if trans == NoTrans {
		//        Form  C := alpha*A*A**T + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if a.Get(j-1, l-1) != zero {
						temp = alpha * a.Get(j-1, l-1)
						for i = 1; i <= j; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
						}
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if a.Get(j-1, l-1) != zero {
						temp = alpha * a.Get(j-1, l-1)
						for i = j; i <= n; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
						}
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*A**T*A + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = j; i <= n; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		}
	}

	return
}

// Syr2k performs one of the symmetric rank 2k operations
//    C := alpha*A*B**T + alpha*B*A**T + beta*C,
// or
//    C := alpha*A**T*B + alpha*B**T*A + beta*C,
// where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
// and  A and B  are  n by k  matrices  in the  first  case  and  k by n
// matrices in the second case.
func (mx *Matrix) Syr2k(uplo MatUplo, trans MatTrans, n, k int, alpha float64, a, b *Matrix, beta float64) (err error) {
	var upper bool
	var one, temp1, temp2, zero float64
	var i, j, l, nrowa int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if trans == NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, nrowa) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, nrowa))
	} else if mx.Rows < max(1, n) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", mx.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if upper {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
			}
		} else {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
			}
		}
		return
	}

	//     Start the operations.
	if trans == NoTrans {
		//        Form  C := alpha*A*B**T + alpha*B*A**T + C.
		if upper {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if (a.Get(j-1, l-1) != zero) || (b.Get(j-1, l-1) != zero) {
						temp1 = alpha * b.Get(j-1, l-1)
						temp2 = alpha * a.Get(j-1, l-1)
						for i = 1; i <= j; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+a.Get(i-1, l-1)*temp1+b.Get(i-1, l-1)*temp2)
						}
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if (a.Get(j-1, l-1) != zero) || (b.Get(j-1, l-1) != zero) {
						temp1 = alpha * b.Get(j-1, l-1)
						temp2 = alpha * a.Get(j-1, l-1)
						for i = j; i <= n; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+a.Get(i-1, l-1)*temp1+b.Get(i-1, l-1)*temp2)
						}
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*A**T*B + alpha*B**T*A + C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					temp1 = zero
					temp2 = zero
					for l = 1; l <= k; l++ {
						temp1 += a.Get(l-1, i-1) * b.Get(l-1, j-1)
						temp2 += b.Get(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp1+alpha*temp2)
					} else {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+alpha*temp1+alpha*temp2)
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = j; i <= n; i++ {
					temp1 = zero
					temp2 = zero
					for l = 1; l <= k; l++ {
						temp1 += a.Get(l-1, i-1) * b.Get(l-1, j-1)
						temp2 += b.Get(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp1+alpha*temp2)
					} else {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+alpha*temp1+alpha*temp2)
					}
				}
			}
		}
	}

	return
}

func NewMatrix(r, c int, opts ...MatOpts) *Matrix {
	var o MatOpts

	if opts != nil {
		o = opts[0]
	}
	return &Matrix{Rows: r, Cols: c, Opts: o, Data: newMatrixData(r, c)}
}

func MatrixFactory() func(int, int, ...MatOpts) *Matrix {
	return func(r, c int, opts ...MatOpts) *Matrix {
		return NewMatrix(r, c, opts...)
	}
}

func MatrixDataFactory() func(*Matrix, []float64) *Matrix {
	return func(m *Matrix, d []float64) *Matrix {
		out := m.DeepCopy()
		out.Data = d
		return out
	}
}

func newMatrixData(r, c int) []float64 {
	return make([]float64, r*c, 2*r*c)
}

type CTransBuilder struct {
	rows, cols int
	data       []complex128
}

func NewCTransBuilder(r, c int, x []complex128) *CTransBuilder {
	return &CTransBuilder{rows: r, cols: c, data: x}
}
func (t *CTransBuilder) T() []complex128 {
	newmat := make([]complex128, len(t.data))
	for i := 0; i < t.rows; i++ {
		for j := 0; j < t.cols; j++ {
			newmat[j+t.cols*i] = t.data[i+t.rows*j]
		}
	}
	copy(t.data, newmat)
	return t.data
}

type CMatrix struct {
	Rows, Cols int
	Data       []complex128
	Opts       MatOpts
}

func (m *CMatrix) AppendCol(x []complex128) {
	m.Cols += 1
	newData := make([]complex128, m.Rows*m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			newData[getIdx(m.Opts.Major, m.Rows, m.Cols, i, j)] = m.Get(i, j)
		}
	}
	m.Data = newData
	for i := 0; i < m.Rows; i++ {
		m.Set(i, m.Cols-1, x[i])
	}
}
func (m *CMatrix) AppendRow(x []complex128) {
	m.Rows += 1
	newData := make([]complex128, m.Rows*m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			newData[getIdx(m.Opts.Major, m.Rows, m.Cols, i, j)] = m.Get(i, j)
		}
	}
	m.Data = newData
	for i := 0; i < m.Cols; i++ {
		m.Set(m.Rows-1, i, x[i])
	}
}

// func (m *CMatrix) Rows() int {
// 	return m.Rows
// }
// func (m *CMatrix) Cols() int {
// 	return m.Cols
// }
func (m *CMatrix) Shape() (int, int) {
	return m.Rows, m.Cols
}

// func (m *CMatrix) Data() []complex128 {
// 	return m.Data
// }
func (m *CMatrix) Get(r, c int) complex128 {
	return m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)]
}
func (m *CMatrix) GetConj(r, c int) complex128 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetConj: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return cmplx.Conj(m.Get(r, c))
}
func (m *CMatrix) GetMag(r, c int) float64 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetMag: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return cmplx.Abs(m.Get(r, c))
}
func (m *CMatrix) GetArg(r, c int) float64 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetArg: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return cmplx.Phase(m.Get(r, c))
}
func (m *CMatrix) GetDeg(r, c int) float64 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetDeg: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return cmplx.Phase(m.Get(r, c)) * 180 / math.Pi
}
func (m *CMatrix) GetRe(r, c int) float64 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetRe: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return real(m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)])
}
func (m *CMatrix) GetIm(r, c int) float64 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetIm: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return imag(m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)])
}
func (m *CMatrix) GetReCmplx(r, c int) complex128 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetReCmplx: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return complex(real(m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)]), 0)
}
func (m *CMatrix) GetImCmplx(r, c int) complex128 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetImCmplx: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return complex(0, imag(m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)]))
}
func (m *CMatrix) GetConjProd(r, c int) float64 {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetConjProd: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return real(cmplx.Conj(m.Get(r, c)) * m.Get(r, c))
}
func (m *CMatrix) GetPtr(r, c int) *complex128 {
	rnew, cnew := r, c

	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.GetPtr: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	return &m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, rnew, cnew)]
}
func (m *CMatrix) GetIdx(idx int) complex128 {
	return m.Data[idx]
}
func (m *CMatrix) GetConjIdx(idx int) complex128 {
	return cmplx.Conj(m.GetIdx(idx))
}
func (m *CMatrix) GetMagIdx(idx int) float64 {
	return cmplx.Abs(m.GetIdx(idx))
}
func (m *CMatrix) GetArgIdx(idx int) float64 {
	return cmplx.Phase(m.GetIdx(idx))
}
func (m *CMatrix) GetDegIdx(idx int) float64 {
	return cmplx.Phase(m.GetIdx(idx)) * 180 / math.Pi
}
func (m *CMatrix) GetReIdx(idx int) complex128 {
	return complex(real(m.GetIdx(idx)), 0)
}
func (m *CMatrix) GetImIdx(idx int) complex128 {
	return complex(0, imag(m.GetIdx(idx)))
}
func (m *CMatrix) GetConjProdIdx(idx int) float64 {
	return real(cmplx.Conj(m.GetIdx(idx)) * m.GetIdx(idx))
}
func (m *CMatrix) GetIdxPtr(idx int) *complex128 {
	return &m.Data[idx]
}
func (m *CMatrix) Set(r, c int, x complex128) {
	// if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
	// 	log.Panicf("CMatrix.Set: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	// }

	m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)] = x
}
func (m *CMatrix) SetAll(x complex128) {
	for i := range m.Data {
		m.Data[i] = x
	}
}
func (m *CMatrix) SetCol(c int, x complex128) {
	for i := 0; i < m.Rows; i++ {
		m.Set(i, c, x)
	}
}
func (m *CMatrix) SetRow(r int, x complex128) {
	for i := 0; i < m.Cols; i++ {
		m.Set(r, i, x)
	}
}
func (m *CMatrix) SetRe(r, c int, x float64) {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.SetRe: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)] = complex(x, 0)
}
func (m *CMatrix) SetIm(r, c int, x float64) {
	if r > m.Rows || c > m.Cols || r < 0 || c < 0 {
		log.Panicf("CMatrix.SetIm: Invalid row/column: got (%d,%d) have (%d,%d)\n", r, c, m.Rows, m.Cols)
	}

	m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c)] = complex(0, x)
}
func (m *CMatrix) SetReAll(x float64) {
	for i := range m.Data {
		m.Data[i] = complex(x, 0)
	}
}
func (m *CMatrix) SetReCol(c int, x float64) {
	for i := 0; i < m.Rows; i++ {
		m.SetRe(i, c, x)
	}
}
func (m *CMatrix) SetReRow(r int, x float64) {
	for i := 0; i < m.Cols; i++ {
		m.SetRe(r, i, x)
	}
}
func (m *CMatrix) SetImAll(x float64) {
	for i := range m.Data {
		m.Data[i] = complex(0, x)
	}
}
func (m *CMatrix) SetImCol(c int, x float64) {
	for i := 0; i < m.Rows; i++ {
		m.SetIm(i, c, x)
	}
}
func (m *CMatrix) SetImRow(r int, x float64) {
	for i := 0; i < m.Cols; i++ {
		m.SetIm(r, i, x)
	}
}
func (m *CMatrix) SetIdx(idx int, x complex128) {
	m.Data[idx] = x
}
func (m *CMatrix) T() []complex128 {
	t := NewCTransBuilder(m.Rows, m.Cols, m.Data)
	return t.T()
}
func (m *CMatrix) Copy(r, c int) *CMatrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := CMatrix{}
	_ = d.Decode(&result)
	result.Data = m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c):]
	return &result
}
func (m *CMatrix) CopyIdx(idx int) *CMatrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	_ = e.Encode(m)
	d := gob.NewDecoder(&b)
	result := CMatrix{}
	_ = d.Decode(&result)
	result.Data = m.Data[idx:]
	return &result
}
func (m *CMatrix) DeepCopy() *CMatrix {
	b := bytes.Buffer{}
	e := gob.NewEncoder(&b)
	if err := e.Encode(m); err != nil {
		panic(err)
	}
	d := gob.NewDecoder(&b)
	result := &CMatrix{}
	if err := d.Decode(result); err != nil {
		panic(err)
	}

	if len(m.Data) == 0 {
		result.Data = make([]complex128, 0)
	} else {
		copy(result.Data, m.Data)
	}

	return result
}
func (m *CMatrix) Off(r, c int) *CMatrix {
	if getIdx(m.Opts.Major, m.Rows, m.Cols, r, c) >= m.Rows*m.Cols {
		log.Panicf("\n\ntrying to refernce start value greater than total array size! r=%v, c=%v, rows=%v, cols=%v, got %v, have %v\n\n\n", r, c, m.Rows, m.Cols, getIdx(m.Opts.Major, m.Rows, m.Cols, r, c), m.Rows*m.Cols)
	}
	return &CMatrix{Rows: m.Rows, Cols: m.Cols, Opts: m.Opts, Data: m.Data[getIdx(m.Opts.Major, m.Rows, m.Cols, r, c):]}
}
func (m *CMatrix) OffIdx(idx int) *CMatrix {
	return &CMatrix{Rows: m.Rows, Cols: m.Cols, Opts: m.Opts, Data: m.Data[idx:]}
}
func (m *CMatrix) CVector() *CVector {
	return &CVector{size: len(m.Data), data: m.Data}
}
func (m *CMatrix) UpdateSize(r, c int) *CMatrix {
	m.Rows, m.Cols = r, c
	return m
}
func (m *CMatrix) UpdateCols(c int) *CMatrix {
	m.Cols = c
	m.Rows = len(m.Data) / m.Cols
	return m
}
func (m *CMatrix) UpdateRows(r int) *CMatrix {
	m.Rows = r
	m.Cols = len(m.Data) / m.Rows
	return m
}
func (m *CMatrix) ToColMajor() {
	if m.Opts.Major == Col {
		return
	}

	optsNew := m.Opts.DeepCopy()
	optsNew.Major = Col
	a := &CMatrix{Rows: m.Rows, Cols: m.Cols, Opts: optsNew, Data: newCMatrixData(m.Rows, m.Cols)}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			a.Set(i, j, m.Get(i, j))
		}
	}
	m.Opts.Major = Col
	m.Data = a.Data
}
func (m *CMatrix) ToRowMajor() {
	if m.Opts.Major == Row {
		return
	}

	optsNew := m.Opts.DeepCopy()
	optsNew.Major = Row
	a := &CMatrix{Rows: m.Rows, Cols: m.Cols, Opts: optsNew, Data: newCMatrixData(m.Rows, m.Cols)}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			a.Set(i, j, m.Get(i, j))
		}
	}
	m.Opts.Major = Row
	m.Data = a.Data
}

// Gerc performs the rank 1 operation
//
//    A := alpha*x*y**H + A,
//
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix.
func (mx *CMatrix) Gerc(m, n int, alpha complex128, x *CVector, incx int, y *CVector, incy int) (err error) {
	var temp, zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || (alpha == zero) {
		return
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	xiter := x.Iter(m, incx)
	yiter := y.Iter(n, incy)

	for j = 0; j < n; j++ {
		if y.Get(yiter[j]) != zero {
			temp = alpha * y.GetConj(yiter[j])
			for i = 0; i < m; i++ {
				mx.Set(i, j, mx.Get(i, j)+x.Get(xiter[i])*temp)
			}
		}
	}

	return
}

// Geru performs the rank 1 operation
//
//    A := alpha*x*y**T + A,
//
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix.
func (mx *CMatrix) Geru(m, n int, alpha complex128, x *CVector, incx int, y *CVector, incy int) (err error) {
	var temp, zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || (alpha == zero) {
		return
	}

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through A.
	xiter := x.Iter(m, incx)
	yiter := y.Iter(n, incy)

	for j = 0; j < n; j++ {
		if y.Get(yiter[j]) != zero {
			temp = alpha * y.Get(yiter[j])
			for i = 0; i < m; i++ {
				mx.Set(i, j, mx.Get(i, j)+x.Get(xiter[i])*temp)
			}
		}
	}

	return
}

// Her performs the hermitian rank 1 operation
//
//    A := alpha*x*x**H + A,
//
// where alpha is a real scalar, x is an n element vector and A is an
// n by n hermitian matrix.
func (mx *CMatrix) Her(uplo MatUplo, n int, alpha float64, x *CVector, incx int) (err error) {
	var temp, zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if mx.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", mx.Rows, max(1, n))
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

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == Upper {
		//        Form  A  when A is stored in upper triangle.
		for j = 0; j < n; j++ {
			if x.Get(xiter[j]) != zero {
				temp = complex(alpha, 0) * x.GetConj(xiter[j])
				for i = 0; i < j; i++ {
					mx.Set(i, j, mx.Get(i, j)+x.Get(xiter[i])*temp)
				}
				mx.Set(j, j, mx.GetReCmplx(j, j)+complex(real(x.Get(xiter[j])*temp), 0))
			} else {
				mx.Set(j, j, mx.GetReCmplx(j, j))
			}
		}
	} else {
		//        Form  A  when A is stored in lower triangle.
		for j = 0; j < n; j++ {
			if x.Get(xiter[j]) != zero {
				temp = complex(alpha, 0) * x.GetConj(xiter[j])
				mx.Set(j, j, mx.GetReCmplx(j, j)+complex(real(temp*x.Get(xiter[j])), 0))
				for i = j + 1; i < n; i++ {
					mx.Set(i, j, mx.Get(i, j)+x.Get(xiter[i])*temp)
				}
			} else {
				mx.Set(j, j, mx.GetReCmplx(j, j))
			}
		}
	}

	return
}

// Her2 performs the hermitian rank 2 operation
//
//    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
//
// where alpha is a scalar, x and y are n element vectors and A is an n
// by n hermitian matrix.
func (mx *CMatrix) Her2(uplo MatUplo, n int, alpha complex128, x *CVector, incx int, y *CVector, incy int) (err error) {
	var temp1, temp2, zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if mx.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", mx.Rows, max(1, n))
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

	//     Start the operations. In this version the elements of A are
	//     accessed sequentially with one pass through the triangular part
	//     of A.
	if uplo == Upper {
		//        Form  A  when A is stored in the upper triangle.
		for j = 0; j < n; j++ {
			if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
				temp1 = alpha * y.GetConj(yiter[j])
				temp2 = cmplx.Conj(alpha * x.Get(xiter[j]))
				for i = 0; i < j; i++ {
					mx.Set(i, j, mx.Get(i, j)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
				}
				mx.Set(j, j, mx.GetReCmplx(j, j)+complex(real(x.Get(xiter[j])*temp1+y.Get(yiter[j])*temp2), 0))
			} else {
				mx.Set(j, j, mx.GetReCmplx(j, j))
			}
		}
	} else {
		//        Form  A  when A is stored in the lower triangle.
		for j = 0; j < n; j++ {
			if (x.Get(xiter[j]) != zero) || (y.Get(yiter[j]) != zero) {
				temp1 = alpha * y.GetConj(yiter[j])
				temp2 = cmplx.Conj(alpha * x.Get(xiter[j]))
				mx.Set(j, j, mx.GetReCmplx(j, j)+complex(real(x.Get(xiter[j])*temp1+y.Get(yiter[j])*temp2), 0))
				for i = j + 1; i < n; i++ {
					mx.Set(i, j, mx.Get(i, j)+x.Get(xiter[i])*temp1+y.Get(yiter[i])*temp2)
				}
			} else {
				mx.Set(j, j, mx.GetReCmplx(j, j))
			}
		}
	}

	return
}

// Gemm performs one of the matrix-matrix operations
//
//    C := alpha*op( A )*op( B ) + beta*C,
//
// where  op( X ) is one of
//
//    op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,
//
// alpha and beta are scalars, and A, B and C are matrices, with op( A )
// an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
func (mx *CMatrix) Gemm(transa, transb MatTrans, m, n, k int, alpha complex128, a, b *CMatrix, beta complex128) (err error) {
	var conja, conjb, nota, notb bool
	var one, temp, zero complex128
	var i, j, l, nrowa, nrowb int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
	//     conjugated or transposed, set  CONJA and CONJB  as true if  A  and
	//     B  respectively are to be  transposed but  not conjugated  and set
	//     NROWA, NCOLA and  NROWB  as the number of rows and  columns  of  A
	//     and the number of rows of  B  respectively.
	nota = transa == NoTrans
	notb = transb == NoTrans
	conja = transa == ConjTrans
	conjb = transb == ConjTrans
	if nota {
		nrowa = m
		// ncola = k
	} else {
		nrowa = k
		// ncola = m
	}
	if notb {
		nrowb = k
	} else {
		nrowb = n
	}

	//     Test the input parameters.
	if (!nota) && (!conja) && (transa != Trans) {
		err = fmt.Errorf("transa invalid: %v", transa.String())
	} else if (!notb) && (!conjb) && (transb != Trans) {
		err = fmt.Errorf("transb invalid: %v", transb.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, nrowb) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, nrowb))
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if beta == zero {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, zero)
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
				}
			}
		}
		return
	}

	//     Start the operations.
	if notb {
		if nota {
			//           Form  C := alpha*A*B + beta*C.
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					temp = alpha * b.Get(l-1, j-1)
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
					}
				}
			}
		} else if conja {
			//           Form  C := alpha*A**H*B + beta*C.
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.GetConj(l-1, i-1) * b.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		} else {
			//           Form  C := alpha*A**T*B + beta*C
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * b.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		}
	} else if nota {
		if conjb {
			//           Form  C := alpha*A*B**H + beta*C.
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					temp = alpha * b.GetConj(j-1, l-1)
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
					}
				}
			}
		} else {
			//           Form  C := alpha*A*B**T + beta*C
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					temp = alpha * b.Get(j-1, l-1)
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
					}
				}
			}
		}
	} else if conja {
		if conjb {
			//           Form  C := alpha*A**H*B**H + beta*C.
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp = temp + a.GetConj(l-1, i-1)*b.GetConj(j-1, l-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		} else {
			//           Form  C := alpha*A**H*B**T + beta*C
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp = temp + a.GetConj(l-1, i-1)*b.Get(j-1, l-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		}
	} else {
		if conjb {
			//           Form  C := alpha*A**T*B**H + beta*C
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * b.GetConj(j-1, l-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		} else {
			//           Form  C := alpha*A**T*B**T + beta*C
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * b.Get(j-1, l-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		}
	}

	return
}

// Hemm performs one of the matrix-matrix operations
//
//    C := alpha*A*B + beta*C,
//
// or
//
//    C := alpha*B*A + beta*C,
//
// where alpha and beta are scalars, A is an hermitian matrix and  B and
// C are m by n matrices.
func (mx *CMatrix) Hemm(side MatSide, uplo MatUplo, m, n int, alpha complex128, a, b *CMatrix, beta complex128) (err error) {
	var upper bool
	var one, temp1, temp2, zero complex128
	var i, j, k, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Set NROWA as the number of rows of A.
	if side == Left {
		nrowa = m
	} else {
		nrowa = n
	}
	upper = uplo == Upper

	//     Test the input parameters.
	if !side.IsValid() {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, m))
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if beta == zero {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, zero)
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
				}
			}
		}
		return
	}

	//     Start the operations.
	if side == Left {
		//        Form  C := alpha*A*B + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp1 = alpha * b.Get(i-1, j-1)
					temp2 = zero
					for k = 1; k <= i-1; k++ {
						mx.Set(k-1, j-1, mx.Get(k-1, j-1)+temp1*a.Get(k-1, i-1))
						temp2 += b.Get(k-1, j-1) * a.GetConj(k-1, i-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, temp1*a.GetReCmplx(i-1, i-1)+alpha*temp2)
					} else {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+temp1*a.GetReCmplx(i-1, i-1)+alpha*temp2)
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = m; i >= 1; i-- {
					temp1 = alpha * b.Get(i-1, j-1)
					temp2 = zero
					for k = i + 1; k <= m; k++ {
						mx.Set(k-1, j-1, mx.Get(k-1, j-1)+temp1*a.Get(k-1, i-1))
						temp2 += b.Get(k-1, j-1) * a.GetConj(k-1, i-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, temp1*a.GetReCmplx(i-1, i-1)+alpha*temp2)
					} else {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+temp1*a.GetReCmplx(i-1, i-1)+alpha*temp2)
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*B*A + beta*C.
		for j = 1; j <= n; j++ {
			temp1 = alpha * a.GetReCmplx(j-1, j-1)
			if beta == zero {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, temp1*b.Get(i-1, j-1))
				}
			} else {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+temp1*b.Get(i-1, j-1))
				}
			}
			for k = 1; k <= j-1; k++ {
				if upper {
					temp1 = alpha * a.Get(k-1, j-1)
				} else {
					temp1 = alpha * a.GetConj(j-1, k-1)
				}
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp1*b.Get(i-1, k-1))
				}
			}
			for k = j + 1; k <= n; k++ {
				if upper {
					temp1 = alpha * a.GetConj(j-1, k-1)
				} else {
					temp1 = alpha * a.Get(k-1, j-1)
				}
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp1*b.Get(i-1, k-1))
				}
			}
		}
	}

	return
}

// Symm performs one of the matrix-matrix operations
//
//    C := alpha*A*B + beta*C,
//
// or
//
//    C := alpha*B*A + beta*C,
//
// where  alpha and beta are scalars, A is a symmetric matrix and  B and
// C are m by n matrices.
func (mx *CMatrix) Symm(side MatSide, uplo MatUplo, m, n int, alpha complex128, a, b *CMatrix, beta complex128) (err error) {
	var upper bool
	var one, temp1, temp2, zero complex128
	var i, j, k, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Set NROWA as the number of rows of A.
	if side == Left {
		nrowa = m
	} else {
		nrowa = n
	}
	upper = uplo == Upper

	//     Test the input parameters.
	if !side.IsValid() {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, m))
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (m == 0) || (n == 0) || ((alpha == zero) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if beta == zero {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, zero)
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
				}
			}
		}
		return
	}

	//     Start the operations.
	if side == Left {
		//        Form  C := alpha*A*B + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= m; i++ {
					temp1 = alpha * b.Get(i-1, j-1)
					temp2 = zero
					for k = 1; k <= i-1; k++ {
						mx.Set(k-1, j-1, mx.Get(k-1, j-1)+temp1*a.Get(k-1, i-1))
						temp2 += b.Get(k-1, j-1) * a.Get(k-1, i-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, temp1*a.Get(i-1, i-1)+alpha*temp2)
					} else {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+temp1*a.Get(i-1, i-1)+alpha*temp2)
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = m; i >= 1; i-- {
					temp1 = alpha * b.Get(i-1, j-1)
					temp2 = zero
					for k = i + 1; k <= m; k++ {
						mx.Set(k-1, j-1, mx.Get(k-1, j-1)+temp1*a.Get(k-1, i-1))
						temp2 += b.Get(k-1, j-1) * a.Get(k-1, i-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, temp1*a.Get(i-1, i-1)+alpha*temp2)
					} else {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+temp1*a.Get(i-1, i-1)+alpha*temp2)
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*B*A + beta*C.
		for j = 1; j <= n; j++ {
			temp1 = alpha * a.Get(j-1, j-1)
			if beta == zero {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, temp1*b.Get(i-1, j-1))
				}
			} else {
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+temp1*b.Get(i-1, j-1))
				}
			}
			for k = 1; k <= j-1; k++ {
				if upper {
					temp1 = alpha * a.Get(k-1, j-1)
				} else {
					temp1 = alpha * a.Get(j-1, k-1)
				}
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp1*b.Get(i-1, k-1))
				}
			}
			for k = j + 1; k <= n; k++ {
				if upper {
					temp1 = alpha * a.Get(j-1, k-1)
				} else {
					temp1 = alpha * a.Get(k-1, j-1)
				}
				for i = 1; i <= m; i++ {
					mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp1*b.Get(i-1, k-1))
				}
			}
		}
	}

	return
}

// Trmm performs one of the matrix-matrix operations
//
//    B := alpha*op( A )*B,   or   B := alpha*B*op( A )
//
// where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
// non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
//
//    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
func (mx *CMatrix) Trmm(side MatSide, uplo MatUplo, transa MatTrans, diag MatDiag, m, n int, alpha complex128, a *CMatrix) (err error) {
	var lside, noconj, nounit, upper bool
	var one, temp, zero complex128
	var i, j, k, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	lside = side == Left
	if lside {
		nrowa = m
	} else {
		nrowa = n
	}
	noconj = transa == Trans
	nounit = diag == NonUnit
	upper = uplo == Upper

	if (!lside) && (side != Right) {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if (!upper) && (uplo != Lower) {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if (transa != NoTrans) && (transa != Trans) && (transa != ConjTrans) {
		err = fmt.Errorf("transa invalid: %v", transa.String())
	} else if (diag != Unit) && (diag != NonUnit) {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				mx.Set(i-1, j-1, zero)
			}
		}
		return
	}

	//     Start the operations.
	if lside {
		if transa == NoTrans {
			//           Form  B := alpha*A*B.
			if upper {
				for j = 1; j <= n; j++ {
					for k = 1; k <= m; k++ {
						if mx.Get(k-1, j-1) != zero {
							temp = alpha * mx.Get(k-1, j-1)
							for i = 1; i <= k-1; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, k-1))
							}
							if nounit {
								temp *= a.Get(k-1, k-1)
							}
							mx.Set(k-1, j-1, temp)
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for k = m; k >= 1; k-- {
						if mx.Get(k-1, j-1) != zero {
							temp = alpha * mx.Get(k-1, j-1)
							mx.Set(k-1, j-1, temp)
							if nounit {
								mx.Set(k-1, j-1, mx.Get(k-1, j-1)*a.Get(k-1, k-1))
							}
							for i = k + 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, k-1))
							}
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*A**T*B   or   B := alpha*A**H*B.
			if upper {
				for j = 1; j <= n; j++ {
					for i = m; i >= 1; i-- {
						temp = mx.Get(i-1, j-1)
						if noconj {
							if nounit {
								temp *= a.Get(i-1, i-1)
							}
							for k = 1; k <= i-1; k++ {
								temp += a.Get(k-1, i-1) * mx.Get(k-1, j-1)
							}
						} else {
							if nounit {
								temp *= a.GetConj(i-1, i-1)
							}
							for k = 1; k <= i-1; k++ {
								temp += a.GetConj(k-1, i-1) * mx.Get(k-1, j-1)
							}
						}
						mx.Set(i-1, j-1, alpha*temp)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= m; i++ {
						temp = mx.Get(i-1, j-1)
						if noconj {
							if nounit {
								temp *= a.Get(i-1, i-1)
							}
							for k = i + 1; k <= m; k++ {
								temp += a.Get(k-1, i-1) * mx.Get(k-1, j-1)
							}
						} else {
							if nounit {
								temp *= a.GetConj(i-1, i-1)
							}
							for k = i + 1; k <= m; k++ {
								temp += a.GetConj(k-1, i-1) * mx.Get(k-1, j-1)
							}
						}
						mx.Set(i-1, j-1, alpha*temp)
					}
				}
			}
		}
	} else {
		if transa == NoTrans {
			//           Form  B := alpha*B*A.
			if upper {
				for j = n; j >= 1; j-- {
					temp = alpha
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, temp*mx.Get(i-1, j-1))
					}
					for k = 1; k <= j-1; k++ {
						if a.Get(k-1, j-1) != zero {
							temp = alpha * a.Get(k-1, j-1)
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*mx.Get(i-1, k-1))
							}
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					temp = alpha
					if nounit {
						temp *= a.Get(j-1, j-1)
					}
					for i = 1; i <= m; i++ {
						mx.Set(i-1, j-1, temp*mx.Get(i-1, j-1))
					}
					for k = j + 1; k <= n; k++ {
						if a.Get(k-1, j-1) != zero {
							temp = alpha * a.Get(k-1, j-1)
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*mx.Get(i-1, k-1))
							}
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*B*A**T   or   B := alpha*B*A**H.
			if upper {
				for k = 1; k <= n; k++ {
					for j = 1; j <= k-1; j++ {
						if a.Get(j-1, k-1) != zero {
							if noconj {
								temp = alpha * a.Get(j-1, k-1)
							} else {
								temp = alpha * a.GetConj(j-1, k-1)
							}
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*mx.Get(i-1, k-1))
							}
						}
					}
					temp = alpha
					if nounit {
						if noconj {
							temp *= a.Get(k-1, k-1)
						} else {
							temp *= a.GetConj(k-1, k-1)
						}
					}
					if temp != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, temp*mx.Get(i-1, k-1))
						}
					}
				}
			} else {
				for k = n; k >= 1; k-- {
					for j = k + 1; j <= n; j++ {
						if a.Get(j-1, k-1) != zero {
							if noconj {
								temp = alpha * a.Get(j-1, k-1)
							} else {
								temp = alpha * a.GetConj(j-1, k-1)
							}
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*mx.Get(i-1, k-1))
							}
						}
					}
					temp = alpha
					if nounit {
						if noconj {
							temp *= a.Get(k-1, k-1)
						} else {
							temp *= a.GetConj(k-1, k-1)
						}
					}
					if temp != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, temp*mx.Get(i-1, k-1))
						}
					}
				}
			}
		}
	}

	return
}

// Trsm solves one of the matrix equations
//
//    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
//
// where alpha is a scalar, X and B are m by n matrices, A is a unit, or
// non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
//
//    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
//
// The matrix X is overwritten on B.
func (mx *CMatrix) Trsm(side MatSide, uplo MatUplo, transa MatTrans, diag MatDiag, m, n int, alpha complex128, a *CMatrix) (err error) {
	var lside, noconj, nounit, upper bool
	var one, temp, zero complex128
	var i, j, k, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	lside = side == Left
	if lside {
		nrowa = m
	} else {
		nrowa = n
	}
	noconj = transa == Trans
	nounit = diag == NonUnit
	upper = uplo == Upper

	if !side.IsValid() {
		err = fmt.Errorf("side invalid: %v", side.String())
	} else if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !transa.IsValid() {
		err = fmt.Errorf("transa invalid: %v", transa.String())
	} else if !diag.IsValid() {
		err = fmt.Errorf("diag invalid: %v", diag.String())
	} else if m < 0 {
		err = fmt.Errorf("m invalid: %v", m)
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if mx.Rows < max(1, m) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", mx.Rows, max(1, m))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				mx.Set(i-1, j-1, zero)
			}
		}
		return
	}

	//     Start the operations.
	if lside {
		if transa == NoTrans {
			//           Form  B := alpha*inv( A )*B.
			if upper {
				for j = 1; j <= n; j++ {
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, alpha*mx.Get(i-1, j-1))
						}
					}
					for k = m; k >= 1; k-- {
						if mx.Get(k-1, j-1) != zero {
							if nounit {
								mx.Set(k-1, j-1, mx.Get(k-1, j-1)/a.Get(k-1, k-1))
							}
							for i = 1; i <= k-1; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-mx.Get(k-1, j-1)*a.Get(i-1, k-1))
							}
						}
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, alpha*mx.Get(i-1, j-1))
						}
					}
					for k = 1; k <= m; k++ {
						if mx.Get(k-1, j-1) != zero {
							if nounit {
								mx.Set(k-1, j-1, mx.Get(k-1, j-1)/a.Get(k-1, k-1))
							}
							for i = k + 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-mx.Get(k-1, j-1)*a.Get(i-1, k-1))
							}
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*inv( A**T )*B
			//           or    B := alpha*inv( A**H )*B.
			if upper {
				for j = 1; j <= n; j++ {
					for i = 1; i <= m; i++ {
						temp = alpha * mx.Get(i-1, j-1)
						if noconj {
							for k = 1; k <= i-1; k++ {
								temp = temp - a.Get(k-1, i-1)*mx.Get(k-1, j-1)
							}
							if nounit {
								temp = temp / a.Get(i-1, i-1)
							}
						} else {
							for k = 1; k <= i-1; k++ {
								temp = temp - a.GetConj(k-1, i-1)*mx.Get(k-1, j-1)
							}
							if nounit {
								temp = temp / a.GetConj(i-1, i-1)
							}
						}
						mx.Set(i-1, j-1, temp)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = m; i >= 1; i-- {
						temp = alpha * mx.Get(i-1, j-1)
						if noconj {
							for k = i + 1; k <= m; k++ {
								temp = temp - a.Get(k-1, i-1)*mx.Get(k-1, j-1)
							}
							if nounit {
								temp = temp / a.Get(i-1, i-1)
							}
						} else {
							for k = i + 1; k <= m; k++ {
								temp = temp - a.GetConj(k-1, i-1)*mx.Get(k-1, j-1)
							}
							if nounit {
								temp = temp / a.GetConj(i-1, i-1)
							}
						}
						mx.Set(i-1, j-1, temp)
					}
				}
			}
		}
	} else {
		if transa == NoTrans {
			//           Form  B := alpha*B*inv( A ).
			if upper {
				for j = 1; j <= n; j++ {
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, alpha*mx.Get(i-1, j-1))
						}
					}
					for k = 1; k <= j-1; k++ {
						if a.Get(k-1, j-1) != zero {
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-a.Get(k-1, j-1)*mx.Get(i-1, k-1))
							}
						}
					}
					if nounit {
						temp = one / a.Get(j-1, j-1)
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, temp*mx.Get(i-1, j-1))
						}
					}
				}
			} else {
				for j = n; j >= 1; j-- {
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, alpha*mx.Get(i-1, j-1))
						}
					}
					for k = j + 1; k <= n; k++ {
						if a.Get(k-1, j-1) != zero {
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-a.Get(k-1, j-1)*mx.Get(i-1, k-1))
							}
						}
					}
					if nounit {
						temp = one / a.Get(j-1, j-1)
						for i = 1; i <= m; i++ {
							mx.Set(i-1, j-1, temp*mx.Get(i-1, j-1))
						}
					}
				}
			}
		} else {
			//           Form  B := alpha*B*inv( A**T )
			//           or    B := alpha*B*inv( A**H ).
			if upper {
				for k = n; k >= 1; k-- {
					if nounit {
						if noconj {
							temp = one / a.Get(k-1, k-1)
						} else {
							temp = one / a.GetConj(k-1, k-1)
						}
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, temp*mx.Get(i-1, k-1))
						}
					}
					for j = 1; j <= k-1; j++ {
						if a.Get(j-1, k-1) != zero {
							if noconj {
								temp = a.Get(j-1, k-1)
							} else {
								temp = a.GetConj(j-1, k-1)
							}
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-temp*mx.Get(i-1, k-1))
							}
						}
					}
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, alpha*mx.Get(i-1, k-1))
						}
					}
				}
			} else {
				for k = 1; k <= n; k++ {
					if nounit {
						if noconj {
							temp = one / a.Get(k-1, k-1)
						} else {
							temp = one / a.GetConj(k-1, k-1)
						}
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, temp*mx.Get(i-1, k-1))
						}
					}
					for j = k + 1; j <= n; j++ {
						if a.Get(j-1, k-1) != zero {
							if noconj {
								temp = a.Get(j-1, k-1)
							} else {
								temp = a.GetConj(j-1, k-1)
							}
							for i = 1; i <= m; i++ {
								mx.Set(i-1, j-1, mx.Get(i-1, j-1)-temp*mx.Get(i-1, k-1))
							}
						}
					}
					if alpha != one {
						for i = 1; i <= m; i++ {
							mx.Set(i-1, k-1, alpha*mx.Get(i-1, k-1))
						}
					}
				}
			}
		}
	}

	return
}

// Herk performs one of the hermitian rank k operations
//
//    C := alpha*A*A**H + beta*C,
//
// or
//
//    C := alpha*A**H*A + beta*C,
//
// where  alpha and beta  are  real scalars,  C is an  n by n  hermitian
// matrix and  A  is an  n by k  matrix in the  first case and a  k by n
// matrix in the second case.
func (mx *CMatrix) Herk(uplo MatUplo, trans MatTrans, n, k int, alpha float64, a *CMatrix, beta float64) (err error) {
	var upper bool
	var temp complex128
	var one, rtemp, zero float64
	var i, j, l, nrowa int

	one = 1.0
	zero = 0.0

	//     Test the input parameters.
	if trans == NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() || trans == Trans {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if mx.Rows < max(1, n) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", mx.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if upper {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						mx.SetRe(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j-1; i++ {
						mx.Set(i-1, j-1, complex(beta, 0)*mx.Get(i-1, j-1))
					}
					mx.SetRe(j-1, j-1, beta*real(mx.Get(j-1, j-1)))
				}
			}
		} else {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						mx.SetRe(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					mx.SetRe(j-1, j-1, beta*real(mx.Get(j-1, j-1)))
					for i = j + 1; i <= n; i++ {
						mx.Set(i-1, j-1, complex(beta, 0)*mx.Get(i-1, j-1))
					}
				}
			}
		}
		return
	}

	//     Start the operations.
	if trans == NoTrans {
		//        Form  C := alpha*A*A**H + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= j; i++ {
						mx.SetRe(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= j-1; i++ {
						mx.Set(i-1, j-1, complex(beta, 0)*mx.Get(i-1, j-1))
					}
					mx.SetRe(j-1, j-1, beta*real(mx.Get(j-1, j-1)))
				} else {
					mx.SetRe(j-1, j-1, real(mx.Get(j-1, j-1)))
				}
				for l = 1; l <= k; l++ {
					if a.Get(j-1, l-1) != complex(zero, 0) {
						temp = complex(alpha, 0) * a.GetConj(j-1, l-1)
						for i = 1; i <= j-1; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
						}
						mx.SetRe(j-1, j-1, real(mx.GetReCmplx(j-1, j-1))+real(temp*a.Get(i-1, l-1)))
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = j; i <= n; i++ {
						mx.SetRe(i-1, j-1, zero)
					}
				} else if beta != one {
					mx.SetRe(j-1, j-1, beta*real(mx.Get(j-1, j-1)))
					for i = j + 1; i <= n; i++ {
						mx.Set(i-1, j-1, complex(beta, 0)*mx.Get(i-1, j-1))
					}
				} else {
					mx.SetRe(j-1, j-1, real(mx.Get(j-1, j-1)))
				}
				for l = 1; l <= k; l++ {
					if a.Get(j-1, l-1) != complex(zero, 0) {
						temp = complex(alpha, 0) * a.GetConj(j-1, l-1)
						mx.SetRe(j-1, j-1, real(mx.Get(j-1, j-1))+real(temp*a.Get(j-1, l-1)))
						for i = j + 1; i <= n; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
						}
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*A**H*A + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j-1; i++ {
					temp = complex(zero, 0)
					for l = 1; l <= k; l++ {
						temp += a.GetConj(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, complex(alpha, 0)*temp)
					} else {
						mx.Set(i-1, j-1, complex(alpha, 0)*temp+complex(beta, 0)*mx.Get(i-1, j-1))
					}
				}
				rtemp = zero
				for l = 1; l <= k; l++ {
					rtemp += a.GetConjProd(l-1, j-1)
				}
				if beta == zero {
					mx.SetRe(j-1, j-1, alpha*rtemp)
				} else {
					mx.SetRe(j-1, j-1, alpha*rtemp+beta*real(mx.Get(j-1, j-1)))
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				rtemp = zero
				for l = 1; l <= k; l++ {
					rtemp += a.GetConjProd(l-1, j-1)
				}
				if beta == zero {
					mx.SetRe(j-1, j-1, alpha*rtemp)
				} else {
					mx.SetRe(j-1, j-1, alpha*rtemp+beta*real(mx.Get(j-1, j-1)))
				}
				for i = j + 1; i <= n; i++ {
					temp = complex(zero, 0)
					for l = 1; l <= k; l++ {
						temp += a.GetConj(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, complex(alpha, 0)*temp)
					} else {
						mx.Set(i-1, j-1, complex(alpha, 0)*temp+complex(beta, 0)*mx.Get(i-1, j-1))
					}
				}
			}
		}
	}

	return
}

// Syrk performs one of the symmetric rank k operations
//
//    C := alpha*A*A**T + beta*C,
//
// or
//
//    C := alpha*A**T*A + beta*C,
//
// where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
// and  A  is an  n by k  matrix in the first case and a  k by n  matrix
// in the second case.
func (mx *CMatrix) Syrk(uplo MatUplo, trans MatTrans, n, k int, alpha complex128, a *CMatrix, beta complex128) (err error) {
	var upper bool
	var one, temp, zero complex128
	var i, j, l, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if trans == NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() || trans == ConjTrans {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if mx.Rows < max(1, n) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", mx.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if upper {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
			}
		} else {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
			}
		}
		return
	}

	//     Start the operations.
	if trans == NoTrans {
		//        Form  C := alpha*A*A**T + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if a.Get(j-1, l-1) != zero {
						temp = alpha * a.Get(j-1, l-1)
						for i = 1; i <= j; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
						}
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if a.Get(j-1, l-1) != zero {
						temp = alpha * a.Get(j-1, l-1)
						for i = j; i <= n; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+temp*a.Get(i-1, l-1))
						}
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*A**T*A + beta*C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = j; i <= n; i++ {
					temp = zero
					for l = 1; l <= k; l++ {
						temp += a.Get(l-1, i-1) * a.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp)
					} else {
						mx.Set(i-1, j-1, alpha*temp+beta*mx.Get(i-1, j-1))
					}
				}
			}
		}
	}

	return
}

// Her2k performs one of the hermitian rank 2k operations
//
//    C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
//
// or
//
//    C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,
//
// where  alpha and beta  are scalars with  beta  real,  C is an  n by n
// hermitian matrix and  A and B  are  n by k matrices in the first case
// and  k by n  matrices in the second case.
func (mx *CMatrix) Her2k(uplo MatUplo, trans MatTrans, n, k int, alpha complex128, a, b *CMatrix, beta float64) (err error) {
	var upper bool
	var temp1, temp2, zero complex128
	var one float64
	var i, j, l, nrowa int

	one = 1.0
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if trans == NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() || trans == Trans {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, nrowa) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, nrowa))
	} else if mx.Rows < max(1, n) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", mx.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if upper {
			if beta == real(zero) {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j-1; i++ {
						mx.Set(i-1, j-1, complex(beta, 0)*mx.Get(i-1, j-1))
					}
					mx.SetRe(j-1, j-1, beta*real(mx.Get(j-1, j-1)))
				}
			}
		} else {
			if beta == real(zero) {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					mx.SetRe(j-1, j-1, beta*real(mx.Get(j-1, j-1)))
					for i = j + 1; i <= n; i++ {
						mx.Set(i-1, j-1, complex(beta, 0)*mx.Get(i-1, j-1))
					}
				}
			}
		}
		return
	}

	//     Start the operations.
	if trans == NoTrans {
		//        Form  C := alpha*A*B**H + conjg( alpha )*B*A**H +
		//                   C.
		if upper {
			for j = 1; j <= n; j++ {
				if beta == real(zero) {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= j-1; i++ {
						mx.Set(i-1, j-1, complex(beta, 0)*mx.Get(i-1, j-1))
					}
					mx.SetRe(j-1, j-1, beta*real(mx.Get(j-1, j-1)))
				} else {
					mx.Set(j-1, j-1, mx.GetReCmplx(j-1, j-1))
				}
				for l = 1; l <= k; l++ {
					if (a.Get(j-1, l-1) != zero) || (b.Get(j-1, l-1) != zero) {
						temp1 = alpha * b.GetConj(j-1, l-1)
						temp2 = cmplx.Conj(alpha * a.Get(j-1, l-1))
						for i = 1; i <= j-1; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+a.Get(i-1, l-1)*temp1+b.Get(i-1, l-1)*temp2)
						}
						mx.SetRe(j-1, j-1, real(mx.Get(j-1, j-1))+real(a.Get(j-1, l-1)*temp1+b.Get(j-1, l-1)*temp2))
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if beta == real(zero) {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = j + 1; i <= n; i++ {
						mx.Set(i-1, j-1, complex(beta, 0)*mx.Get(i-1, j-1))
					}
					mx.SetRe(j-1, j-1, beta*real(mx.Get(j-1, j-1)))
				} else {
					mx.Set(j-1, j-1, mx.GetReCmplx(j-1, j-1))
				}
				for l = 1; l <= k; l++ {
					if (a.Get(j-1, l-1) != zero) || (b.Get(j-1, l-1) != zero) {
						temp1 = alpha * b.GetConj(j-1, l-1)
						temp2 = cmplx.Conj(alpha * a.Get(j-1, l-1))
						for i = j + 1; i <= n; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+a.Get(i-1, l-1)*temp1+b.Get(i-1, l-1)*temp2)
						}
						mx.SetRe(j-1, j-1, real(mx.Get(j-1, j-1))+real(a.Get(j-1, l-1)*temp1+b.Get(j-1, l-1)*temp2))
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*A**H*B + conjg( alpha )*B**H*A +
		//                   C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					temp1 = zero
					temp2 = zero
					for l = 1; l <= k; l++ {
						temp1 = temp1 + a.GetConj(l-1, i-1)*b.Get(l-1, j-1)
						temp2 = temp2 + b.GetConj(l-1, i-1)*a.Get(l-1, j-1)
					}
					if i == j {
						if beta == real(zero) {
							mx.SetRe(j-1, j-1, real(alpha*temp1+cmplx.Conj(alpha)*temp2))
						} else {
							mx.SetRe(j-1, j-1, beta*real(mx.Get(j-1, j-1))+real(alpha*temp1+cmplx.Conj(alpha)*temp2))
						}
					} else {
						if beta == real(zero) {
							mx.Set(i-1, j-1, alpha*temp1+cmplx.Conj(alpha)*temp2)
						} else {
							mx.Set(i-1, j-1, complex(beta, 0)*mx.Get(i-1, j-1)+alpha*temp1+cmplx.Conj(alpha)*temp2)
						}
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = j; i <= n; i++ {
					temp1 = zero
					temp2 = zero
					for l = 1; l <= k; l++ {
						temp1 = temp1 + a.GetConj(l-1, i-1)*b.Get(l-1, j-1)
						temp2 = temp2 + b.GetConj(l-1, i-1)*a.Get(l-1, j-1)
					}
					if i == j {
						if beta == real(zero) {
							mx.SetRe(j-1, j-1, real(alpha*temp1+cmplx.Conj(alpha)*temp2))
						} else {
							mx.SetRe(j-1, j-1, beta*real(mx.Get(j-1, j-1))+real(alpha*temp1+cmplx.Conj(alpha)*temp2))
						}
					} else {
						if beta == real(zero) {
							mx.Set(i-1, j-1, alpha*temp1+cmplx.Conj(alpha)*temp2)
						} else {
							mx.Set(i-1, j-1, complex(beta, 0)*mx.Get(i-1, j-1)+alpha*temp1+cmplx.Conj(alpha)*temp2)
						}
					}
				}
			}
		}
	}

	return
}

// Syr2k performs one of the symmetric rank 2k operations
//
//    C := alpha*A*B**T + alpha*B*A**T + beta*C,
//
// or
//
//    C := alpha*A**T*B + alpha*B**T*A + beta*C,
//
// where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
// and  A and B  are  n by k  matrices  in the  first  case  and  k by n
// matrices in the second case.
func (mx *CMatrix) Syr2k(uplo MatUplo, trans MatTrans, n, k int, alpha complex128, a, b *CMatrix, beta complex128) (err error) {
	var upper bool
	var one, temp1, temp2, zero complex128
	var i, j, l, nrowa int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	if trans == NoTrans {
		nrowa = n
	} else {
		nrowa = k
	}
	upper = uplo == Upper

	if !uplo.IsValid() {
		err = fmt.Errorf("uplo invalid: %v", uplo.String())
	} else if !trans.IsValid() || trans == ConjTrans {
		err = fmt.Errorf("trans invalid: %v", trans.String())
	} else if n < 0 {
		err = fmt.Errorf("n invalid: %v", n)
	} else if k < 0 {
		err = fmt.Errorf("k invalid: %v", k)
	} else if a.Rows < max(1, nrowa) {
		err = fmt.Errorf("a.Rows invalid: %v < %v", a.Rows, max(1, nrowa))
	} else if b.Rows < max(1, nrowa) {
		err = fmt.Errorf("b.Rows invalid: %v < %v", b.Rows, max(1, nrowa))
	} else if mx.Rows < max(1, n) {
		err = fmt.Errorf("c.Rows invalid: %v < %v", mx.Rows, max(1, n))
	}
	if err != nil {
		return
	}

	//     Quick return if possible.
	if (n == 0) || (((alpha == zero) || (k == 0)) && (beta == one)) {
		return
	}

	//     And when  alpha.eq.zero.
	if alpha == zero {
		if upper {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
			}
		} else {
			if beta == zero {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, zero)
					}
				}
			} else {
				for j = 1; j <= n; j++ {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
			}
		}
		return
	}

	//     Start the operations.
	if trans == NoTrans {
		//        Form  C := alpha*A*B**T + alpha*B*A**T + C.
		if upper {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = 1; i <= j; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if (a.Get(j-1, l-1) != zero) || (b.Get(j-1, l-1) != zero) {
						temp1 = alpha * b.Get(j-1, l-1)
						temp2 = alpha * a.Get(j-1, l-1)
						for i = 1; i <= j; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+a.Get(i-1, l-1)*temp1+b.Get(i-1, l-1)*temp2)
						}
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if beta == zero {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, zero)
					}
				} else if beta != one {
					for i = j; i <= n; i++ {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1))
					}
				}
				for l = 1; l <= k; l++ {
					if (a.Get(j-1, l-1) != zero) || (b.Get(j-1, l-1) != zero) {
						temp1 = alpha * b.Get(j-1, l-1)
						temp2 = alpha * a.Get(j-1, l-1)
						for i = j; i <= n; i++ {
							mx.Set(i-1, j-1, mx.Get(i-1, j-1)+a.Get(i-1, l-1)*temp1+b.Get(i-1, l-1)*temp2)
						}
					}
				}
			}
		}
	} else {
		//        Form  C := alpha*A**T*B + alpha*B**T*A + C.
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j; i++ {
					temp1 = zero
					temp2 = zero
					for l = 1; l <= k; l++ {
						temp1 = temp1 + a.Get(l-1, i-1)*b.Get(l-1, j-1)
						temp2 = temp2 + b.Get(l-1, i-1)*a.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp1+alpha*temp2)
					} else {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+alpha*temp1+alpha*temp2)
					}
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				for i = j; i <= n; i++ {
					temp1 = zero
					temp2 = zero
					for l = 1; l <= k; l++ {
						temp1 = temp1 + a.Get(l-1, i-1)*b.Get(l-1, j-1)
						temp2 = temp2 + b.Get(l-1, i-1)*a.Get(l-1, j-1)
					}
					if beta == zero {
						mx.Set(i-1, j-1, alpha*temp1+alpha*temp2)
					} else {
						mx.Set(i-1, j-1, beta*mx.Get(i-1, j-1)+alpha*temp1+alpha*temp2)
					}
				}
			}
		}
	}

	return
}

func newCMatrixData(r, c int) []complex128 {
	return make([]complex128, r*c)
}

func NewCMatrix(r, c int, opts ...MatOpts) *CMatrix {
	var o MatOpts

	if opts != nil {
		o = opts[0]
	} else {
		o = NewMatOpts()
	}
	return &CMatrix{Rows: r, Cols: c, Opts: o, Data: newCMatrixData(r, c)}
}

func CMatrixFactory() func(int, int, ...MatOpts) *CMatrix {
	return func(r, c int, opts ...MatOpts) *CMatrix {
		return NewCMatrix(r, c, opts...)
	}
}

func CMatrixDataFactory() func(*CMatrix, []complex128) *CMatrix {
	return func(m *CMatrix, d []complex128) *CMatrix {
		out := m.DeepCopy()
		out.Data = d
		return out
	}
}
